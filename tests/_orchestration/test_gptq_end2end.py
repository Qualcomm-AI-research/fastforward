# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#
# This file includes code derived from https://github.com/IST-DASLab/gptq
# Licensed under the Apache License, Version 2.0.

# pylint: disable=missing-function-docstring

"""End-to-end GPTQ quantization benchmark on Llama.

Reproduces the default GPTQ setup from IST-DASLab/gptq:
    python llama.py LLAMA_HF_FOLDER c4 --wbits 4 --true-sequential --act-order --new-eval
"""

import functools
import importlib.util
import math
import os
import random

from typing import Protocol

import fastforward as ff
import pytest
import torch

from fastforward._orchestration import graph_module
from fastforward._orchestration.graph_module import inference_mode, local_optimize
from fastforward._orchestration.instruction_engine import OffloadEverything
from fastforward._orchestration.trace import trace

pytestmark = pytest.mark.benchmark


skip_without_datasets_and_transformers = pytest.mark.skipif(
    importlib.util.find_spec("datasets") is None
    or importlib.util.find_spec("transformers") is None,
    reason="requires `datasets` and `transformers` (install with `[docs]` extra)",
)


class TokenizedData(Protocol):
    input_ids: torch.Tensor


def _get_c4(model_name: str, sequence_length: int, seed: int) -> list[torch.Tensor]:
    """Load a subset of C4 as a calibration set."""
    import datasets  # type: ignore[import-untyped]

    from transformers import AutoTokenizer

    traindata = datasets.load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    random.seed(seed)
    loader = []
    for _ in range(128):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tmp: TokenizedData = tokenizer(
                traindata[i]["text"],
                return_tensors="pt",
                truncation=True,
                max_length=sequence_length * 2,
            )
            if tmp.input_ids.shape[1] >= sequence_length:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - sequence_length - 1)
        j = i + sequence_length
        loader.append(tmp.input_ids[:, i:j])
    return loader


def _get_wikitext2(
    model_name: str, nsamples: int, sequence_length: int, seed: int
) -> list[torch.Tensor]:
    """Build a WikiText-2 validation set for perplexity evaluation."""
    import datasets

    from transformers import AutoTokenizer

    testdata = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    testenc: TokenizedData = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    random.seed(seed)
    sequences: list[torch.Tensor] = []
    for _ in range(nsamples):
        i = random.randint(0, testenc.input_ids.shape[1] - sequence_length - 1)
        j = i + sequence_length
        sequences.append(testenc.input_ids[:, i:j])
    return sequences


def _evaluate(
    model: graph_module.GraphModule, dataset: list[torch.Tensor], device: torch.device
) -> float:
    """Compute perplexity on a dataset of token-id sequences."""
    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
    nll_sum = 0.0
    total_tokens = 0

    for batch in dataset:
        logits = model(batch, use_cache=False).logits
        nll_sum += loss_fct(
            logits[:, :-1].reshape(-1, logits.size(-1)).to(device),
            batch[:, 1:].reshape(-1).to(device),
        ).item()
        total_tokens += batch.size(1) - 1

    return math.exp(nll_sum / total_tokens)


@skip_without_datasets_and_transformers
def test_gptq_llama_7b_perplexity() -> None:
    """GPTQ W4 quantization of Llama-7B with act-order, evaluated on WikiText-2.

    Expected perplexity: ~6.09 (per-channel, from the original GPTQ paper).
    """
    from transformers import LlamaForCausalLM

    # Given: configuration matching the original GPTQ paper
    model_name = os.environ.get("FF_GPTQ_MODEL", "huggyllama/llama-7b")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_bits = 4
    symmetric = False
    granularity: ff.granularity.Granularity = ff.PerChannel(channel_dim=0)
    block_size = 128
    perc_damp = 0.01
    act_order = True

    # When: we load, trace, quantize, and run GPTQ optimization
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    assert isinstance(model, LlamaForCausalLM)

    example_input = torch.randint(0, model.config.vocab_size, (1, 2048))
    graph = trace(model, example_input, use_cache=False)

    ff.autoquantize(
        model, output_path="llama_7b_autoquant.py", auto_import=True, force_overwrite=True
    )
    ff.quantize_model(model)

    w_quantizers = ff.find_quantizers(model, "**/layers/*/self_attn/*/[quantizer:parameter/weight]")
    w_quantizers |= ff.find_quantizers(model, "**/layers/*/mlp/*/[quantizer:parameter/weight]")
    w_quantizers.initialize(
        ff.nn.LinearQuantizer, num_bits=num_bits, granularity=granularity, symmetric=symmetric
    )

    gptq_fn = functools.partial(
        ff.quantization.gptq, block_size=block_size, perc_damp=perc_damp, actorder=act_order
    )

    projection_names = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    specs = []
    for proj_name in projection_names:
        for match in ff.mpath.search(f"**/{proj_name}", graph):
            specs.append(graph_module.SubgraphSpec(region=match.module, fn=gptq_fn))

    offloading = OffloadEverything(compute_device=device, storage_device=torch.device("cpu"))

    calibration_set = _get_c4(model_name, sequence_length=2048, seed=0)

    with torch.inference_mode(), ff.strict_quantization(False):
        with local_optimize(graph, specs, offloading_strategy=offloading):
            graph(calibration_set)

    # Then: perplexity on WikiText-2 should be close to the original paper's 6.09
    validation_set = _get_wikitext2(model_name, nsamples=128, sequence_length=2048, seed=0)
    with inference_mode(graph, offloading_strategy=offloading), ff.strict_quantization(False):
        perplexity = _evaluate(graph, validation_set, device)

    print(f"Wiki2 PPL 4bit-GPTQ LLaMA-7B: {perplexity:.4f}  (original paper: 6.09)")
    assert perplexity < 6.5, f"Perplexity {perplexity:.4f} exceeds threshold (expected < 6.5)"
