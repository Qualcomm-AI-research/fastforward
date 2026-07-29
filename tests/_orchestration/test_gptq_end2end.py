# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#
# This file includes code derived from https://github.com/IST-DASLab/gptq
# Licensed under the Apache License, Version 2.0.

# pylint: disable=missing-function-docstring

import functools
import importlib.util
import logging
import math
import random

from typing import Protocol

import fastforward as ff
import pytest
import torch

from fastforward._orchestration.instruction_engine import OffloadEverything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

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


@torch.inference_mode()
def _evaluate(model: torch.nn.Module, dataset: list[torch.Tensor], device: torch.device) -> float:
    """Compute perplexity on a dataset of token-id sequences."""
    original_device = next(model.parameters()).device
    model.to(device)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
    nll_sum = 0.0
    total_tokens = 0

    for batch in dataset:
        batch = batch.to(device)
        logits = model(batch, use_cache=False).logits
        shift_logits = logits[:, :-1].reshape(-1, logits.size(-1))
        shift_labels = batch[:, 1:].reshape(-1)
        nll_sum += loss_fct(shift_logits, shift_labels).item()
        total_tokens += batch.size(1) - 1

    model.to(original_device)
    return math.exp(nll_sum / total_tokens)


@skip_without_datasets_and_transformers
def test_gptq_layerwise_optimize_perplexity() -> None:
    """GPTQ W4 quantization of Llama-3.2-1B-Instruct with act-order, evaluated on WikiText-2.

    As a reference, on WikiText-2 for the Llama-3.2-1B-Instruct model:
    - No quantization scores ~15 perplexity
    - Min-Max quantization at 4 bits scores ~280 perplexity
    - GPTQ quantization at 4 bits scores ~36 perplexity.

    This workload tracks the literature: the original GPTQ paper reports 6.09
    perplexity for 4-bit Llama-1-7B, which this pipeline can reproduce.
    """
    from docs.examples.doc_helpers import quantized_llama as quantized_llama
    from docs.examples.doc_helpers.quantized_llama import QuantizedLlamaSDPAttention
    from transformers import LlamaForCausalLM
    from transformers.models.llama.modeling_llama import LlamaAttention

    # GIVEN a pre-trained model that we quantize with reasonable settings for GPTQ
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_bits = 4
    symmetric = False
    granularity: ff.granularity.Granularity = ff.PerChannel(channel_dim=0)
    block_size = 128
    perc_damp = 0.01
    act_order = True

    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        use_cache=False,
    )
    model.eval()

    ff.quantize_model(model, extra_conversion={LlamaAttention: QuantizedLlamaSDPAttention})

    # WHEN we select all decoder-layer QuantizedLinear layers as optimization targets
    quant_targets = ff.mpath.query("**/layers/**/[cls:ff.nn.QuantizedLinear]")
    w_quantizers = ff.find_quantizers(model, quant_targets / "[quantizer:parameter/weight]")
    w_quantizers.initialize(
        ff.nn.LinearQuantizer, num_bits=num_bits, granularity=granularity, symmetric=symmetric
    )

    # WHEN running GPTQ Using the orchestration framework on a calibration set
    calibration_set = _get_c4(model_name, sequence_length=2048, seed=0)
    with torch.inference_mode(), ff.strict_quantization(False):
        gptq_fn = functools.partial(
            ff.quantization.gptq, block_size=block_size, perc_damp=perc_damp, actorder=act_order
        )
        offloading = OffloadEverything(compute_device=device, storage_device=torch.device("cpu"))
        ff.layerwise_optimize(
            model, calibration_set, gptq_fn, targets=quant_targets, offloading=offloading
        )

    # WHEN running perplexity calculation on a validation set
    validation_set = _get_wikitext2(model_name, nsamples=128, sequence_length=2048, seed=0)
    with ff.strict_quantization(False):
        perplexity = _evaluate(model, validation_set, device)

    # THEN the perplexity is expected to be around 36.
    print(f"Wiki2 PPL 4bit-GPTQ Llama-3.2-1B-Instruct: {perplexity:.4f} (expected ~36)")
    assert perplexity < 50, f"Perplexity {perplexity:.4f} exceeds threshold (expected < 50)"


@skip_without_datasets_and_transformers
def test_qwen3_w4_gptq_autoquant_gs32_perplexity() -> None:
    """W4 min-max and GPTQ perplexity for Qwen3-1.7B via ff.autoquantize.

    Uses the on-device quantization group size (32) and the reference GPTQ recipe
    (block_size=128, perc_damp=0.05, act_order=False). No hand-written quantized
    modules — autoquant generates them from the FP model. The min-max baseline runs
    at the same granularity so the GPTQ improvement can be read directly.

    Prints, in order: FP32 baseline, W4 min-max, W4 GPTQ.
    """
    from transformers import Qwen3ForCausalLM

    model_name = "Qwen/Qwen3-1.7B"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_bits = 4
    symmetric = True
    granularity: ff.granularity.Granularity = ff.granularity.PerBlock(
        block_dims=1, block_sizes=32, per_channel_dims=0
    )
    block_size = 128
    perc_damp = 0.05
    act_order = False

    validation_set = _get_wikitext2(model_name, nsamples=128, sequence_length=2048, seed=0)
    calibration_set = _get_c4(model_name, sequence_length=2048, seed=0)

    # FP32 baseline
    fp_model = Qwen3ForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, use_cache=False
    )
    fp_model.eval()
    fp_perplexity = _evaluate(fp_model, validation_set, device)
    del fp_model

    # W4 min-max — same autoquant path and granularity as GPTQ below so the two are
    # directly comparable
    minmax_model: torch.nn.Module = Qwen3ForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, use_cache=False
    )
    minmax_model.eval()
    ff.autoquantize(
        minmax_model,
        output_path="_autoquantized_qwen_benchmark.py",
        force_overwrite=True,
        auto_import=True,
    )
    ff.quantize_model(minmax_model, skip_quantized_modules=True)
    minmax_targets = ff.mpath.query("**/layers/**/[cls:ff.nn.QuantizedLinear]")
    minmax_w_quantizers = ff.find_quantizers(
        minmax_model, minmax_targets / "[quantizer:parameter/weight]"
    )
    minmax_w_quantizers.initialize(
        ff.nn.LinearQuantizer, num_bits=num_bits, granularity=granularity, symmetric=symmetric
    )
    with torch.inference_mode(), ff.strict_quantization(False):
        minmax_model.to(device)
        with ff.estimate_ranges(minmax_model, ff.range_setting.smoothed_minmax):
            for batch in calibration_set:
                minmax_model(batch.to(device), use_cache=False)
        minmax_perplexity = _evaluate(minmax_model, validation_set, device)
    del minmax_model

    # W4 GPTQ
    gptq_model = Qwen3ForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, use_cache=False
    )
    gptq_model.eval()
    ff.autoquantize(
        gptq_model,
        output_path="_autoquantized_qwen_benchmark.py",
        force_overwrite=True,
        auto_import=True,
    )
    ff.quantize_model(gptq_model, skip_quantized_modules=True)
    gptq_targets = ff.mpath.query("**/layers/**/[cls:ff.nn.QuantizedLinear]")
    gptq_w_quantizers = ff.find_quantizers(
        gptq_model, gptq_targets / "[quantizer:parameter/weight]"
    )
    gptq_w_quantizers.initialize(
        ff.nn.LinearQuantizer, num_bits=num_bits, granularity=granularity, symmetric=symmetric
    )
    with torch.inference_mode(), ff.strict_quantization(False):
        gptq_fn = functools.partial(
            ff.quantization.gptq, block_size=block_size, perc_damp=perc_damp, actorder=act_order
        )
        offloading = OffloadEverything(compute_device=device, storage_device=torch.device("cpu"))
        ff.layerwise_optimize(
            gptq_model, calibration_set, gptq_fn, targets=gptq_targets, offloading=offloading
        )
    with ff.strict_quantization(False):
        gptq_perplexity = _evaluate(gptq_model, validation_set, device)

    print(f"Wiki2 PPL Qwen3-1.7B FP32 baseline:           {fp_perplexity:.4f}")
    print(f"Wiki2 PPL Qwen3-1.7B W4 min-max  (qgroup=32): {minmax_perplexity:.4f}")
    print(f"Wiki2 PPL Qwen3-1.7B W4 GPTQ     (qgroup=32): {gptq_perplexity:.4f}")
