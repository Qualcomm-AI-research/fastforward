# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # GPTQ Quantization of Llama
#
# This notebook demonstrates how to apply GPTQ quantization to a Llama model using
# the FastForward orchestration framework. It reproduces the default GPTQ setup from
# [IST-DASLab/gptq](https://github.com/IST-DASLab/gptq):
#
# ```
# python llama.py LLAMA_HF_FOLDER c4 --wbits 4 --true-sequential --act-order --new-eval
# ```
#
# ### Requirements
#
# ```
# pip install transformers datasets
# ```
#
# ## Configuration

# +
import functools
import logging
import os
import random

from typing import Protocol

import datasets
import fastforward as ff
import torch

from fastforward._orchestration import graph_module
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from doc_helpers.llama_graph_module import to_graph_module
from doc_helpers.quick_start.quantized_models import (
    quantized_llama as quantized_llama,  # noqa: F401
)

model_name = os.environ.get("FF_GPTQ_MODEL", "huggyllama/llama-7b")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO)


# -

# ## GPTQ Parameters
#
# This section assumes familiarity with the GPTQ algorithm. For details, see
# [Frantar et al., 2022](https://arxiv.org/abs/2210.17323).
#
# - `num_bits`: Bit-width for weight quantization.
# - `symmetric`: Whether to use symmetric or asymmetric quantization.
# - `granularity`: Controls how many scale/offset parameters are used per weight matrix.
#   Finer granularity means more parameters and typically better perplexity at the cost of
#   storage overhead.
# - `block_size`: Number of columns processed per GPTQ block iteration.
# - `perc_damp`: Dampening factor as a percentage of the mean Hessian diagonal, used to
#   stabilize the Cholesky inversion.
# - `act_order`: Whether to reorder columns by activation magnitude before quantization.
#
# For `granularity`, the following results were obtained on Llama-7B (W4, act-order, C4
# calibration, WikiText-2 evaluation):
#
# | Granularity | Tile size | #scales | Perplexity |
# |---|---|---|---|
# | `ff.PerChannel(channel_dim=0)` | (1, 4096) | 4,096 | 6.09 |
# | `ff.PerTile((32, 32))` | (32, 32) | 16,384 | 5.92 |
# | `ff.PerBlock(block_dims=1, block_sizes=128, per_channel_dims=0)` | (1, 128) | 131,072 | 5.79 |
#

# +
num_bits = 4
symmetric = False

granularity: ff.granularity.Granularity = ff.PerChannel(channel_dim=0)

block_size = 128
perc_damp = 0.01
act_order = True
# -

# ## Load and Prepare Model
#
# Load the Llama model, convert it to a quantization-ready model with `ff.quantize_model`,
# and initialize weight quantizers for all attention and MLP projections.

# +
model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype="auto")
assert isinstance(model, LlamaForCausalLM)

ff.quantize_model(model)

w_quantizers = ff.find_quantizers(model, "**/layers/*/self_attn/*/[quantizer:parameter/weight]")
w_quantizers |= ff.find_quantizers(model, "**/layers/*/mlp/*/[quantizer:parameter/weight]")
w_quantizers.initialize(
    ff.nn.LinearQuantizer, num_bits=num_bits, granularity=granularity, symmetric=symmetric
)

graph = to_graph_module(model)
model.to(device)
# -

# ## Load Calibration Dataset
#
# C4 was used as the calibration set in the original GPTQ paper. The loader below is adapted
# from [datautils.py](https://github.com/IST-DASLab/gptq/blob/main/datautils.py#L54-L100).


class TokenizedData(Protocol):
    """Huggingface tokenizers returns an object with input_ids."""

    input_ids: torch.Tensor


# +
def get_c4(
    model: str,
    sequence_length: int,
    seed: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Load a subset of C4 as a calibration set."""
    traindata = datasets.load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

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
        loader.append(tmp.input_ids[:, i:j].to(device))
    return loader


calibration_set = get_c4(model_name, sequence_length=2048, seed=0, device=device)
# -

# ## Run GPTQ Optimization
#
# Each attention projection (Q, K, V, O) and MLP projection (gate, up, down) is quantized
# individually in sequential order - equivalent to `--true-sequential` in the original GPTQ.
# Each layer is wrapped in a `SubgraphSpec` that tells the `LocalOptimizer` which subgraph
# to pass to `ff.quantization.gptq`.

# +
gptq_fn = functools.partial(
    ff.quantization.gptq, block_size=block_size, perc_damp=perc_damp, actorder=act_order
)

projection_names = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

specs = []
for proj_name in projection_names:
    for match in ff.mpath.search(f"*/{proj_name}", graph):
        layer_ref = graph.node_ref(match.module)
        specs.append(graph_module.SubgraphSpec(input=layer_ref, output=layer_ref, fn=gptq_fn))

optimizer = graph_module.LocalOptimizer(graph, specs)

with torch.inference_mode(), ff.strict_quantization(False):
    optimizer.optimize(calibration_set)
# -

# ## Evaluate on WikiText-2
#
# Perplexity is computed on WikiText-2, following the evaluation protocol from the original paper.
# The loader below is adapted from
# [datautils.py](https://github.com/IST-DASLab/gptq/blob/main/datautils.py#L10-L30).
# The original GPTQ implementation reports **6.09 perplexity** for this configuration
# (Llama-7B, W4, per-channel, act-order, C4 calibration, WikiText-2 evaluation).


# +
def get_wikitext2(
    model: str,
    nsamples: int,
    sequence_length: int,
    seed: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Build a WikiText-2 validation set for perplexity evaluation."""
    testdata = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    testenc: TokenizedData = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    random.seed(seed)
    sequences: list[torch.Tensor] = []
    for _ in range(nsamples):
        i = random.randint(0, testenc.input_ids.shape[1] - sequence_length - 1)
        j = i + sequence_length
        sequences.append(testenc.input_ids[:, i:j].to(device))
    return sequences


@torch.no_grad()
def evaluate(model: LlamaForCausalLM, dataset: list[torch.Tensor]) -> float:
    """Compute perplexity on a dataset of token-id sequences."""
    model_device = next(model.parameters()).device
    nll_sum = 0.0
    total_tokens = 0

    for batch in dataset:
        batch = batch.to(model_device)
        out: CausalLMOutputWithPast = model(input_ids=batch)
        shift_logits = out.logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
        loss: torch.Tensor = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        nll_sum += float(loss.item())
        total_tokens += shift_labels.numel()

    return float(torch.exp(torch.tensor(nll_sum / total_tokens)))


validation_set = get_wikitext2(
    model_name, nsamples=128, sequence_length=2048, seed=0, device=device
)
with ff.strict_quantization(False):
    perplexity = evaluate(model, validation_set)

print(f"Wiki2 PPL 4bit-GPTQ LLaMA-7B: {perplexity:.4f}  (original paper: 6.09)")
# -

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#
# This file includes code derived from https://github.com/IST-DASLab/gptq
# Licensed under the Apache License, Version 2.0.
