# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# # AutoQuant: Quantizing Llama 3.2
#
# This tutorial walks through `ff.autoquantize`, FastForward's automated quantization workflow,
# applied to Meta's Llama 3.2 1B Instruct model pulled from HuggingFace.
#
# We will:
#
# 1. Load the floating-point model together with a calibration / evaluation dataset.
# 2. Establish a perplexity baseline on the FP model.
# 3. Run AutoQuant to generate a quantization-ready version of the model source code.
# 4. Configure quantizers for weights and activations.
# 5. Calibrate the quantizers and measure the resulting perplexity.
#
# We start by downloading the model and tokenizer through the HuggingFace API.

# %%
import os

import fastforward as ff
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, LlamaForCausalLM, default_data_collator

model_name_or_path = os.environ.get("FF_QUICKSTART_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
model_dtype = torch.bfloat16
device = "cuda"

model = LlamaForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name_or_path,
    torch_dtype=model_dtype,
    attn_implementation="eager",
    from_tf=False,
)
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_name_or_path, attn_implementation="eager"
)


# %% [markdown]
# We will also need some data:
# - A *validation-set*  for **evaluating** perplexity before and after quantization.
# - A *calibration-set* for **calibrating** the quantizers, i.e. running enough forward passes to
#  **estimate the quantizers ranges** and set quantization parameters accordingly.
#
# We use WikiText-2, where the training split feeds calibration, the validation split is used for evaluation.

# %%
from torch.utils.data import DataLoader

from doc_helpers.data_utils import tokenize_dataset

sequence_length = 1024
batch_size = 1

# Load Dataset
raw_validset = load_dataset("wikitext", "wikitext-2-v1", split="train")
raw_trainset = load_dataset("wikitext", "wikitext-2-v1", split="validation")

# Tokenize Dataset
tokenized_validset = tokenize_dataset(raw_validset, tokenizer, sequence_length)
tokenized_trainset = tokenize_dataset(raw_trainset, tokenizer, sequence_length)

# Create Dataloader
valid_loader = DataLoader(tokenized_validset, batch_size, collate_fn=default_data_collator)
train_loader = DataLoader(tokenized_trainset, batch_size, collate_fn=default_data_collator)


# %% [markdown]
# ---
# # Evaluate the FP Model
#
# Before quantizing anything, we measure the floating-point model's perplexity on the validation split.
# This is the reference number that the quantized model will be compared against.

# %%
from doc_helpers.data_utils import sliced_tqdm


def prepare_batch(batch: dict, device: torch.device):
    return {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "labels": batch["labels"].to(torch.long).to(device),
    }


@torch.no_grad()
def evaluate_model(model, valid_loader, device, limit=None):
    model.eval()
    losses = []
    for batch in sliced_tqdm(valid_loader, limit):
        batch = prepare_batch(batch, device)
        outputs = model(**batch)
        losses.append(outputs.loss)

    eval_loss = torch.stack(losses).mean()
    perplexity = torch.exp(eval_loss)
    return float(perplexity)


model.to(device)
fp_task_results = evaluate_model(model, valid_loader, device, limit=8)
print(fp_task_results)

# %% [markdown]
# ---
# # AutoQuant: Generate the Quantization-Ready Source Code
#
# A single call to `ff.autoquantize` inspects the model and emits a Python file containing
# quantization-ready versions of every module it encounters.
# The generated code is a drop-in replacement: each module exposes the same interface as the original,
# but with explicit quantizer stubs ready to be configured.
#
# The output is regular python source file; you can read it, edit it, and check it into version control like any other source file.

# %%
print(f"Autoquantize model {type(model).__name__}")

code = ff.autoquantize(
    model,
    output_path="_autoquantized_llama.py",
    force_overwrite=True,
    auto_import=True,  # immediately import the generated code
)


# %% [markdown]
# ---
# # Create the Quantization-Ready Model
#
# With the generated module imported, `ff.quantize_model` rewrites the original model in place:
# every layer that has a quantized counterpart is swapped in.
# Functionally the model still behaves like the FP version because the quantizer stubs are inactive —
# they only start operating once we initialize them in the next step.
#
#
# Note:
#  - To use autoquantized model, you must import the quantization-ready modules from the generated file.
#  - If you just run `ff.autoquantize` with `auto_import=True`, you don't need to explicitly import.

# %%
from _autoquantized_llama import QuantizedLlamaForCausalLM

# Transform the original model into a quantized-ready using the imported quantized-modules
ff.quantize_model(model, skip_quantized_modules=True)

# %%
# OPTIONAL: you can cast the model to QuantizedLlamaForCausalLM to help LSP or IDE
from typing import cast

model: QuantizedLlamaForCausalLM = cast(QuantizedLlamaForCausalLM, model)


# %% [markdown]
# ---
# # Initialize the Quantizers
#
# The model now contains placeholder quantizer stubs at every site where a quantizer could be inserted.
# We decide which stubs to activate, and with what configuration.
#
# Quantizers are selected with `ff.find_quantizers`, which uses a glob-style pattern over fully-qualified module names.
# The `[quantizer:...]` filter narrows the match by quantizer kind (`parameter` vs `activation`) and target (`weight`, `output`, ...).
#
# We group quantizers by role so each group can receive its own configuration:
#
# - **Linear weight quantizers** in attention and MLP layers, plus `lm_head` and `embed_tokens`: 4-bit, per-channel.
# - **LayerNorm weight quantizers**: 4-bit per-tensor (per-channel does not apply — the weight is 1D).
# - **Activation quantizers** across the transformer blocks and residual paths: 16-bit, asymmetric.
#
# Tweaking these settings is how the accuracy / efficiency trade-off is dialed in.

# %%
from fastforward.nn import DynamicLinearQuantizer, LinearQuantizer

# Granularities
per_block_32 = ff.granularity.PerBlock(block_dims=1, block_sizes=32, per_channel_dims=0)
per_tensor = ff.granularity.PerTensor()

# Model weight quantizers: find and initialize
w_quants = ff.find_quantizers(model, "**/self_attn/**/[quantizer:parameter/weight]")
w_quants |= ff.find_quantizers(model, "**/mlp/**/[quantizer:parameter/weight]")
w_quants.initialize(LinearQuantizer, num_bits=4, granularity=per_block_32)
print(f"MLP and self-attention wegiht quantizers: {len(w_quants)}")

lmhead_w_quants = ff.find_quantizers(model, "**/lm_head/[quantizer:parameter/weight]")
lmhead_w_quants.initialize(LinearQuantizer, num_bits=4, granularity=per_block_32)
print(f"LM-Head wegiht quantizers: {len(lmhead_w_quants)}")

embed_w_quants = ff.find_quantizers(model, "**/embed_tokens/[quantizer:parameter/weight]")
embed_w_quants.initialize(LinearQuantizer, num_bits=4, granularity=per_block_32)
print(f"Embedding wegiht quantizers: {len(embed_w_quants)}")


# %% [markdown]
# ---
# # Calibrate the Quantizers
#
# Static quantizers like `LinearQuantizer` need to observe real activations to choose appropriate quantization ranges.
# We run a few batches from the training split through the model inside the `ff.estimate_ranges` context manager,
# which collects running min/max statistics and uses them to set each quantizer's range.
#
# `strict_quantization(False)` context manager lets the model run even when some operations are not yet fully quantized.

# %%

model.to(device)

with ff.estimate_ranges(model, ff.range_setting.running_minmax), ff.strict_quantization(False):
    for batch in sliced_tqdm(valid_loader, limit=4):
        batch = prepare_batch(batch, device)
        outputs = model(**batch)

# %% [markdown]
# ---
# # Initialize Dynamic Quantizers
# Dynamic quantizers do not need calibration, so we initialize them after we already
# calibrated all the static quantizers.

# %%

# Granularity
per_token = ff.granularity.PerChannel(channel_dim=1)

# Model activation quantizers: find and initialize
a_quants = ff.find_quantizers(model, "**/mlp/**/[quantizer:activation/output]")
a_quants |= ff.find_quantizers(model, "**/self_attn/**/[quantizer:activation/output]")
a_quants |= ff.find_quantizers(model, "**/input_layernorm/[quantizer:activation/output]")
a_quants |= ff.find_quantizers(model, "**/post_attention_layernorm/[quantizer:activation/output]")
a_quants |= ff.find_quantizers(model, "**/norm/[quantizer:activation/output]")
a_quants |= ff.find_quantizers(model, "**/attn_res_act_quantizer")
a_quants |= ff.find_quantizers(model, "**/mlp_res_act_quantizer")
a_quants |= ff.find_quantizers(model, "**/lm_head/[quantizer:activation/output]")
a_quants.initialize(DynamicLinearQuantizer, num_bits=8, symmetric=False, granularity=per_token)
print(f"Activation quantizers: {len(a_quants)}")


# %% [markdown]
# ---
# # Evaluate the Quantized Model
#
# Finally, we recompute perplexity on the validation split and compare against the FP baseline.
# The gap reflects the cost of the quantization configuration we picked above;
# tightening or loosening bit-widths and granularities is how we trade off accuracy against efficiency.

# %%

model.to(device)
with ff.strict_quantization(False):
    q_task_results = evaluate_model(model, valid_loader, limit=8, device=device)

print("Quantized LLaMA performance:")
print(f"---> FP perplexity:    {fp_task_results}")
print(f"---> Quant perplexity: {q_task_results}")


# %% [markdown]
# ---
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
