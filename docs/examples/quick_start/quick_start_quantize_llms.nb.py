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

# # Quick Start: Quantization of Llama
#
# In this tutorial we will show how to quantize a large language model (Llama-v3) using **FastForward**.
#
#
# ### Step 1: Install Dependencies
# First, make sure you have all the necessary dependencies installed. You can do this by running the following command:
# ```
# pip install transformers==4.46.3 sentencepiece==0.2.0 ipywidgets==8.1.5 datasets==3.1.0
# ```
# For instructions on installing `fastforward`, please refer to the project's documentation and/or readme.
#
# ### Step 2: Load the Model, Tokenizer, and Datasets
# Next, we'll load the model, tokenizer, and datasets using the HuggingFace's `transformers` and `datasets` libraries.

# +
import os

import fastforward as ff
import torch

from datasets import load_dataset
from quick_start_utils import tokenize_dataset
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM, default_data_collator

model_dtype = torch.float16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sequence_length = 1024
batch_size = 1

valid_percent = 20
train_percent = 5

model_name_or_path = os.environ.get("FF_QUICKSTART_MODEL", "meta-llama/Llama-3.2-1B-Instruct")

# Load Model
from_tf = bool(".ckpt" in model_name_or_path)
model = LlamaForCausalLM.from_pretrained(
    model_name_or_path, from_tf=from_tf, torch_dtype=model_dtype
)
model.to(device)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, legacy=False, use_fast=True)

# Load Dataset
_valid_split = "validation" if valid_percent is None else f"validation[:{valid_percent}%]"
_train_split = "train" if train_percent is None else f"train[:{train_percent}%]"
validset = load_dataset("wikitext", "wikitext-2-raw-v1", split=_valid_split)
trainset = load_dataset("wikitext", "wikitext-2-raw-v1", split=_train_split)

# Tokenize Dataset
tokenized_validset = tokenize_dataset(validset, tokenizer, sequence_length)
tokenized_trainset = tokenize_dataset(trainset, tokenizer, sequence_length)

# Create Dataloader
valid_loader = DataLoader(
    tokenized_validset, batch_size, collate_fn=default_data_collator, shuffle=False
)
train_loader = DataLoader(
    tokenized_trainset, batch_size, collate_fn=default_data_collator, shuffle=True
)

# -

# ## Base Model Evaluation
#
# ### Step 3: Establish an Inference Loop
# First, we'll create an inference loop to assess the performance of the base model. This loop will be our foundation, allowing us to later compare the results with those from the quantized versions of the model.
#


# +
def prepare_batch(batch: dict, device: torch.device):
    return {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "labels": batch["labels"].to(torch.long).to(device),
    }


def evaluate_model(model, valid_loader, device, max_num_batches=None):
    model.eval()
    losses = []
    for batch_idx, batch in enumerate(tqdm(valid_loader)):
        if max_num_batches is not None and batch_idx >= max_num_batches:
            break

        batch = prepare_batch(batch, device)

        with torch.no_grad():
            outputs = model(**batch)

        losses.append(outputs.loss)

    eval_loss = torch.stack(losses).mean()
    perplexity = torch.exp(eval_loss)
    return float(perplexity)


orig_perplexity = evaluate_model(model, valid_loader, device=device, max_num_batches=None)
print(
    f"Perplexity over wikitext-validation using full-precision model ({model_dtype}): {orig_perplexity:.4}"
)

# -

# ## Quantized Model
# Now that we have the original full-precision model and the tokenized dataset, we can start quantizing the model using FastForward.
#
# ### Step 4: Convert to a Quantization-Ready Model
# First, we need to convert our model into a _quantization-ready_ one. This type of model, called a `QuantizedModule`, allows us to fully or partially quantize the model easily. These modules work like standard `PyTorch` modules but have extra features for seamless interaction with `FastForward` APIs.
#
# Currently, converting a model into a quantized module is semi-automatic and requires a custom implementation of all the PyTorch modules involved. If you want to create a custom QuantizedModule, check out [the tutorial on manually quantizing custom modules](https://compression.morpheus-gitlab-pages.qualcomm.com/fastforward/latest/examples/quantizing_networks.nb/#43-quantizing-custom-modules-manual-quantization). However, for this tutorial, we will use pre-provided modules to quantize the Llama model.

# +

# Import all the custom QuantizedModules required to quantize our llama model.
# We just need to import those modules so that `ff.quantize_model` will be able to find them.
from quantized_models import quantized_llama as quantized_llama  # noqa: E402

# Convert the model into a QuantizedModel  (inplace operation)
ff.quantize_model(model)

# -

# By default, FastForward operates in *"strict quantization"* mode. In this mode, many quantization errors, such as calling a quantized function without quantized tensors, are treated as errors. This is beneficial for quantizing models that require strict adherence to quantization rules. However, strict quantization is not always necessary. In this tutorial, we only partially quantize a model, so we disable strict quantization.

ff.set_strict_quantization(False)

# ## Weight-only Quantization
#
# A `QuantizedModule` contains `QuantizerStub` instances at specific locations. Each quantizer stub is a placeholder module that can be easily replaced with any FastForward quantizer.
#
# ### Step 5: Replace Stubs with Actual Quantizers
# We can start quantizing the model by replacing the stubs with actual quantizers. To do this, we use the `find_quantizers` function to select certain stubs and initialize them as `LinearQuantizer` objects.
#
# In this example, we will limit our quantization to all the weights in the self-attention and MLP modules within the Llama decoder layers.
# +

w_bits = 8

# Set Weight Quantization
w_quantizers = ff.find_quantizers(model, "**/layers/*/self_attn/*/[quantizer:parameter/weight]")
w_quantizers |= ff.find_quantizers(model, "**/layers/*/mlp/*/[quantizer:parameter/weight]")
w_quantizers.initialize(ff.nn.LinearQuantizer, num_bits=w_bits, granularity=ff.PerChannel())
print(f"Found {len(w_quantizers)} weight quantizers.")

# Move model to target device: quantizer parameters (scale/offsets) should be placed on the target device too.
model.to(device)

# -

# ### Background: `fastforward.mpath`
# The `find_quantizers` function is a tool for filtering and selecting specific quantizers within the model, using the capabilities provided by `fastforward.mpath`.
#
# By passing _queries_ to `find_quantizer`, we can navigate the model and select quantizers similarly to how we select files in a Unix-like file system. Using strings and wildcards, we can match modules and quantizers just like matching folders and files from the terminal.
#
# Additionally, **mpath** offers advanced functionalities to select modules and quantizers based on special rules. In this example, we selected only the quantizers with the tag parameter/weight because we aim to perform weights-only quantization.
#
# Each `find_quantizer` call returns a QuantizerCollection object containing the selected quantizers, which behaves similarly to a Python set.
# In this case, we merged two collections simply using the `|` operator.
#
# For more information about **mpath** and its full range of functionalities,
# we recommend reading the [**MPath tutorial**](https://compression.morpheus-gitlab-pages.qualcomm.com/fastforward/latest/examples/mpath.nb/).
#

# ## Calibrate Weight-Quantized Model
#
# Before performing inference, we need to initialize the quantizer parameters using calibration data.
#
# ### Step 6: Estimate Quantization Ranges
# FastForward provides a method for estimating quantization ranges by running the model's forward pass. This is done using the `fastforward.estimate_ranges` context manager.
#
# For weight-only quantization, passing dummy data is sufficient since no activation quantizers are involved. In this example, we use a *running min-max* estimator, which sets the quantizer ranges to the minimum and maximum values observed in the tensors during the forward pass.

# +

print("Calibrating weight-quantizers")

with torch.no_grad(), ff.estimate_ranges(model, ff.range_setting.running_minmax):
    x = next(iter(train_loader))
    x = prepare_batch(x, device)
    model(**x)

print("Calibrated!")
# -

# ### Step 7: Evaluate the weight-quantized model
# Now, we can perform inference and evaluate the model's performance using the same procedure applied to the original model.

w_quant_perplexity = evaluate_model(model, valid_loader, device=device, max_num_batches=None)
print("Perplexity over wikitext-validation:")
print(f" - Original model:       {orig_perplexity:.4f}  ({model_dtype}) ")
print(f" - W-Quantized model:    {w_quant_perplexity:.4f}  (W{w_bits})")


# ## Weight-Activation Quantization
# In addition to the weight quantizers, we will now initialize some of the input quantizers for the linear layers; enabling weight-activation quantization.
#
# Generally, quantizing all activations can significantly degrade accuracy. Therefore, in this example, we will limit our quantization to the inputs of the linear layers within the model.
#
# ### Step 8: Initialize Input Quantizers
# We are initializing input quantizers for the linear layers to enable weight-activation quantization.

# +
# We must import QuantizedLinear to enable the fragment [cls:QuantizedLinear] in find_quantizer.

a_bits = 16

# Select all linear layers output quantizers:
a_quantizers = ff.find_quantizers(
    model, "**/layers/**/[cls:torch.nn.Linear]/[quantizer:activation/input]"
)
a_quantizers.initialize(
    ff.nn.LinearQuantizer, num_bits=a_bits, symmetric=False, granularity=ff.PerTensor()
)
print(f"Found {len(a_quantizers)} activation quantizers.")

# Move model to target device: quantizer parameters (scale/offsets) should be placed on the target device too.
model.to(device)

# -

# ### Step 8: Calibrating Activation Quantizers
#
# Activation quantizers are significantly more sensitive to calibration.
# Unlike weight quantizers, the exact range of data passing through activation quantizers cannot be determined in advance. Therefore, we need to estimate the activation ranges using a calibration set.
#
# In this example, we use the Wikitext training set for calibration. Evaluation is conducted on a separate validation set.
#

# +
model.eval()

print("Calibrating quantizers over training set...")

with torch.no_grad(), ff.estimate_ranges(model, ff.range_setting.running_minmax):
    for i, x in enumerate(tqdm(train_loader)):
        x = prepare_batch(x, device)
        model(**x)

print("Calibrated!")


# -

# ### Step 9: Evaluate Weight-Activation Quantized Model
# Now, we can evaluate our weight-activation quantized model and compare its performance to the other models.

# +
wa_quant_perplexity = evaluate_model(model, valid_loader, device=device, max_num_batches=None)

print("Perplexity over wikitext-validation:")
print(f" - Original model:       {orig_perplexity:.4f}  ({model_dtype}) ")
print(f" - W-Quantized model:    {w_quant_perplexity:.4f}  (W{w_bits})")
print(f" - W+A Quantized model:  {wa_quant_perplexity:.4f}  (W{w_bits}A{a_bits})")
# -
# ## Wrap up


# In this tutorial, we demonstrated how to use FastForward to apply weight-only and weight-activation quantization to a large language model. We also evaluated the performance differences compared to the original model.
#
# FastForward, currently, provides a semi-automatic process for converting a model into a quantized one. However, if your model includes custom PyTorch modules, some manual work is still required to create a quantized version of those modules.
#
# For more information on how to quantize a model from scratch, check out the tutorial:[Getting Started: Quantizing a LLM from scratch](https://compression.morpheus-gitlab-pages.qualcomm.com/fastforward/latest/examples/quantizing_networks.nb/).
