# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # LLaMA Export Notebook
#
# > ⚠️ **WARNING**: Export is an experimental feature and is currently under active development. Please expect API changes. We encourage you to file bug reports if you run into any problems.
#
# `fastforward` provides an export feature for deploying models on devices. This feature is designed to take a quantized model and generate the necessary artifacts for executing model inference on a device. Currently, export targets QNN (the Qualcomm AI Engine Direct SDK), Qualcomm's primary runtime for on-device AI inference, so the artifacts it produces are formatted for QNN.
#
# In this tutorial, we will demonstrate how to take an already quantized LLaMA module and produce the relevant artifacts. Additionally, we will explore some extra functionalities that export offers to help troubleshoot issues with model conversion or performance.
#
# This tutorial assumes that readers are familiar with fastforward's method for [quantizing networks](quantizing_networks.nb.py). If not, we recommend reading the quantizing_networks tutorial and the quick start guide for LLMs.
#
# Under the hood, `export` is a thin wrapper around a pipeline of stages (capture → cleanup → ONNX conversion → encodings extraction). For most users the convenience entry point used in this tutorial is enough; if you need to inspect, mutate, or replace the pipeline, see the [Building and customizing export pipelines](export_pipeline.nb.py) tutorial.
#

# ## Setup
#
# The tested Python, PyTorch, and CUDA versions are defined in
# [`scripts/versions.py`](https://github.com/Qualcomm-AI-research/fastforward/blob/main/scripts/versions.py).

# +
import logging
import pickle
import warnings

from pathlib import Path
from tempfile import TemporaryDirectory

import fastforward as ff
import torch

from fastforward.export import export
from fastforward.export.module_export import export_modules
from transformers import LlamaConfig, LlamaForCausalLM

from doc_helpers import quantized_llama as quantized_llama  # noqa: F401

warnings.filterwarnings("ignore")
logging.getLogger("torch.onnx._internal._registration").setLevel(logging.ERROR)
logging.getLogger("torch.onnx._internal.exporter").setLevel(logging.ERROR)

torch.set_grad_enabled(False);  # fmt: skip  # noqa: E703
# -

# ## Model Definition and Quantization
#
# We build a small LLaMA model from a `LlamaConfig` and quantize it using FastForward. For convenience, we use a smaller model to avoid downloading large model weights.

# +
SEQUENCE_LENGTH = 128

config = LlamaConfig(
    vocab_size=320,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    max_position_embeddings=SEQUENCE_LENGTH,
    attn_implementation="eager",
    use_cache=False,
)
model = LlamaForCausalLM(config)
ff.quantize_model(model)

# Initialize weight quantizers for self_attn and MLP layers
w_quantizers = ff.find_quantizers(model, "**/layers/*/self_attn/*/[quantizer:parameter/weight]")
w_quantizers |= ff.find_quantizers(model, "**/layers/*/mlp/*/[quantizer:parameter/weight]")
w_quantizers.initialize(ff.nn.LinearQuantizer, num_bits=8, granularity=ff.PerChannel())

# Calibrate quantizers with a single forward pass
ff.set_strict_quantization(False)
with ff.estimate_ranges(model, ff.range_setting.running_minmax):
    model(input_ids=torch.randint(0, config.vocab_size, (1, SEQUENCE_LENGTH)))
# -

# ## Quantized Model
#
# For completeness we print out the quantized LLaMA model, with all its applied quantized modules.

print("Quantized model:")
model

# ## Export
#
# Now we can use the `export` function from FastForward. For a more detailed overview of the `export` function we invite users to read through the docstring of `export`.
#
# > 💡 **Note**: Quantization-spec propagation is now performed automatically by the export pipeline (the `PropagateFFQuantSpecs` pass runs as part of `stage_convert_captured_impl_ff`). The old `enable_encodings_propagation` keyword has been removed; you no longer need to opt in.
#
# We do not want the model to construct its causal attention mask inside the forward pass
# when running on device. Instead, we precompute the mask ahead of time and pass it as an
# explicit input, so it appears in the exported graph as data rather than as control flow
# the QNN converter would have to synthesize.


# +
def causal_attention_mask(sequence_length: int, lowest_value: float = -100.0) -> torch.Tensor:
    """Build an additive causal mask of shape (1, 1, seq, seq).

    Positions a token may attend to are 0; future positions are masked with `lowest_value`.
    """
    mask = torch.full((sequence_length, sequence_length), lowest_value)
    mask = torch.triu(mask, diagonal=1)
    return mask.reshape(1, 1, sequence_length, sequence_length)


work_dir = TemporaryDirectory()
work_root = Path(work_dir.name)

model_name = "llama_test"
output_directory = work_root / model_name

# Prepare sample inputs for export: the token ids and the precomputed attention mask.
sample_input_ids = torch.randint(0, config.vocab_size, (1, SEQUENCE_LENGTH))
attention_mask = causal_attention_mask(SEQUENCE_LENGTH)

model_kwargs = {
    "input_ids": sample_input_ids,
    "attention_mask": attention_mask,
}

export_result = export(
    model=model,
    data=(),
    output_directory=output_directory,
    model_name=model_name,
    model_kwargs=model_kwargs,
    verbose=False,
)
# -

# As a result our chosen output directory is now populated with all the relevant files. Because we choose to split the parameters from the ONNX graph, we will get separate files for each model parameter. However, running the command below, we can see on the top the ONNX and encodings files which are the most relevant to us.
# Note that we limit the directory print-out to the first 15 items.

sorted(output_directory.iterdir())[:15]

# It is advised to store both the input batch and the model output in order to later check with the same data the performance of the model on device. Here we store these as a dictionary, keeping the `input` field as None since the input to the model is a dictionary batch, so it contains only `kwargs`.

# +
model_output = model(**model_kwargs)

data_location = output_directory / "input_output.pickle"

input_output_registry = {
    "input": None,
    "output": model_output,
    "kwargs": model_kwargs,
}

with open(data_location, "wb") as fp:
    pickle.dump(input_output_registry, fp)
# -

# ## Module Level Export
#
# To further demonstrate the capabilities of FastForward's export feature, we illustrate how the `export_modules` function can be utilized to prepare a collection of modules for deployment on a device. In this example, we use [MPath](mpath.nb.py) to select various modules, which are then passed to the `export_modules` function.
#
# After execution, all the individual modules are stored along with their respective inputs/outputs tensors (both for quantized and non-quantized versions of the model to aid comparison). This functionality aims to aid in cases where the full model might be failing during either QNN conversion or deployment steps, as well as providing insight to problems arising in performance (either regarding execution speed or unexpected quantization errors)

# +
modules_output_path = work_root / "modules"
modules_folder_name = "llama_modules"

module_collection = ff.mpath.search("**/layers/0/mlp", model)
module_collection |= ff.mpath.search("**/layers/0/self_attn/[cls:torch.nn.Linear]", model)
module_collection |= ff.mpath.search("**/layers/0/input_layernorm", model)
module_collection |= ff.mpath.search("**/layers/0/post_attention_layernorm", model)
module_collection |= ff.mpath.search("**/layers/0/self_attn", model)
module_collection |= ff.mpath.search("**/layers/0/mlp/[cls:torch.nn.Linear]", model)
module_collection |= ff.mpath.search("**/layers/0", model)
module_collection |= ff.mpath.search("**/lm_head", model)
module_collection |= ff.mpath.search("**/norm", model)
module_collection |= ff.mpath.search("**/embed_tokens", model)

paths = export_modules(
    model,
    None,
    module_collection,
    model_name=modules_folder_name,
    kwargs=model_kwargs,
    output_path=modules_output_path,
    verbose=False,
)
# -

# In the dictionary below we can see that each exported module is mapped to a directory, where all its relevant artifacts are stored.

paths

# For example for the `lm_head` module we get the following directory structure

sorted(paths["lm_head"].iterdir())

# ## Cleanup

work_dir.cleanup()

# ## Conclusion
#
# In this tutorial we have presented the functionality that FastForward provides for exporting a LLaMA-type module. The resulting artifacts can then be used for
# deployment through QNN. Instructions on how to proceed with on-device deployment can be found on the [QNN documentation](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/general_introduction.html).
#
# If you want to customize what `export` does — adding logging stages, replacing how artifacts are saved, or registering an entirely new pipeline — see the [Building and customizing export pipelines](export_pipeline.nb.py) tutorial, which walks through the same machinery on a small ConvNet.

# + [markdown] magic_args="[markdown]"
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
