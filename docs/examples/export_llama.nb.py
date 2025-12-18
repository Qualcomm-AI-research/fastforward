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
# `fastforward` provides an export feature for deploying models on devices. This feature is designed to take a quantized model and generate the necessary artifacts for executing model inference on a device. Currently, export supports only QNN as the deployment framework, so the artifacts it produces are formatted for QNN.
#
# In this tutorial, we will demonstrate how to take an already quantized LLaMA module (using the tinyLLaMA weights) and produce the relevant artifacts. Additionally, we will explore some extra functionalities that export offers to help troubleshoot issues with model conversion or performance.
#
# This tutorial assumes that readers are familiar with fastforward's method for [quantizing networks](quantizing_networks.nb.py). If not, we recommend reading the quantizing_networks tutorial and the quick start guide for LLMs.
#

# ## Python requirements
#
# Please ensure that the below pip requirements are used.
#
# ```
# datasets==3.3.2
# transformers==4.46.3
# torch==2.5
# tqdm
# ```

# +
import os

os.environ["HF_DATASETS_CACHE"] = "/prj/corp/llm/lasvegas/llm-systems-scratch/cache/datasets/"
os.environ["HF_HUB_CACHE"] = "/prj/corp/llm/lasvegas/llm-systems-scratch/cache/models"
os.environ["HF_HUB_OFFLINE"] = "1"

import logging
import pickle
import warnings

import datasets
import fastforward as ff
import torch

from fastforward.export.export import export
from fastforward.export.module_export import export_modules

# Helper function for generating the attention mask
from doc_helpers.export.benchmark.util import generate_attention_mask

# Imports to get the prepared LLaMA model and quantize it appropriately
from doc_helpers.export.prepare_quantized_llama import get_llama_and_dataloaders
from doc_helpers.utils import create_output_directory

warnings.filterwarnings("ignore")
logging.getLogger("torch.onnx._internal._registration").setLevel(logging.ERROR)
logging.getLogger("torch.onnx._internal.exporter").setLevel(logging.ERROR)
datasets.utils.logging.get_logger("datasets").setLevel("ERROR")
# -

# ## Mode setup
#
# Since we are not training the model, and to avoid memory issues, we switch off gradient calculation.

torch.set_grad_enabled(False);  # fmt: skip  # noqa: E703

# ## Model Definition and Quantization

# The below code brings in the quantized LLaMA model and its respective dataloaders.
#
# Please note that the LLaMA model here is slightly altered in order to accomodate deployment to QNN. For this reason we have simplified the LLaMA code located in the `export_helpers/llama` directory.
#
# > ⚠️ **WARNING**: The LLaMA code and evaluation pipeline in this notebook (and its relevant files) are modified specifically for deployment through QNN. Be very careful when performing changes to the code as even simple alterations can cause the deployment to fail.
#
# More specifically on the quantization parameters:
#
# - We use W8A16 quantization, where the weights are using per channel quantization.
# - By default we perform PTQ for 16 batches, and the loss/PPL are also calculated on 16 batches to speed up execution.
#
# > ⚠️ **WARNING**: Our implementation is currently limited in working with different configurations. For example, W4A16 per channel will result in an OOM error when executing on QNN.
#
# For the specific quantization parametrization used for exporting this LLaMA model you can look into the `docs/examples/doc_helpers/export/prepare_quantized_llama.py` file.

# +
SEQUENCE_LENGTH = 128

model, ptq_dataloader, test_dataloader = get_llama_and_dataloaders(
    model_tag="TinyLlama/TinyLlama_v1.1",
    device="cuda",
    sequence_length=SEQUENCE_LENGTH,
)
# -

# ## Quantized Model
#
# For completeness we print out the quantized LLaMA model, with all its applied quantized modules.

print("Quantized model:")
model

# ## Export Preparation
#
# Here we perform some setup and housekeeping to export the model, mainly through moving the model/data to the correct device, preparing the batch format and storing data for later usage.

# +
model_name = "llama_test"
output_directory = create_output_directory() / model_name

# There are certain issues with using the GPU in dynamo (
# for using the `.to` [method](https://github.com/pytorch/pytorch/issues/119665)
# and for the `layernorm` [operation](https://github.com/pytorch/pytorch/issues/133388))
# so in order to export we move both the model and data to the CPU.
model = model.cpu()

# We use a modified batch, removing arguments that are not
# relevant to export
batch = next(iter(test_dataloader))
batch["use_cache"] = False
batch.pop("labels")
# We do not want to generate the attention mask in the model's
# forward pass when running the model on device, so we instead
# prepare it ahead of time and pass it as an input.
attention_mask = generate_attention_mask(SEQUENCE_LENGTH)
batch["attention_mask"] = attention_mask.cpu()
# -

# Now we can use the `export` function from FastForward. For a more detailed overview of the `export` function we invite users to read through the docstring of `export`. In addition, we also set the `enable_encodings_propagation` to `True`, which can provide a more complete mapping of quantization settings to operations (which is relevant for QNN). We again invite users to read through the docstring for the `propagate_encodings` function.

export(
    model=model,
    data=(),
    output_directory=output_directory,
    model_name=model_name,
    model_kwargs=batch,
    enable_encodings_propagation=True,
    verbose=False,
)

# As a result our chosen output directory is now populated with all the relevant files. Because we choose to split the parameters from the ONNX graph, we will get separate files for each model parameter. However, running the command below, we can see on the top the ONNX and encondings files which are the most relevant to us.
# Note that we limit the directory print-out to the first 15 items.

sorted(output_directory.iterdir())[:15]

# It is advised to store both the input batch and the model output in order to later check with the same data the performance of the model on device. Here we store these as a dictionary, keeping the `input` field as None since the input to the model is a dictionary batch, so it contains only `kwargs`.

# +
model_output = model(**batch)

data_location = output_directory / "input_output.pickle"

input_output_registry = {
    "input": None,
    "output": model_output,
    "kwargs": batch,
}

with open(data_location, "wb") as fp:
    pickle.dump(input_output_registry, fp)
# -

# ## Module Level Export
#
# To further demonstrate the capabilities of FastForward's export feature, we illustrate how the export_modules function can be utilized to prepare a collection of modules for deployment on a device. In this example, we use [MPath](mpath.nb.py) to select various modules, which are then passed to the export_modules function.
#
# After execution, all the individual modules are stored along with their respective inputs/outputs tensors (both for quantized and non-quantized versions of the model to aid comparison). This functionality aims to aid in cases where the full model might be failing during either QNN conversion or deployment steps, as well as providing insight to problems arising in performance (either regarding execution speed or unexpected quantization errors)

# +
modules_output_path = create_output_directory()
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
    enable_encodings_propagation=True,
    kwargs=batch,
    output_path=modules_output_path,
    verbose=False,
)
# -

# In the dictionary below we can see that each exported module is mapped to a directory, where all its relevant artifacts are stored.

paths

# For example for the `lm_head` module we get the following directory structure

sorted(paths["lm_head"].iterdir())

# We can also pass the full LLaMA model through the `export_modules` function. In that case we do not need to manually save the input/output tensors of the model as we have done when using the standard `export` function.

# +
full_llama_output_path = create_output_directory()
full_llama_name = "full_llama"

full_llama = export_modules(
    model,
    None,
    model,
    model_name=full_llama_name,
    enable_encodings_propagation=True,
    kwargs=batch,
    output_path=full_llama_output_path,
    verbose=False,
)
# -

# ## Conclusion
#
# In this tutorial we have presented the functionality that FastForward provides for exporting a LLaMA-type module. The resulting artifacts can then be used for
# deployment through QNN. Instructions on how to proceed with on-device deployment can be found on the [QNN documentation](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/general_introduction.html).

# + [markdown] magic_args="[markdown]"
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
