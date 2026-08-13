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

# # Saving the Quantization State
#
# In this notebook we will explain how to save and load the quantization state of
# a model quantized with fastforward.
#
# First of all, we need a model to play with: let's build a small multi-layer
# perceptron from standard `torch.nn` modules.

# +
import torch

NUM_LAYERS = 3
HIDDEN_SIZE = 32


def make_model() -> torch.nn.Sequential:
    """Build a small MLP whose blocks live under a `layers` attribute."""
    layers = [
        torch.nn.Sequential(
            torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        )
        for _ in range(NUM_LAYERS)
    ]
    model = torch.nn.Sequential()
    model.add_module("layers", torch.nn.Sequential(*layers))
    return model


model_name = "tiny-mlp-demo"
model = make_model()

# Now we can quantize the model and initialize a few quantizers.
# We select the weight quantizers of the first linear layer in each block, and
# instantiate them to 4 bit per-channel linear quantizers.

# +
import fastforward as ff

ff.quantize_model(model)

ff.find_quantizers(model, "**/layers/*/0/[quantizer:parameter/weight]").initialize(
    ff.nn.LinearQuantizer, num_bits=4, granularity=ff.PerChannel(), quantized_dtype=torch.float32
)

# + [md]
# The quantization state is composed by:
#  - The quantizers instantiation information (e.g.: linear, 4 bits, per channel),
#  - The quantization parameters (e.g.: scale and offset of each quantizer).
#
# We already have the instantiation information, but we still have to compute the quantization
# parameters.
# Let's do that with a simple round-to-nearest min-max range estimator, using some random data:
# +
with (
    torch.no_grad(),
    ff.estimate_ranges(model, ff.range_setting.running_minmax),
    ff.set_strict_quantization(False),
):
    model(torch.randn(1, HIDDEN_SIZE))

print("Scale of the first quantizer: ", model.layers[0][0].weight_quantizer.scale)

# + [md]
# 💾 Now we can save the quantization state.
# We pass a `name_or_path` to identify the saved state (see the note on the `name_or_path` argument below).
# +
from pathlib import Path
from tempfile import TemporaryDirectory

tmpdir = Path(TemporaryDirectory().name)
model.save_quantization_state(cache_dir=tmpdir, name_or_path=model_name)
for p in tmpdir.glob("**/*"):
    print(str(p))

# + [md]
# Both `save_quantization_state()` and `load_quantization_state()` methods accept several
# arguments to control where and how the quantization state is stored.
#
# - `cache_dir`: specifies the base directory where quantization states are stored.
#
# - `tag`: allows you to create multiple versions or variants of quantization
#   states for the same model. This is useful when you want to save different quantization
#   configurations (e.g., different bit widths, granularities) for the same base model.
#
# - `name_or_path`: specifies the model identifier used to organize quantization
#   states. By default, it uses the value of the model's property `config.name_or_path`.
#
# You can lookup the API reference to know more: [save_quantization_state](/reference/fastforward/nn/quantized_module/#fastforward.nn.quantized_module.QuantizedModule.save_quantization_state).

# + [md]
# ## Quantization State Files
# The quantization state consists of two files:
# - `config.yaml`: a text file where other quantizer attributes are stored.
# - `model.safetensors`: a binary file where state_dict (parameters and buffers) of all quantizers is saved.
# +
import pygments

from IPython.display import HTML, display

config = pygments.highlight(
    next(tmpdir.glob("**/config.yaml")).read_text(encoding="utf8"),
    pygments.lexers.YamlLexer(),
    pygments.formatters.HtmlFormatter(),
)
display(HTML(f"<details><summary>config.yaml</summary>{config}</details>"))


# + [md]
# A `config.yaml` file might look scary due to its advanced, but valid, `yaml` syntax.
# Thus, it might be good to refresh your knowledge about yaml features:
# * sequence vs mapping
# * tags (`!` and `!!`)
# * anchors (`&` and `*`)
# * complex key definition (`?`)
#
# This [learnxinyminutes explanation](https://learnxinyminutes.com/yaml/) is a good reference guide
# for yaml ant its syntax.

# + [md]
# ## Loading the Quantization State
# To load the quantization state you should use the
# [`load_quantization_state`](../../reference/fastforward/nn/quantized_module/#fastforward.nn.quantized_module.QuantizedModule.load_quantization_state)
# function.
#
# We start from a fresh quantized model and load the saved state into it:

# +
new_model = make_model()
ff.quantize_model(new_model)
new_model_str = str(new_model)
new_model.load_quantization_state(cache_dir=tmpdir, name_or_path=model_name)

# + [md]
# 🔍 To see what loading changed, we diff the model's structure before and after the load. The
# green lines show where the `QuantizerStub` placeholders were replaced by the real quantizers:

# +
import difflib

diff = pygments.highlight(
    "\n".join(difflib.unified_diff(new_model_str.splitlines(), str(new_model).splitlines())),
    pygments.lexers.DiffLexer(),
    pygments.formatters.HtmlFormatter(),
)
display(HTML(diff))

# + [md]
# ✅ The stubs were replaced by the real quantizers.
# The quantizer parameters match the original model too - here we check that the loaded weight quantizer has the same scale:

# +
torch.testing.assert_close(
    model.layers[0][0].weight_quantizer.scale,
    new_model.layers[0][0].weight_quantizer.scale,
)

# + [md]
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
