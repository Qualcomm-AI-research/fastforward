# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,.nb.py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # FastForward: Getting Started
#
# This notebook introduces the core building blocks of `fastforward` and shows
# how to quantize simple modules such as a multi-layer perceptron. It is a good
# starting point if you want to familiarize yourself with the library.
#
# The tutorial is organized in four sections:
#
# 1. **Quantized Tensors** — `QuantizedTensor`, a subclass of `torch.Tensor`
#    that is the fundamental datatype in FastForward.
# 2. **Quantizers** — `torch.nn.Module`s that turn floating point tensors into
#    `QuantizedTensor`s and can learn their parameters from data.
# 3. **Quantized Modules** — how to turn an unquantized module into a quantized
#    one by changing the module class, inserting quantizers, and estimating
#    the range for each quantizer.
# 4. **Quantized Models** — how to automate the steps above with
#    `quantize_model` and `QuantizationConfig`, and how to handle custom or
#    third-party modules.


# %% [markdown]
# ## 1. Quantized Tensors
#
# The `ff.quantized_tensor.QuantizedTensor` is a subclass of `torch.Tensor`
# designed to hold quantized data. It supports any quantization scheme
# (uniform, dynamic, vector, and so on), but in this tutorial we focus on
# integer quantization on a fixed per-tensor or per-channel grid.
#
# You do not need to know the details of this scheme to follow along, but if
# you are curious please refer to
# [A White Paper on Neural Network Quantization (Nagel et al., 2021)](https://arxiv.org/abs/2106.08295).
#
# Let's start by creating some floating point data.

# %%
import torch

in_features = 4
data = torch.rand(1, in_features) - 0.5
data


# %% [markdown]
# Now we quantize the data using 8-bit per-tensor quantization.

# %%
import fastforward as ff

scale = torch.tensor([0.1])
num_bits = 8
quantized_data = ff.quantization.affine.quantize_per_tensor(data, num_bits=num_bits, scale=scale)

quantized_data


# %% [markdown]
# The result is a `QuantizedTensor`, which makes it easy to check whether a
# tensor is quantized. It carries both the actual data (with the same shape
# as the input) and the quantization parameters — in this case only the
# scale, which we set manually.
#
# Because the values live in a different coordinate system, they are not
# directly comparable to the original floating point values. To recover the
# floating point representation we dequantize the tensor.

# %%
quantized_data.dequantize()


# %% [markdown]
# ## 2. Quantizers
#
# In the previous section we quantized a floating point tensor by providing
# the quantization parameters ourselves. In practice these parameters are rarely
# known in advance.
#
# A `ff.nn.Quantizer`s is a `nn.Module` that quantize data in it's forward
# pass and can also estimate or learn the quantization parameters from
# data. Let's create a `LinearQuantizer`:

# %%
quantizer = ff.nn.linear_quantizer.LinearQuantizer(num_bits=2)
quantizer


# %% [markdown]
# If we try to quantize our data now, the call fails because the
# quantization range has not been set yet, and consequently the
# quantizer parameters are not initialized.

# %%
try:
    quantizer(data)
except ValueError as e:
    print("[ERROR]", e, "\n")

print(f"{quantizer.has_uninitialized_params=}")
print(f"{quantizer.quantization_range=}")  # min, max values the quantizer can represent


# %% [markdown]
# We could set `quantizer.quantization_range` directly, but this requires us
# to know the desired `(min, max)` up front. A more common approach is to use
# *range estimation* to derive it from data.

# %%
with ff.range_setting.estimate_ranges(quantizer, ff.range_setting.smoothed_minmax):
    quantizer(data)

print(f"{quantizer.has_uninitialized_params=}")
print(f"{quantizer.quantization_range=}")
print(f"{data.min()=} {data.max()=}")


# %% [markdown]
# The quantizer parameters are now initialized, and its range matches the
# range of the data batch. We can use it to quantize the data.

# %%
quantized_data = quantizer(data)  # type: ignore[assignment]
quantized_data


# %% [markdown]
# ## 3. Quantized Modules
#
# Quantizers are rarely used in isolation — most of the time we want to
# quantize a full model. This section walks through turning a single module
# into a quantized module and explains what happens under the hood. The next
# section introduces convenience methods for larger models.
#
# We start with a simple unquantized linear layer.

# %%
out_features = 8

unquantized_linear = torch.nn.Linear(in_features, out_features)
print(unquantized_linear)


# %% [markdown]
# FastForward provides `ff.nn.QuantizedModule` classes as drop-in
# replacements for `torch.nn.Module`. Most modules in `torch.nn` have a
# quantized counterpart in `ff.nn`. These modules:
#
# - behave exactly like their floating point counterparts, exposing the
#   same methods with the same signatures;
# - add quantizer children and override the forward pass so that operations
#   run in quantized form.
#
# If a module you need is missing from `ff.nn`, you can either open an issue
# or implement the quantized version yourself.
#
# Let's take a closer look at `ff.nn.QuantizedLinear`. For clarity we copy
# the weights from the unquantized layer manually so that the two layers
# start out identical.

# %%
quantized_linear = ff.nn.QuantizedLinear(in_features, out_features)
quantized_linear.weight.data = unquantized_linear.weight.data.clone()
quantized_linear.bias.data = unquantized_linear.bias.data.clone()

print(quantized_linear)


# %% [markdown]
# The `QuantizedLinear` has the same structure as the `Linear`, plus four
# quantizer children. All of them are initialized to `QuantizerStub`, a
# no-op placeholder that can be replaced with a real quantizer when needed.
#
# Let's try to push data through the layer.

# %%
try:
    quantized_output = quantized_linear(data)
except ff.exceptions.QuantizationError as e:
    print("[ERROR]", e, "\n")


# %% [markdown]
# The call fails because `strict_quantization=True` by default. This flag
# guards against a common pitfall in simulated quantization: forgetting to
# assign quantizers and unintentionally running the layer in floating point.
# Since we have not assigned any quantizers yet, the layer would behave as
# a floating point layer, which strict mode does not allow.
#
# Let's disable strict quantization temporarily and confirm the behavior.

# %%
with ff.strict_quantization(False):
    quantized_output = quantized_linear(data)

unquantized_output = unquantized_linear(data)

print(f"{unquantized_output=}")
print(f"{quantized_output=}")


# %% [markdown]
# As expected, `quantized_linear` behaves identically to `unquantized_linear`
# because no quantizers are active. Let's now assign quantizers to each
# quantizer field.

# %%
quantized_linear.input_quantizer = ff.nn.linear_quantizer.LinearQuantizer(num_bits=2)
quantized_linear.weight_quantizer = ff.nn.linear_quantizer.LinearQuantizer(num_bits=2)
quantized_linear.output_quantizer = ff.nn.linear_quantizer.LinearQuantizer(num_bits=2)
print(quantized_linear)


# %% [markdown]
# Just as before, we need to initialize the quantizers by passing data
# through them.

# %%
print("Before range estimation")
print(f"{quantized_linear.input_quantizer.quantization_range=}")
print(f"{quantized_linear.weight_quantizer.quantization_range=}")
print(f"{quantized_linear.output_quantizer.quantization_range=}")
print()

with ff.range_setting.estimate_ranges(quantized_linear, ff.range_setting.smoothed_minmax):
    quantized_linear(data)

print("After range estimation")
print(f"{quantized_linear.input_quantizer.quantization_range=}")
print(f"{quantized_linear.weight_quantizer.quantization_range=}")
print(f"{quantized_linear.output_quantizer.quantization_range=}")


# %% [markdown]
# All quantizers are now initialized and we can call the layer.

# %%
unquantized_output = unquantized_linear(data)
quantized_output = quantized_linear(data)

print(f"{unquantized_output=}")
print()
print(f"{quantized_output=}")
print()
print(f"{quantized_output.dequantize()=}")


# %% [markdown]
# `quantized_linear` now behaves as expected:
#
# - the output is a `QuantizedTensor`;
# - the dequantized output is close to the floating point output, but
#   differs slightly due to quantization error.


# %% [markdown]
# ## 4. Quantized Models
#
# The previous section showed the three steps needed to quantize a module:
#
# 1. Turn the unquantized module into a quantized module.
# 2. Replace the desired `QuantizerStub`s with real `Quantizer`s.
# 3. Estimate the quantizer ranges by passing data through the model.
#
# Doing this by hand is tedious for anything larger than a single layer.
# FastForward provides helpers to automate steps 1 and 2, which we cover
# next.
#
# Let's start with a small MLP.

# %%
hidden_features = 3

unquantized_model = torch.nn.Sequential(
    torch.nn.Linear(in_features, hidden_features),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_features, hidden_features),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_features, out_features),
    torch.nn.ReLU(),
)

unquantized_model


# %% [markdown]
# ### Replacing modules with their quantized counterparts
#
# `ff.quantize_model` walks a model in place and replaces every module with
# its `QuantizedModule` counterpart. Internally it uses a dictionary that
# maps `torch.nn.Module` subclasses to their `ff.nn.QuantizedModule`
# equivalents. Let's inspect it.

# %%
ff.quantized_module_map()


# %% [markdown]
# Because `ff.quantize_model` mutates the model in place, we deepcopy the
# floating point model first so we can still compare against it later.

# %%
import copy

quantized_model = copy.deepcopy(unquantized_model)
ff.quantize_model(quantized_model)
quantized_model


# %% [markdown]
# All modules have been replaced with their quantized counterparts. Since
# no quantizers are inserted yet, the quantized model should still behave
# like the unquantized one.

# %%
with ff.strict_quantization(False):
    quantized_output = quantized_model(data)

unquantized_output = unquantized_model(data)

print(f"{unquantized_output=}")
print(f"{quantized_output=}")


# %% [markdown]
# ### Inserting quantizers with `QuantizationConfig`
#
# `ff.QuantizationConfig` automates the replacement of `QuantizerStub`s with
# real `Quantizer`s through a set of rules. Each rule has two parts:
#
# 1. A **query** that selects the layers the rule applies to. Filtering
#    uses the `ff.mpath` library — see the [MPath tutorial](/examples/mpath.nb) for details.
# 2. A **quantizer class or factory**. When given a class, a new quantizer
#    of that class is created for each match using the provided keyword
#    arguments. When given a factory function, the function receives the
#    full name of the quantizer and the current quantizer at that location,
#    and is expected to return an initialized quantizer.
#
# When multiple rules match the same quantizer, the rule added last wins.
#
# Let's build a configuration.

# %%
config = ff.QuantizationConfig()

# Quantize all weights in the model.
config.add_rule(
    "**/[quantizer:parameter/weight]",
    ff.nn.LinearQuantizer,
    num_bits=8,
    symmetric=True,
    granularity=ff.PerChannel(),
)

# Quantize all outputs in the model.
config.add_rule(
    "**/[quantizer:activation/output]",
    ff.nn.LinearQuantizer,
    num_bits=8,
    symmetric=False,
    granularity=ff.PerTensor(),
)


# Enable the input quantizer only on the first layer, so that a floating
# point input can be turned into a quantized input. Subsequent layers
# already receive quantized inputs from the previous output quantizer.
def input_factory(name: str, current_quantizer: ff.nn.Quantizer) -> ff.nn.Quantizer:  # noqa: ARG001
    return ff.nn.LinearQuantizer(num_bits=8, symmetric=False, granularity=ff.PerTensor())


config.add_rule("0/[quantizer:activation/input]", input_factory)

config


# %% [markdown]
# For the input quantizer rule we could have passed the quantizer class
# directly. We used a factory function instead to illustrate how it works:
# the function receives both the name and the current quantizer at that
# location (either an initialized quantizer or a `QuantizerStub`).
#
# Applying the configuration to the model is a single call.

# %%
config.initialize(quantized_model)
quantized_model


# %% [markdown]
# The quantizers are wired up as expected. All that is left is estimating
# their ranges, and the model is ready to use.

# %%
with ff.range_setting.estimate_ranges(quantized_model, ff.range_setting.smoothed_minmax):
    quantized_model(data)

quantized_model(data)


# %% [markdown]
# ### Quantizing custom modules
#
# Real-world models often contain third-party or custom modules on top of
# the standard `torch.nn` ones. `quantize_model` only knows how to convert
# modules registered in its module map, so it cannot handle a custom layer
# out of the box.
#
# > **Tip:** [`ff.autoquantize`](autoquant.md) (experimental) can also handle
# > this step for you.
#
# Let's define a custom self-attention layer.

# %%
from typing_extensions import override


class MySelfAttentionLayer(torch.nn.Module):
    def __init__(self, feature_size) -> None:
        print("Calling MySelfAttentionLayer.__init__")
        super().__init__()
        self.feature_size = feature_size

        # Linear transformations for Q, K, V from the same source
        self.key = torch.nn.Linear(feature_size, feature_size)
        self.query = torch.nn.Linear(feature_size, feature_size)
        self.value = torch.nn.Linear(feature_size, feature_size)

    @override
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        print("Calling MySelfAttentionLayer.forward")
        # Apply linear transformations
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        # Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1))
        scores = scores / torch.sqrt(torch.tensor(self.feature_size, dtype=torch.float32))

        # Apply softmax
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)

        # Multiply weights with values
        output = torch.matmul(attention_weights, values)

        return output, attention_weights


# %%
num_features = 8
my_unquantized_layer = MySelfAttentionLayer(num_features)
my_unquantized_layer


# %% [markdown]
# If we try to convert it with `ff.quantize_model` as is, the call fails
# because there is no mapping for `MySelfAttentionLayer`.

# %%
from pprint import pprint

my_quantized_layer = copy.deepcopy(my_unquantized_layer)
try:
    ff.quantize_model(my_quantized_layer)
except ff.exceptions.QuantizationError as e:
    print("[ERROR]", e, "\n")

print("ff.quantized_module_map():")
pprint(ff.quantized_module_map())


# %% [markdown]
# To make it work we manually define the quantized equivalent of
# `MySelfAttentionLayer`.


# %%
class MyQuantizedSelfAttentionLayer(MySelfAttentionLayer, ff.nn.quantized_module.QuantizedModule):
    def __init_quantization__(self) -> None:
        print("Calling MyQuantizedSelfAttentionLayer.__init_quantization__")
        super().__init_quantization__()

        self.attention_scores_output_quantizer = ff.nn.QuantizerStub(output_quantizer=True)
        self.attention_weights_output_quantizer = ff.nn.QuantizerStub(output_quantizer=True)
        self.attention_features_output_quantizer = ff.nn.QuantizerStub(output_quantizer=True)

    # This function is only wrapped for demonstration purposes
    def quantize_children(self, *args, **kwargs) -> None:
        print("Calling MyQuantizedSelfAttentionLayer.quantize_children")
        super().quantize_children(*args, **kwargs)

    def forward(self, x):
        print("Calling MyQuantizedSelfAttentionLayer.forward")
        # Apply linear transformations
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        # Scaled dot-product attention
        scores = ff.nn.functional.matmul(
            queries,
            keys.transpose(-2, -1),
            output_quantizer=self.attention_scores_output_quantizer,
        )
        scores = scores / torch.sqrt(torch.tensor(self.feature_size, dtype=torch.float32))

        # Apply softmax
        attention_weights = ff.nn.functional.softmax(
            scores, dim=-1, output_quantizer=self.attention_weights_output_quantizer
        )

        # Multiply weights with values
        output = ff.nn.functional.matmul(
            attention_weights,
            values,
            output_quantizer=self.attention_features_output_quantizer,
        )

        return output, attention_weights


# %% [markdown]
# We made two changes relative to the unquantized layer:
#
# 1. We re-implemented the forward pass, replacing every operation from
#    `torch.nn.functional` with its FastForward quantized equivalent.
#    - This means duplicating the forward-pass code, which is a real
#      downside.
#    - Watch out for functionals hidden inside helper functions called
#      from the forward pass; they need to be rewritten too.
#    - If you are adapting a third-party class, freeze the dependency so
#      that your rewritten module does not silently drift when the
#      upstream implementation changes.
#    - Using the quantized functionals requires quantizers on the module,
#      which brings us to the second change.
# 2. We added an `__init_quantization__` method that inserts the
#    `QuantizerStub`s used later for quantization.
#    - No code from the original `__init__` needs to be duplicated.
#    - As we will see below, `__init_quantization__` can be used both to
#      initialize a `QuantizedModule` from scratch and to convert an
#      existing `Module` into a `QuantizedModule`.
#
# Let's see how `MyQuantizedSelfAttentionLayer` behaves when initialized
# from scratch.

# %%
new_quantized_layer = MyQuantizedSelfAttentionLayer(num_features)
new_quantized_layer


# %% [markdown]
# Observe that:
#
# 1. `MySelfAttentionLayer.__init__` is called first, initializing the
#    layer through the unquantized base class.
# 2. `MyQuantizedSelfAttentionLayer.__init_quantization__` is then called,
#    inserting the quantizer stubs.
# 3. When initialized from scratch, child modules are not converted to
#    their quantized counterparts.
#
# In practice we rarely initialize quantized modules from scratch. It is
# more common to take a floating point model and recursively convert its
# submodules. Before doing so, let's look at the `quantized_module_map`
# again.

# %%
print("ff.quantized_module_map():")
pprint(ff.quantized_module_map()[MySelfAttentionLayer])


# %% [markdown]
# `MySelfAttentionLayer` now appears in `quantized_module_map`. All
# subclasses of `QuantizedModule` are picked up automatically, provided the
# class has been imported. If your class does not show up, import it, or
# use the `extra_conversion` argument to override entries in
# `quantized_module_map`.
#
# Now let's convert an existing instance with `quantize_model`.

# %%
my_quantized_layer = copy.deepcopy(my_unquantized_layer)
ff.quantize_model(my_quantized_layer)

my_quantized_layer


# %% [markdown]
# Observe that:
#
# 1. Because we are converting an existing layer,
#    `MySelfAttentionLayer.__init__` is not called again.
# 2. The module's class changes from `MySelfAttentionLayer` to
#    `MyQuantizedSelfAttentionLayer`.
# 3. `MyQuantizedSelfAttentionLayer.__init_quantization__` is still called,
#    which inserts the quantizer stubs into the previously unquantized
#    layer.
# 4. Child modules are also converted to their quantized counterparts via
#    `MyQuantizedSelfAttentionLayer.quantize_children`.
#
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
