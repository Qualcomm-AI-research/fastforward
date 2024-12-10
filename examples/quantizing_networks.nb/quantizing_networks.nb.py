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
# # Overview
#
# In this notebook, we will go over the most important objects and classes in `fastforward`. At the end of the notebook, we will have covered how to quantize simple modules like a multi-layer perceptron as well as the large language model OPT. This is a great start if you want to familiarize yourself with fastforward.
#
# The notebook consists of five sections:
# 1. **Quantized Tensors**: `QuantizedTensors`, a subclass of `torch.Tensors` which are the fundamental datatype in FastForward.
# 2. **Quantizers**: `Quantizers` are `torch.nn.Modules` that turn floating point tensors into `QuantizedTensors` and can learn from data.
# 3. **Quantized Modules**: Quantizing a module consists of three steps: 1) Changing the module to a `QuantizedModule`, 2. inserting desired quantizers, and 3. estimating the ranges for each quantizers.
# 4. **Quantized Models**: How to automate the first steps described above using 1) the `quantize_model` function and 2) the `QuantizationConfig`. This section also shows how to manually quantize custom and 3rd party modules.
# 5. **Quantizing 3rd Party models**: We show how we applied all of the above to quantize the huggingface OPT model.

# %%
import copy

from pprint import pprint

import torch

import fastforward as ff

# %% [markdown]
# # 1. Quantized Tensors
#
# We start our tutorial with the `ff.quantized_tensor.QuantizedTensor`. This datatype is a subclass of a `torch.Tensor` designed to hold quantized data. A `QuantizedTensor` can be quantized using any type of quantization (uniform quantization, dynamic quantization, vector quantization, ...) but we will focus on linear / uniform quantization.
#
# There are many kinds of quantization, but in this notebook we focus on integer quantization on a fixed per-tensor or per-channel grid.
#
# It is not required that you know the details of this (very common) quantization scheme, but if you want to know more please refer to [A White Paper on Neural Network Quantization (Nagel et al. 2021)](https://arxiv.org/abs/2106.08295)
#

# %% [markdown]
# ⏩ Let's start by creating some floating point data

# %%
in_features = 4

data = torch.rand(1, in_features) - 0.5
data

# %% [markdown]
# ⏩ Now, we quantize the data using 8-bit per-tensor quantization.

# %%
scale = torch.tensor([0.1])
num_bits = 8
quantized_data = ff.quantization.affine.quantize_per_tensor(data, num_bits=num_bits, scale=scale)

quantized_data

# %% [markdown]
# ✅ We can see that `quantized_data` is now a `QuantizedTensor`.
# This makes it very easy to see in FastForward if your data is actually quantized or not.
#
# ✅ This tensor both holds the actual data (of same shape as `data`) as well as the hyperparameters of the quantizer. For this specific case the only hyperparameter is the quantization scale, which we have set manually.
#
# ❌ Because this data is transformed to a new coordinate system, it is not easy to see what floating point values they represent.
#
# ⏩ For this purpose, we can dequantize the tensor, which we do below.

# %%
quantized_data.dequantize()

# %% [markdown]
# # 2. Quantizers
#
# In the previous paragraph, we saw above how a floating point tensor can be quantized.
#
# ❌ Quantization often involves hyperparameters which we do not know in advance.
#
# ✅ For this purpose, we can use `ff.nn.Quantizers`. These are `nn.Modules` that can quantize a data in their forward pass and can also be used to estimate or learn the quantization hyperparameters.
#
# ⏩ We create a `LinearQuantizer` now

# %%
quantizer = ff.nn.linear_quantizer.LinearQuantizer(num_bits=2)
quantizer

# %% [markdown]
# ⏩ Next, we try to quantize our data with our quantizer.

# %%
try:
    quantizer(data)
except ValueError as e:
    print("[ERROR]", e, "\n")

print(f"{quantizer.has_uninitialized_params=}")
print(f"{quantizer.quantization_range=}")  # min, max values that quantizer can represent.

# %% [markdown]
# ❌ We can see that our quantizer will not quantize any data just yet.
# The reason for this is that this specific quantizer has hyperparameters
# that need to be fitted before we can quantize any data. As a result, the quantization range is not yet set.
#
# ⏩ We could set `quantizer.quantization_range` directly, but we would need to know the desired quantization range `(min, max)`.
#
# ⏩ A more common approach is to use _range estimation_ to find the optimal range based on data. We do this below.

# %%
with ff.range_setting.estimate_ranges(quantizer, ff.range_setting.smoothed_minmax):
    quantizer(data)

print(f"{quantizer.has_uninitialized_params=}")
print(f"{quantizer.quantization_range=}")
print(f"{data.min()=} {data.max()=}")

# %% [markdown]
# ✅ We have now set the quantization range and the quantizer is initialized.
#
# ✅ We can see that the quantization range is the same as the range in the data batch.
#
# ⏩ We will now use our quantizer to quantize the data

# %%
quantized_data = quantizer(data)  # type: ignore[assignment]

quantized_data

# %% [markdown]
# # 3. Quantized Modules
#
# `Quantizers` typically don't exist in isolation. Instead, we would often like to quantize an entire model. In this section we show how to turn a module into a quantized module and what is happening under the hood. In the next section we show how to use convience methods for easier quantization of bigger models.
#
#
# ⏩ We start with a simple unquantized linear layer

# %%
out_features = 8

unquantized_linear = torch.nn.Linear(in_features, out_features)
print(unquantized_linear)

# %% [markdown]
# ⏩ In FastForward we use `ff.nn.QuantizedModule`s, they are drop-in replacements of `torch.nn.Module`s but additionally take care of quantization.
#
# ⏩ Most modules in `torch.nn` are mirrored with their quantized counterpart in `ff.nn`
#
#   - The goal of these quantized modules is that they behave exactly the same as their floating point counterparts, they have the same methods which have the same function signatures.
#   - The only difference is that we add quantizer children to the modules and we change the forward pass s.t. it performs quantized operations instead of floating point operations.
#   - ⚠️ If you do not find your desired module in `ff.nn`, you can either open an issue with us, or implement the layer yourself.
#
# ⏩ Let a closer look at the `ff.nn.QuantizedLinear` below.
#   - For now, we manually copied the weight data s.t. the `quantized_linear` matches the `unquantized_linear`. In the next section we will show convenience methods to automatically quantize modules in-place.

# %%
quantized_linear = ff.nn.QuantizedLinear(in_features, out_features)
quantized_linear.weight.data = unquantized_linear.weight.data.clone()
quantized_linear.bias.data = unquantized_linear.bias.data.clone()

print(quantized_linear)

# %% [markdown]
# ⏩ We can see that our QuantizedLinear has the same representation as the Linear, but instead there are four quantizer children added.
#   - In this case the bias_quantizer is `None` since this layer does not have a bias.
#
# ⏩ Observe that all quantizers are set to be `QuantizerStub`s. These are no-op placeholders that can be repaced with quantizers if desired.
#
# ⏩ Let's try to push data trough our `QuantizedLinear`.

# %%
try:
    quantized_output = quantized_linear(data)
except ff.exceptions.QuantizationError as e:
    print("[ERROR]", e, "\n")

# %% [markdown]
# ❌ We see we cannot push data trough our QuantizedLinear because `strict_quantization=True`.
#
#   - This flag tries to catch a common error-case in simulated quantization where no quantization is taking place, and the user is not aware of this.
#   - In our case, we have not assigned any quantizers, so the layer will behave as a floating point layer, which is not allowed if `strict_quantization=True`.
#
# ⏩ Let's temporarily disable the `strict_quantization` setting and see what happens when we call the `quantized_linear`.

# %%
with ff.strict_quantization(False):
    quantized_output = quantized_linear(data)

unquantized_output = unquantized_linear(data)

print(f"{unquantized_output=}")
print(f"{quantized_output=}")

# %% [markdown]
# ✅ Indeed, the `quantized_linear` is behaving exactly as the `unquantized_linear` as we have not specified any quantizers.
#
# ⏩ We will now assign quantizers to all of the quantizer fields

# %%
quantized_linear.input_quantizer = ff.nn.linear_quantizer.LinearQuantizer(num_bits=2)
quantized_linear.weight_quantizer = ff.nn.linear_quantizer.LinearQuantizer(num_bits=2)
quantized_linear.output_quantizer = ff.nn.linear_quantizer.LinearQuantizer(num_bits=2)
print(quantized_linear)

# %% [markdown]
# ⏩ As we know from the example above, we first need to initialize the quantizers by passing data trough. Let's do so.

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
print()

# %% [markdown]
# ✅ We can see that all the quantizers in our layer are initialized now.
#
# ⏩ We should now be able to call our `quantized_linear`. Let's do that!

# %%
unquantized_output = unquantized_linear(data)
quantized_output = quantized_linear(data)

print(f"{unquantized_output=}")
print()
print(f"{quantized_output=}")
print()
print(f"{quantized_output.dequantize()=}")

# %% [markdown]
# ✅ We can now see that our `quantized_linear` is behaving as expected:
#   - The output is a `QuantizedTensor`
#   - The dequantized output is close to the floating point output, but it is not identical due to quantization error.

# %% [markdown]
# # 4. Quantized Models
# In the previous section we showed how to quantize a module:
#   1. Turn an unquantized module into an unquantized module
#   2. Replace the desired QuantizerStubs with the desired Quantizers
#   3. Estimate the quantizer ranges by passing data trough the model.
#
# Performing step 1. and 2. were quite laborious in our above example. Since we have to repeat these steps for every layer in the model, we have created helper tools to automatate these tasks. In the next section we will show how to use `autoquant` tool to automatically replace all layers with their Quantized counterparts (step 1.) and how to use the `QuantizationConfig` to automatically insert quantizers into the model (step 2.).

# %% [markdown]
# ⏩ We start by making a simple unquantized MLP model.

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
# ### 4.1 Automatically replace `torch.nn.Modules` with their `ff.nn.QuantizedModule` counterparts
#
# ⏩ The `quantize_model` function can change a model in-place, recursively replacing all modules with their `QuantizedModule` counterparts.
#
#  - The way this function works is pretty simple. It just uses a dict that maps `torch.nn.Module` types to `ff.nn.QuantizedModule` types.
#
# ⏩ Let's have a look at this dictionary

# %%
ff.quantized_module_map()

# %% [markdown]
# ⏩ We will now run the `quantize_model` function.
#
# ⚠️ Note that the `ff.nn.quantize_model` changes the module classes in-place!
#
#   - Because we want to keep access to the full precision network for our comparison we first deepcopy the floating point model.

# %%
quantized_model = copy.deepcopy(unquantized_model)
ff.quantize_model(quantized_model)
quantized_model

# %% [markdown]
# ✅ We see that all modules in the model are now replaced with their quantized counterpart.
#
# ⏩ Since no quantizers are inserted yet, let's confirm that the `quantized_model` still behaves the same as the `unquantized_model`.

# %%
with ff.strict_quantization(False):
    quantized_output = quantized_model(data)

unquantized_output = unquantized_model(data)

print(f"{unquantized_output=}")
print(f"{quantized_output=}")

# %% [markdown]
# ### 4.2 Automatically inserting `ff.nn.Quantizer`s in the right place in the model.
#
# The `ff.QuantizationConfig` is a tool to automatically replace `QuantizerStubs` with `Quantizers`. It works by adding quantization rules.
#
# A quantization rule consists of two components:
#
#  1. A `query` determines to which layers the rule should be applied. Filtering is done using the `ff.mpath` library. Please see the tutorial on MPath for more information.
#  2. A quantizer class or factory. This determines how the quantizer is created. In the case of a Quantizer class, for each match a quantizer of that class is initialized using the provided kwargs. In the case of a factory function, the function receives the full name of the quantizer and the current quantizer at that location. This function is expected to return an intiailized quantizer.
#
# If multiple rules match a single quantizer, the rule that was added last takes priority.
#
# ⏩ We create our `QuantizationConfig` below, see if you can understand all the rules!

# %%
config = ff.QuantizationConfig()

# We want to quantize all weights in the model.
config.add_rule(
    "**/[quantizer:parameter/weight]",
    ff.nn.LinearQuantizer,
    num_bits=8,
    symmetric=True,
    granularity=ff.PerChannel(),
)

# We want to quantize all the outputs in the model, too.
config.add_rule(
    "**/[quantizer:activation/output]",
    ff.nn.LinearQuantizer,
    num_bits=8,
    symmetric=False,
    granularity=ff.PerTensor(),
)


# We only want to enable the input quantizer of the first layer, so that we can turn a floating point input into a quantized input.
# For the subsequent layers, the input will already be quantized because there will be an output quantizer in the layer before that.
def input_factory(name: str, current_quantizer: ff.nn.Quantizer) -> ff.nn.Quantizer:
    return ff.nn.LinearQuantizer(num_bits=8, symmetric=False, granularity=ff.PerTensor())


config.add_rule("0/[quantizer:activation/input]", input_factory)

config

# %% [markdown]
# ✅ Note that in the rule for input quantizers we could have directly specified the quantizer class, but we instead show an example
# of using a factory function. This function receives both the name and the current quantizer at the given location. This
# can be an initialized quantizer or a QuantizerStub.
#
# ⏩ Applying the QuantizationConfig to the model is very simple:

# %%
config.initialize(quantized_model)
quantized_model

# %% [markdown]
# ✅ Observe that the quantizers in our quantized model are now setup up as expected.
#
# ⏩ All we have to do now is estimate the ranges for the quantizers, and we can use the quantized model!

# %%
with ff.range_setting.estimate_ranges(quantized_model, ff.range_setting.smoothed_minmax):
    quantized_model(data)

quantized_model(data)


# %% [markdown]
# # 4.3 Quantizing Custom Modules: Manual Quantization
#
# Your model might not only consist of `torch.nn.Modules`, but also contain 3rd party or custom modules. Because FastForward does not have fully automated quantization yet, trying to convert these modules using `quantize_model` does not work. Let us build such a custom module:


# %%
class MySelfAttentionLayer(torch.nn.Module):
    def __init__(self, feature_size):
        print("Calling MySelfAttentionLayer.__init__")
        super().__init__()
        self.feature_size = feature_size

        # Linear transformations for Q, K, V from the same source
        self.key = torch.nn.Linear(feature_size, feature_size)
        self.query = torch.nn.Linear(feature_size, feature_size)
        self.value = torch.nn.Linear(feature_size, feature_size)

    def forward(self, x):
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

# %%
my_quantized_layer = copy.deepcopy(my_unquantized_layer)
try:
    ff.quantize_model(my_quantized_layer)
except ff.exceptions.QuantizationError as e:
    print("[ERROR]", e, "\n")

print("ff.quantized_module_map():")
pprint(ff.quantized_module_map())


# %% [markdown]
# ❌ Observe that `ff.quantize_model` does not work, because it does not know which class `MySelfAttentionLayer` should be mapped to.
#
# ⏩ For now, we have to manually define the quantized equivalent of `MySelfAttentionLayer`, we will show how to do so in the next section:


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
# ⏩ Notice that we made two changes to the model:
#   1. We have re-implemented the forward pass, replacing all operations from torch.nn.functional with their FastForward quantized equivalent.
#      1. ❌ Untill autoquant is implemented in FastForward, this means we manually need to duplicate the code from the forward pass.
#      2. ⚠️ NOTE: Some of the functionals might be hidden inside a function that is called in your forward pass, make sure to also rewrite those cases.
#      3. ⚠️ If you are adopting a 3rd party class, you will need to copy-paste the code from the forward pass. Make sure to also freeze the dependency so that your rewritten module will not diverge once the package is updated!
#      4. In order to use the quantized functionals, we have added Quantizers to the model:
#
#   2.  We added an `__init_quantization__` method that adds the `QuantizerStubs` which could be used later for quantization.
#       1. ✅ We do not have to copy-paste any code from the `__init__` function
#       2. ✅ As we will see below, `__init_quantization__` can be used both for initializing a `QuantizedModule` from scratch, or to convert a `Module` to a `QuantizedModule`
#
# ⏩ Let's have a look to see how our `MyQuantizedSelfAttentionLayer` behaves when initialized from scratch:

# %%
new_quantized_layer = MyQuantizedSelfAttentionLayer(num_features)
new_quantized_layer

# %% [markdown]
# ⏩ Observe that:
#   1. `MySelfAttentionLayer.__init__` is first called, initializing the layer using the logic of the unquantized base layer.
#   2. `MyQuantizedSelfAttentionLayer.__init_quantization__` is then called, inserting the quantizer stubs.
#   3. The children modules are not converted to their quantized counterparts when initializing from scratch.
#
# ⏩ In practice, we will typically not initialize quantized modules from scratch, but we will rather take a floating point model and recursively convert all it's submodules.
#
# ⏩ We will now look how `MyQuantizedSelfAttentionLayer` behaves when converted using the `quantize_model` function. First, let's look at the `quantized_module_map`:

# %%
print("ff.quantized_module_map():")
pprint(ff.quantized_module_map()[MySelfAttentionLayer])

# %% [markdown]
# ✅ Note that `MySelfAttentionLayer` automatically appeared in the `quantized_module_map`!
#
# ⚠️ All subclasses of `QuantizedModel` are automatically found in `fastforward.nn.quantized_module.quantized_module_map()`, but this requires the classes to be imported. If your class does not show up, make sure to import it, or use the `extra_conversion` argument if you want to override any mappings in the `quantized_module_map`.
#
# ⏩ We will now look how `MyQuantizedSelfAttentionLayer` behaves when converted using the `quantize_model` function.

# %%
my_quantized_layer = copy.deepcopy(my_unquantized_layer)
ff.quantize_model(my_quantized_layer)

my_quantized_layer

# %% [markdown]
# ⏩ Observe that:
#   1. Since we convert an existing layer, `MySelfAttentionLayer.__init__` is not called again.
#   2. The class of our module is changed from `MySelfAttentionLayer` to `MyQuantizedSelfAttentionLayer`.
#   3. `MyQuantizedSelfAttentionLayer.__init_quantization__` is still called, inserting the quantizer stubs into the previously unquantized layer.
#   4. The children modules are also converted to their quantized counterparts by calling `MyQuantizedSelfAttentionLayer.quantize_children`.

# %% [markdown]
# # 5. Quantizing 3rd party models (Huggingface OPT)
# Based on the tutorial above you should be able to manually quantize any model. We will now show how we quantized the OPT model in our
# [fast-models benchmark repository](https://morpheus-gitlab.qualcomm.com/jpeters/fast-models)
#
# The process of adopting the model consists of the following steps (which are explained in the notebook above):
#
# 1. **Downloading the existing model code** from the huggingface library.
#
#    1. ⚠️ Because we will both be using huggingface as a library, but also copy-paste huggingface code we need to freeze our huggingface dependency so that it matches the version we copied the code from.
#    2. ⏩ Have a look at [this commit](https://morpheus-gitlab.qualcomm.com/jpeters/fast-models/-/commit/ab1a10783f54ce3fc78f17f8884ff64f12705489) to see how we conducted this step.
#
# 2. **Cleaning the existing model code**
#
#    1. We remove everything except the modules that are used in the (OPT) model we aim to quantize.
#    2. Of those modules, we only keep the forward pass.
#    3. ⏩ Have a look at [this commit](https://morpheus-gitlab.qualcomm.com/jpeters/fast-models/-/commit/2d6150b7c6144d26b0cdf8565de2f2e9a4da926d) to see how we conducted this step.
#
# 3. **Modifying the existing model code**
#    1. We change all the functionals in the forward pass to their quantized counterparts.
#
#    ⚠️ NOTE: Sometimes the functionals might be hidden inside a function (such as the `quantized_masked_attention` function in the OPT example), take care to also detect and convert those.
#
#    2. We add an `__init_quantization__` method that adds the required quantizers which are used in the quantized functionals.
#    3. ⏩ Have a look at [this commit](https://morpheus-gitlab.qualcomm.com/jpeters/fast-models/-/commit/9c9b6d2bf25b468004e7208ebaddde5b81c61e5e) to see how we conducted this step.
#
# 5. Adding code to insert the QuantizerStubs by adding a `QuantizationConfig`
#    1. We make a `QuantizationConfig` that determines where to insert quantizers based on our experiment settings.
#    2. ⏩ Have a look at [this commit](https://morpheus-gitlab.qualcomm.com/jpeters/fast-models/-/commit/adaac7a74fde8f4a8c27c35fc7047f54637a8051) to see how we conducted this step.
#
# 6. Running the full benchmark experiments
#    1. ⏩ Have a look at [this commit](https://morpheus-gitlab.qualcomm.com/jpeters/fast-models/-/commit/33e70bd8c690c7bca9a81f7f0a2e10f8d2a6b583) to see how we conducted this step.


# %% [markdown]
# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
