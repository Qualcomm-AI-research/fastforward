# ---
# jupyter:
#   jupytext:
#     formats: py:percent
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
# # **MPath**: A Query-Based Method for Selecting PyTorch Submodules
#
# **MPath** is a utility library that is part of the `fastforward` package (`fastforward.mpath`). It simplifies the process of selecting PyTorch submodules using queries. These queries can be either query strings or manually constructed queries composed of several query fragments.
#
# For example, MPath can help you find all `Linear` modules that are part of a `decoder` module.
#
# ## Examples
#
# Let's look at a few examples. First, we'll create a PyTorch module (with submodules) that we'll refer to as `my_module`:

# %%
import fastforward as ff
import torch


# %%
def module(**kwargs: torch.nn.Module) -> torch.nn.ModuleDict:
    return torch.nn.ModuleDict(kwargs)


def linear() -> torch.nn.Linear:
    return torch.nn.Linear(10, 10)


def conv() -> torch.nn.Conv2d:
    return torch.nn.Conv2d(3, 3, 3)


my_module = module(
    layer1=module(sublayer1=linear(), sublayer2=conv()),
    layer2=module(sublayer1=linear(), activation=torch.nn.ReLU(), sublayer2=conv()),
    layer3=module(sublayer1=linear(), sublayer2=conv()),
    layer4=module(
        sublayer1=linear(),
        sublayer2=module(first=linear(), second=conv(), last=module(only=linear())),
    ),
)
my_module

# %% [markdown]
# ### Finding `Linear` Modules
#
# Let's try to find all `Linear` modules in `my_module`:

# %%
ff.mpath.search("**/[cls:torch.nn.Linear]", my_module)

# %% [markdown]
# When we use `mpath.search`, it returns an `MPathCollection`. This collection contains submodules in `my_module` that match our query string.
#
# In the query string:
# - `**` means "match zero or more modules"
# - `[cls:torch.nn.Linear]` matches exactly one module of type `torch.nn.Linear`
#
# Altarnatively, we could also search for all `Linear` layers specifically within the `layer4` submodule:

# %%
ff.mpath.search("layer4/**/[cls:torch.nn.Linear]", my_module)

# %% [markdown]
# In this example, we included the double wildcard `**` to match _any_ module within the `layer4` submodule.
#
# Alternatively, we could use a single wildcard `*`, which means "match exactly one module". This would result in finding only `layer4.sublayer2.first` in our collection:

# %%
ff.mpath.search("layer4/*/[cls:torch.nn.Linear]", my_module)

# %% [markdown]
# Lastly, if we don't use a wildcard at all, we will only match the `Linear` layers that are direct children of `layer4`:

# %%
ff.mpath.search("layer4/[cls:torch.nn.Linear]", my_module)

# %% [markdown]
# ## Query Strings
#
# By default, a query string is composed of one or multiple module names separated by `/` to indicate hierarchy. For example: `decoder/attention/q_mapping`.
#
# However, MPath queries are more powerful and come with the following three options out of the box. Here are examples of each:
#
# - `[cls:quantified name]` or `[class:quantified name]`: Matches a module if it is an instance of the class identified by the quantified name.
# - `[re:regex pattern]` or `[regex:regex pattern]`: Matches a module if its attribute name on the parent module fully matches the regex pattern.
# - `~`: Matches a module that does **not** match the specified criteria.
#
# ### Class or Instance-Based Matching

# %%
ff.mpath.search("layer4/*/[cls:torch.nn.Linear]", my_module)

# %% [markdown]
# ### Regex based matching

# %%
ff.mpath.search(
    r"[re:layer[12\]]/sublayer1", my_module
)  # we have to escape ']' in the regex because the regex pattern is '[' and ']' delimited

# %% [markdown]
# ### Inverted matching

# %%
ff.mpath.search("layer2/~[cls:torch.nn.Linear]", my_module)

# %% [markdown]
# ### Query String Extension
#
# You can extend query strings and register your own extensions. A good starting point is the implementation of `fastforward.mpath.fragments.RegexPathSelectorFragment` or `fastforward.mpath.fragments.ClassSelectorFragments`. These examples are registered in `fastforward.mpath`.

# %% [markdown]
# # __MPath__ for Quantization Initialization

# %% [markdown]
# ## Quantizer Initialization
#
# Quantizer initialization, which is the process of introducing concrete quantizers to the model, can be achieved using MPath.
#
# First, let's turn `my_module` into a quantization-ready module. This means converting all modules in `my_module` to ones that can operate in a quantization setting. We use `fastforward.quantize_model` for this purpose.

# %%
ff.quantize_model(my_module)

# %% [markdown]
# Let's say we want to initialize all output quantizers for linear layers to 4-bit per-tensor linear quantizers. First, let's find all the relevant quantizers:

# %%
quantizers = ff.find_quantizers(my_module, "**/[cls:torch.nn.Linear]/[quantizer:activation/output]")
quantizers

# %% [markdown]
# Note that instead of using `fastforward.mpath.search`, we are using `ff.find_quantizers`. This returns a `QuantizerCollection` instead of an `MPathCollection`. Members of this collection are always quantizers. It also supports a convenient method for initializing quantizers. Let's do that now:

# %%
quantizers.initialize(ff.nn.LinearQuantizer, num_bits=4, granularity=ff.PerTensor())
quantizers

# %% [markdown]
# In the example above, we created a `fastforward.nn.LinearQuantizer` for each element in the `QuantizerCollection` using the provided keyword arguments. All the `QuantizerStub`s from the initial `QuantizerCollection` have now been replaced by the newly defined `Quantizer` type.
#
# This change is reflected in the module representation below, where the output layers are now `LinearQuantizer`s.

# %%
my_module


# %% [markdown]
# ## Quantizer Tags
#
# In the example above, we used the quantizer tag system to match specific types of quantizers. This system uses the format `[quantizer:<tag>(, <tag>)*]` or `[qtag:<tag>(, <tag>)*]`.
#
# ### How It Works:
# - **Tag Assignment**: Tags are added to `QuantizerStub` when they are created and are assigned to any quantizer that replaces them.
# - **Easy Matching**: This allows us to easily find quantizers that match a certain tag.
#
# For example, we create the following non-quantized module and its quantized counterpart:


# %%
class MyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return data * 2


class QuantizedMyModule(ff.nn.QuantizedModule, MyModule):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.input_quantizer = ff.nn.QuantizerStub("my_tag_hierarchy/my_tag/input")
        self.output_quantizer = ff.nn.QuantizerStub("my_tag_hierarchy/my_tag/output")

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        data = self.input_quantizer(data)
        return self.output_quantizer(data * 2)


# %% [markdown]
# ## Hierarchical Tags
#
# The tags we create can be hierarchical. For example, consider the hierarchy `my_tag_hierarchy -> my_tag -> {input, output}`.
#
# ### How Tag Matching Works:
# - **Root Element Matching**: A tag like `my_tag_hierarchy` will match any tag that has `my_tag_hierarchy` as its root element.
# - **Full Path Matching**: A tag like `my_tag_hierarchy/my_tag` requires both the first and second elements to match.
#
# Continuing our example, we construct a module and use our newly created tags to obtain a `QuantizerCollection` that contains both quantizers.

# %%
my_quantized_module = QuantizedMyModule()

ff.find_quantizers(my_quantized_module, "**/[quantizer:my_tag_hierarchy/my_tag]")

# %% [markdown]
# Alternatively, we can only match the input quantizer:

# %%
ff.find_quantizers(my_quantized_module, "**/[quantizer:my_tag_hierarchy/my_tag/input]")

# %% [markdown]
# To emphasize our earlier point about hierarchy matching, note that using `input` as a tag alone will not match any quantizer. This will not raise an error; it will simply result in an empty `QuantizerCollection`.

# %%
ff.find_quantizers(my_quantized_module, "**/[quantizer:input]")

# %% [markdown]
# Lastly, note that there is a difference between the module hierarchy and the tag hierarchy in a query string. We can mix tag and module hierarchy queries.
#
# For example, `top_module/sub_module/**/[quantizer:parameter/weight]` will match all quantizers in `top_module.sub_module` that have the `parameter/weight` tag.

# %% [markdown]
# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
