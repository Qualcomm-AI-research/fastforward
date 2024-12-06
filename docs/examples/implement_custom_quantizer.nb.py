# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

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

# # Implementing a custom Quantizer
# When implementing a new quantizer, we can identify two main components in the
# _FastForward_ stack: **quantizers** (torch modules) and **quantization
# functions** (static classes).
#
#
# ## Quantizer Module
# `Quantizer` classes should implement the `quantize` method, taking a
# `Tensor` as input and returning a `QuantizedTensor`.
#
# To correctly quantize data, the `Quantizer` should delegate to an
# implementation of a class in the `QuantizationFunction` hierarchy.
#
# _Quantizers_ modules normally lives in the `fastforward.nn` package.
#
#
# ## Quantization Function
# Each `QuantizationFunction` class represents a _class of quantization functions_.
# QuantizationFunctions normally lives in the `fastforward.quantization` package.
#
# These classes should implement `quantize` and `dequantize` methods and do
# the actual computation that transforms a float tensor to a `QuantizeTensor`
# and back to the float format.
#
# _QuantizationFunctions_ are static: they are containers of the functions
# that operate on the data. For this reason you can't create objects of type
# _QuantizationFunction_.
# On the other hand, calling `QuantizationFunction.bind(...)` one can bind
# parameters to a `QuantizationFunction` class, storing these information in
# an object of type `BoundQuantizationFunction`.
#
# More importantly, you can quantize an input tensor simply using
# the method `QuantizationFunction.apply(...)`.
#
# Once a tensor is quantized, a `BoundQuantizationFunction` containing all
# the quantization parameters is attached to it so that one can easily
# interpret the data or dequantize back to well-known data representation
# (floating point) simply calling `dequantize(...)` method on the tensor.
#
#
# ### Quantization _Autograd_ Function
# `QuantizationAutogradFunction` is a variant of
# `QuantizationFunction` with the extra functionality of a custom backward pass
# implementation for the whole `quant&dequant` operation. This enables one to
# optimize code for the backward pass and avoid unneeded computation.
#
# Instead, `QuantizationFunction` backward rely on the standard _torch autograd_,
# applied to the `quantize` and `dequantize` functions, doing some extra
# computation.
#
#
# ![Class diagram for Quantizers and Quantization Functions](../imgs/custom_quantizer_class_diagram.png)
#
# To implement the quantizer, one should at least implement a custom Quantizer
# class inheriting from `fastforward.nn.Quantizer` and a custom
# `QuantizationFunction` class inheriting from
# `fastforward.quantization.function.QuantizationFunction` or
# `fastforward.quantization.function.QuantizationAutogradFunction`.
#
#
#
# ## Minimal Example: implementing a basic linear quantizer
#
# In this example we show how to create a new quantizer from scratch with
# minimum effort. To do so, we first implement a custom
# `QuantizationFunction` in its simplest version (i.e. without custom
# backward pass), and then a custom `Quantizer` class.
#
# We can implement our custom `QuantizationFunction` with the following code:

# +
from typing import Callable

import torch

from torch import Tensor

from fastforward import QuantizedTensor, estimate_ranges
from fastforward.common import ensure_tensor
from fastforward.nn.quantizer import Quantizer
from fastforward.quantization import granularity
from fastforward.quantization.affine import parameters_for_range
from fastforward.quantization.function import QuantizationFunction
from fastforward.quantization.ste import round_ste
from fastforward.range_setting.minmax import RunningMinMaxRangeEstimator


class MyQuantizationFunction(QuantizationFunction):
    """My custom quantization function class"""

    @staticmethod
    def quantize(  # type: ignore[override]
        data: Tensor, num_bits: int, scale: float, offset: float | None = None
    ) -> Tensor:
        if offset is None:
            offset = 0.0
        min_int = -(2 ** (num_bits - 1))
        max_int = -min_int - 1
        return torch.clamp(round_ste(data / scale - offset), min_int, max_int)

    @staticmethod
    def dequantize(  # type: ignore[override]
        quant_data: Tensor, num_bits, scale: float, offset: float | None = None
    ) -> Tensor:
        if offset is not None:
            _offset = torch.round(torch.tensor(offset, device=quant_data.device))
            return (quant_data + _offset) * scale
        else:
            return quant_data * scale


# -

# With this, can already use our `MyQuantizationFunction` to quantize
# a floating point tensor into a `QuantizedTensor` storing integer data:

# +
# Create new random data tensor
data = torch.rand(1024, 1024, requires_grad=True)

# Quantize and dequantize data
bits = 8
qf_data = MyQuantizationFunction.apply(data, num_bits=bits, scale=2 ** (-bits), offset=128)
dqf_data = qf_data.dequantize()

# Compute quantization error
qf_error = torch.abs(data - dqf_data)
print(f"The maximum quantization error found on input data is: {qf_error.max()} ")


# -

#
# Quantizing tensors like this, is not very convenient. But we can write
# unit tests for `MyQuantizationFunction` following this approach.
#
# A much better way to use the quantization function, is to implement
# our custom `MyQuantizer` that will use our custom quantization function
# internally. Instead of passing `scale` and `offset` to the constructor of
# the quantizer, we will let the user of the object to set the **quantization
# range**, delegating to the class internals how to modify `scale` and
# `offset` accordingly.
#
# To do so, we will implement the `fastforward.range_setting.RangeSettable`
# protocol, i.e. the following methods:
#
# - `quantization_range` getter: return to the caller the quantization range
#   on which our quantizer can operate without incurring in clipping.
# - `quantization_range` setter: given a min-max range, this method will set
#   our quantization parameters accordingly.
#

# +


class MyQuantizer(Quantizer):
    """My custom quantizer class"""

    def __init__(self, num_bits: int, symmetric: bool = False, device: torch.device | str = "cpu"):
        super().__init__()
        self.num_bits = num_bits
        self.scale = torch.nn.Parameter(torch.tensor([0.0], device=device))
        self.offset = None if symmetric else torch.nn.Parameter(torch.tensor([0.0], device=device))

    def extra_repr(self) -> str:
        extra = f"num_bits={self.num_bits}, symmetric={self.symmetric}, scale={self.scale}, offset={self.offset}"
        return super().extra_repr() + extra

    def quantize(self, data: Tensor) -> Tensor:
        return MyQuantizationFunction.apply(
            data, num_bits=self.num_bits, scale=self.scale, offset=self.offset
        )

    @property
    def granularity(self) -> granularity.Granularity:
        return granularity.PerTensor()

    @property
    def symmetric(self) -> bool:
        return "offset" in self._buffers or self.offset is None

    @property
    def quantization_range(self) -> tuple[Tensor | float | None, Tensor | float | None]:
        offset = self.offset or 0.0
        range_min = (integer_minimum(self.num_bits) + offset) * self.scale
        range_max = (integer_maximum(self.num_bits) + offset) * self.scale
        return range_min, range_max

    @quantization_range.setter
    def quantization_range(self, quant_range: tuple[Tensor | float, Tensor | float]) -> None:
        if not isinstance(quant_range, (tuple, list)):
            raise ValueError(
                "Tried to set quantization range with a single value. A 2-tuple is expected"
            )
        if not len(quant_range) == 2:
            raise ValueError(
                f"Tried to set quantization range with {len(quant_range)}-tuple. A 2-tuple is expected"
            )

        min, max = (ensure_tensor(t, device=self.scale.device) for t in quant_range)
        with torch.no_grad():
            scale, offset = parameters_for_range(
                min, max, self.num_bits, self.symmetric, allow_one_sided=False
            )
            self.scale.copy_(scale)
            if self.offset is not None and offset is not None:
                self.offset.copy_(offset)


def integer_minimum(num_bits: float) -> float:
    return -(2 ** (num_bits - 1))


def integer_maximum(num_bits: float) -> float:
    return -integer_minimum(num_bits) - 1


# -

# Now we have a more user-friendly interface:
#

# +

# Create new random data tensor
data = torch.randn(128, 64, 4, requires_grad=True)

# Instantiate quantizer:
quantizer = MyQuantizer(num_bits=8, symmetric=False)

# Set quantization range manually:
quantizer.quantization_range = data.min(), data.max()

# Quantize and dequantize data
q_data = quantizer(data)
dq_data = q_data.dequantize()

# Compute quantization error
q_error = torch.abs(data - dq_data)
print(f"The maximum quantization error found on input tensor is: {q_error.max()} ")

# -

# Moreover, we are now able to use standard `estimate_range` functionality
# over our custom quantizer:

# +

# Create new random data tensor
data = torch.randn(10, 10, requires_grad=True)

# Instantiate quantizer:
quantizer = MyQuantizer(num_bits=8, symmetric=False)

# Estimate range
with estimate_ranges(quantizer, RunningMinMaxRangeEstimator):
    q_data = quantizer(data)

# Compute quantization error over the calibration data
dq_data = q_data.dequantize()
q_error = torch.abs(data - dq_data)
print(f"The maximum quantization error found on calibraiton data is: {q_error.max()} ")

# Create new random data tensor and quantize with range fixed previously
data = torch.randn(10, 10, requires_grad=True)
q_data = quantizer(data)
dq_data = q_data.dequantize()
q_error = torch.abs(data - dq_data)
print(f"The maximum quantization error found on new data is: {q_error.max()} ")


# -

#
# Certain range estimator will also require you to implement the
# `fastforward.range_setting.SupportsRangeBasedOperator` protocol. In this
# case, we just need to add the following method to `MyQuantizer`:
#

# +


# class MyQuantizer:
#    ...
def operator_for_range(
    self, min_range: Tensor, max_range: Tensor, data_shape: torch.Size
) -> Callable[[torch.Tensor], QuantizedTensor]:
    scale, offset = parameters_for_range(
        min_range, max_range, self.num_bits, self.symmetric, allow_one_sided=False
    )
    return MyQuantizationFunction.bind(num_bits=self.num_bits, scale=scale, offset=offset)
