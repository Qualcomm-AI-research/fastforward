# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Callable

import fastforward as ff
import pytest
import torch

from fastforward.quantization.random import random_quantized
from fastforward.quantized_tensor import QuantizedTensor


@pytest.mark.parametrize("op_name", ["ones_like", "zeros_like", "full_like", "empty_like"])
def test_functional_like_ops_require_output_quantizer_under_strict(op_name: str) -> None:
    # GIVEN a quantized tensor and strict quantization
    quantized_data = random_quantized((4, 8), scale=0.2)
    functional_op = getattr(ff.nn.functional, op_name)
    args = (quantized_data, 5.0) if op_name == "full_like" else (quantized_data,)

    # WHEN a functional *_like op is called without an output_quantizer under strict
    # THEN it raises, like every other quantized operator (see fallback.linear).
    with ff.strict_quantization(True):
        with pytest.raises(ff.exceptions.QuantizationError):
            functional_op(*args)


@pytest.mark.parametrize(
    ("op_name", "reference"),
    [
        ("ones_like", lambda shape: torch.ones(shape)),
        ("zeros_like", lambda shape: torch.zeros(shape)),
        ("full_like", lambda shape: torch.full(shape, 5.0)),
    ],
)
def test_functional_like_ops_apply_output_quantizer(
    op_name: str, reference: Callable[[torch.Size], torch.Tensor]
) -> None:
    # GIVEN a quantized tensor and a real output quantizer
    quantized_data = random_quantized((4, 8), scale=0.2)
    quantizer = ff.nn.LinearQuantizer(num_bits=4)
    quantizer.quantization_range = (-1, 1)

    functional_op = getattr(ff.nn.functional, op_name)
    args = (quantized_data, 5.0) if op_name == "full_like" else (quantized_data,)

    # WHEN the functional op is called with an output_quantizer
    with ff.strict_quantization(True):
        out = functional_op(*args, output_quantizer=quantizer)

    # THEN the constant tensor is quantized by that quantizer.
    assert isinstance(out, QuantizedTensor)
    assert out.shape == quantized_data.dequantize().shape
    expected = quantizer(reference(quantized_data.dequantize().shape))
    torch.testing.assert_close(out.dequantize(), expected.dequantize())


@pytest.mark.parametrize(
    ("op_name", "reference"),
    [
        ("ones_like", lambda shape: torch.ones(shape)),
        ("zeros_like", lambda shape: torch.zeros(shape)),
        ("full_like", lambda shape: torch.full(shape, 5.0)),
    ],
)
def test_functional_like_ops_without_quantizer_dequantize_under_non_strict(
    op_name: str, reference: Callable[[torch.Size], torch.Tensor]
) -> None:
    # GIVEN a quantized tensor and no output quantizer
    quantized_data = random_quantized((4, 8), scale=0.2)
    functional_op = getattr(ff.nn.functional, op_name)
    args = (quantized_data, 5.0) if op_name == "full_like" else (quantized_data,)

    # WHEN a functional *_like op is called without an output_quantizer under non-strict
    # THEN it returns a plain float tensor built from the dequantized input.
    with ff.strict_quantization(False):
        out = functional_op(*args)

    assert type(out) is torch.Tensor
    assert out.shape == quantized_data.dequantize().shape
    torch.testing.assert_close(out, reference(quantized_data.dequantize().shape))


@pytest.mark.parametrize("op_name", ["ones_like", "zeros_like", "full_like", "empty_like"])
def test_functional_like_ops_honor_shape_dtype_device_contract(op_name: str) -> None:
    # Every *_like op must produce an output whose shape, dtype and device match the
    # dequantized input. empty_like returns uninitialized memory, so its values are
    # undefined; assert only the metadata contract that all of them guarantee.
    quantized_data = random_quantized((4, 8), scale=0.2)
    functional_op = getattr(ff.nn.functional, op_name)
    args = (quantized_data, 5.0) if op_name == "full_like" else (quantized_data,)

    # WHEN a functional *_like op is called without an output_quantizer under non-strict
    with ff.strict_quantization(False):
        out = functional_op(*args)

    # THEN it returns a plain float tensor matching the input's shape/dtype/device.
    assert type(out) is torch.Tensor
    assert out.shape == quantized_data.dequantize().shape
    assert out.dtype == quantized_data.dequantize().dtype
    assert out.device == quantized_data.dequantize().device


@pytest.mark.parametrize("op_name", ["ones_like", "zeros_like", "full_like", "empty_like"])
def test_functional_like_ops_forward_dtype_kwarg(op_name: str) -> None:
    quantized_data = random_quantized((4, 8), scale=0.2)
    functional_op = getattr(ff.nn.functional, op_name)
    args = (quantized_data, 5.0) if op_name == "full_like" else (quantized_data,)

    with ff.strict_quantization(False):
        out = functional_op(*args, dtype=torch.float16)

    assert out.dtype == torch.float16
    assert out.shape == quantized_data.dequantize().shape


@pytest.mark.parametrize("op_name", ["ones_like", "zeros_like", "full_like", "empty_like"])
def test_functional_like_ops_forward_device_kwarg(op_name: str) -> None:
    quantized_data = random_quantized((4, 8), scale=0.2)
    functional_op = getattr(ff.nn.functional, op_name)
    args = (quantized_data, 5.0) if op_name == "full_like" else (quantized_data,)

    with ff.strict_quantization(False):
        out = functional_op(*args, device="cpu")

    assert out.device == torch.device("cpu")
    assert out.shape == quantized_data.dequantize().shape


@pytest.mark.parametrize("op_name", ["ones_like", "zeros_like", "full_like"])
def test_functional_like_ops_with_stub_quantizer_stay_float_under_strict(op_name: str) -> None:
    # GIVEN a quantized tensor and a QuantizerStub (a pass-through placeholder)
    quantized_data = random_quantized((4, 8), scale=0.2)
    stub = ff.nn.QuantizerStub()
    functional_op = getattr(ff.nn.functional, op_name)
    args = (quantized_data, 5.0) if op_name == "full_like" else (quantized_data,)

    # WHEN called under strict quantization with a stub as output_quantizer
    # THEN no error is raised (the stub is not None), but because the stub passes data
    # through unchanged the result is a plain float tensor, not a QuantizedTensor.
    with ff.strict_quantization(True):
        out = functional_op(*args, output_quantizer=stub)

    assert type(out) is torch.Tensor
    assert out.shape == quantized_data.dequantize().shape
    assert out.dtype == torch.float
