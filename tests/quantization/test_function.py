# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.


import pytest
import torch

import fastforward as ff

from fastforward.quantization.affine import StaticAffineQuantParams, quantization_context
from fastforward.quantization.affine.function import AffineQuantParams
from fastforward.quantization.function import (
    QuantizationContext,
    QuantizationFunction,
)
from fastforward.quantized_tensor import QuantizedTensor


class MockQuantizationFunction(QuantizationFunction[StaticAffineQuantParams]):
    @classmethod
    def quantize(cls, data: torch.Tensor, params: StaticAffineQuantParams) -> ff.QuantizedTensor:
        quantized_data = data * params.scale
        context = QuantizationContext(cls, params)
        return ff.QuantizedTensor(quantized_data, context)

    @classmethod
    def dequantize(cls, data: torch.Tensor, params: StaticAffineQuantParams) -> torch.Tensor:
        return data / params.scale


@pytest.fixture
def quant_context() -> QuantizationContext[StaticAffineQuantParams]:
    scale = torch.tensor([0.1], requires_grad=True)
    context = quantization_context(scale=scale, offset=None, granularity=ff.PerTensor(), num_bits=3)
    return context.with_changes(quantization_fn=MockQuantizationFunction)


def test_quantization_function_attach(quant_context: QuantizationContext[AffineQuantParams]):
    data = torch.randn((3, 3))
    quant_data = quant_context.attach(data)
    assert isinstance(quant_data, QuantizedTensor)
    assert quant_data.quant_func is MockQuantizationFunction
    torch.testing.assert_close(quant_data.raw_data, data)


def test_quantization_context_clone(
    quant_context: QuantizationContext[StaticAffineQuantParams],
):
    cloned_context = quant_context.clone_parameters()
    args = quant_context.quantization_params
    cloned_args = cloned_context.quantization_params

    assert isinstance(args.scale, torch.Tensor)
    assert isinstance(cloned_args.scale, torch.Tensor)
    assert args.scale is not cloned_args.scale
    assert args.scale.data_ptr != cloned_args.scale.data_ptr
    torch.testing.assert_close(args.scale, cloned_args.scale)
    assert cloned_args.scale.grad_fn is not None


def test_quantization_context_detach_arguments(
    quant_context: QuantizationContext[StaticAffineQuantParams],
):
    detached_context = quant_context.detach_parameters()
    args = quant_context.quantization_params
    cloned_args = detached_context.quantization_params

    assert isinstance(cloned_args.scale, torch.Tensor)
    assert args.scale is not cloned_args.scale
    torch.testing.assert_close(args.scale, cloned_args.scale)
    assert cloned_args.scale.grad_fn is None


def test_quantization_context_to(
    quant_context: QuantizationContext[StaticAffineQuantParams],
):
    moved_func = quant_context.to("meta")
    assert isinstance(moved_func.quantization_params.scale, torch.Tensor)
    assert moved_func.quantization_params.scale.device == torch.device("meta")
