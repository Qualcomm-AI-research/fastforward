# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


import dataclasses

import fastforward as ff
import pytest
import torch

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


def test_quantization_parameter_apply(
    quant_context: QuantizationContext[AffineQuantParams],
) -> None:
    # GIVEN a QuantizationParameters instance
    quant_params = quant_context.quantization_params

    # WHEN we apply an identity function to it (should not modify any attribute)
    new_quant_params = quant_params._apply(lambda x: x)

    # THEN the returned QuantizationParameters is a new instance
    assert quant_params is not new_quant_params

    # THEN the returned parameter attributes are references to the original
    assert quant_params.scale is new_quant_params.scale  # type: ignore[attr-defined]


def test_quantization_function_attach(
    quant_context: QuantizationContext[AffineQuantParams], _seed_prngs: int
) -> None:
    data = torch.randn((3, 3))
    quant_data = quant_context.attach(data)
    assert isinstance(quant_data, QuantizedTensor)
    assert quant_data.quant_func is MockQuantizationFunction
    torch.testing.assert_close(quant_data.raw_data, data)


def test_quantization_context_clone(
    quant_context: QuantizationContext[StaticAffineQuantParams],
) -> None:
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
) -> None:
    detached_context = quant_context.detach_parameters()
    args = quant_context.quantization_params
    cloned_args = detached_context.quantization_params

    assert isinstance(cloned_args.scale, torch.Tensor)
    assert args.scale is not cloned_args.scale
    torch.testing.assert_close(args.scale, cloned_args.scale)
    assert cloned_args.scale.grad_fn is None


def test_quantization_context_to(
    quant_context: QuantizationContext[StaticAffineQuantParams],
) -> None:
    moved_func = quant_context.to("meta")
    assert isinstance(moved_func.quantization_params.scale, torch.Tensor)
    assert moved_func.quantization_params.scale.device == torch.device("meta")


def test_quantization_context_contiguous_parameters(
    quant_context: QuantizationContext[StaticAffineQuantParams],
) -> None:
    # Given a scale parameter
    scale = torch.ones(6)

    # WHEN we create the quant context with that parameter and then request to
    # make the parameters contiguous
    quant_context = quant_context.with_changes(scale=scale)
    quant_context_contiguous = quant_context.contiguous_parameters()

    # THEN the same parameter should be returned (no deepcopy)
    assert quant_context_contiguous.quantization_params.scale is scale


def test_create_quantization_function() -> None:
    # GIVEN data and quant params
    data = torch.randn(3, 3)
    scale = 2.0
    rescale = 3.5

    # GIVEN an arbitrary quantize and dequantize function where the dequantize
    # function has a default value
    def _quantize(data: torch.Tensor, scale: float) -> torch.Tensor:
        return data * scale

    def _dequantize(data: torch.Tensor, rescale: float = rescale) -> torch.Tensor:
        return data * rescale

    # WHEN a custom quantizer is created
    CustomQuantizerParams, CustomQuantizerFunction, custom_quantizer = (
        ff.quantization.create_quantization_function("CustomQuantizer", _quantize, _dequantize)
    )

    # THEN the generated parameters dataclass must contain all parameters of
    # both functions.
    assert {f.name for f in dataclasses.fields(CustomQuantizerParams)} == {"scale", "rescale"}

    # WHEN the custom quantizer is used using the helper function
    quantized = custom_quantizer(data, scale=scale)

    # THEN the quantized representation and the dequantized representation must match expectations.
    torch.testing.assert_close(quantized.raw_data, _quantize(data, scale), atol=0, rtol=0)
    torch.testing.assert_close(
        quantized.dequantize(), _dequantize(_quantize(data, scale)), atol=0, rtol=0
    )
    assert quantized.quant_func is CustomQuantizerFunction  # type: ignore[comparison-overlap]

    # THEN the quantization arguments must be represented in the tensor's quant_args
    quant_args = quantized.quant_args()
    assert isinstance(quantized.quant_args(), CustomQuantizerParams)
    assert quant_args.scale == scale  # type: ignore[attr-defined]
    assert quant_args.rescale == rescale  # type: ignore[attr-defined]
