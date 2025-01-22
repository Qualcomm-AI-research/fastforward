# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

"""
This file contains autograd functions for quantization.

We following the following autograd semantics: all gradients/gradient
approximations are implemented in the 'quantize' functions. As a result, the
backward pass of the dequantize functions is always an identity function. By
doing this the value of a quantized tensor and its dequantized tensor are only
different in representation, not in value. These semantics are not strictly
enforced but implementers of new quantization functions are encouraged to
follow the same semantics.

`quantize_affine`, `dequantize_affine` and `quantize_dynamic_affine` are typed
interfaces to their corresponding autograd functions.
"""

from typing import Any, Literal

import torch

from typing_extensions import override

from fastforward.common import tensor_or_none


def quantize_affine(
    data: torch.Tensor,
    scale: float | torch.Tensor,
    offset: float | torch.Tensor | None,
    tile_size: torch.Size | Literal["data_shape"],
    num_bits: int,
    quantized_dtype: torch.dtype | None,
) -> torch.Tensor:
    dtype = data.dtype if data.dtype.is_floating_point else torch.get_default_dtype()
    scale = tensor_or_none(scale, dtype=dtype, device=data.device)
    offset = tensor_or_none(offset, dtype=dtype, device=data.device)
    return QuantizeStaticAffine.apply(data, scale, offset, tile_size, num_bits, quantized_dtype)  # type: ignore[no-any-return]


def dequantize_affine(
    data: torch.Tensor,
    scale: float | torch.Tensor,
    offset: float | torch.Tensor | None,
    tile_size: torch.Size | Literal["data_shape"],
    dtype: torch.dtype | None,
) -> torch.Tensor:
    if dtype is None:
        dtype = data.dtype if data.dtype.is_floating_point else torch.get_default_dtype()
    scale = tensor_or_none(scale, dtype=dtype, device=data.device)
    offset = tensor_or_none(offset, dtype=dtype, device=data.device)
    return DequantizeAffine.apply(data, scale, offset, tile_size, dtype)  # type: ignore[no-any-return]


def quantize_dynamic_affine(
    data: torch.Tensor,
    tile_size: torch.Size | Literal["data_shape"],
    num_bits: int,
    quantized_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    return QuantizeDynamicAffine.apply(data, tile_size, num_bits, quantized_dtype)  # type: ignore[no-any-return]


class QuantizeStaticAffine(torch.autograd.Function):
    @staticmethod
    @override
    def forward(  # type: ignore[override]
        ctx: Any,
        data: torch.Tensor,
        scale: torch.Tensor,
        offset: torch.Tensor | None,
        tile_size: torch.Size | Literal["data_shape"],
        num_bits: int,
        quantized_dtype: torch.dtype | None,
    ) -> torch.Tensor:
        tile_size = data.shape if tile_size == "data_shape" else tile_size
        quant_dtype = quantized_dtype or data.dtype

        ctx.save_for_backward(data, scale, offset)
        tile_size = data.shape if tile_size == "data_shape" else tile_size
        ctx.tile_size = tile_size
        ctx.num_bits = num_bits

        return torch.ops.fastforward.quantize_by_tile(  # type: ignore[no-any-return]
            data, scale, tile_size, num_bits, quant_dtype, offset)

    @staticmethod
    @override
    @torch.autograd.function.once_differentiable  # type: ignore[misc]
    def backward(
        ctx: Any, output_grad: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None, None, None]:
        data, scale, offset = ctx.saved_tensors
        tile_size = ctx.tile_size
        num_bits = ctx.num_bits
        grads = torch.ops.fastforward.quantize_by_tile_backward(
            data, output_grad, scale, tile_size, num_bits, offset
        )
        data_grad, scale_grad, offset_grad_ = grads
        offset_grad = offset_grad_ if offset is not None else None
        return (data_grad, scale_grad, offset_grad, None, None, None)


class QuantizeDynamicAffine(torch.autograd.Function):
    @staticmethod
    @override
    def forward(
        ctx: Any,
        data: torch.Tensor,
        tile_size: torch.Size | Literal["data_shape"],
        num_bits: int,
        quantized_dtype: torch.dtype | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        quant_dtype = quantized_dtype or data.dtype
        tile_size = data.shape if tile_size == "data_shape" else tile_size
        return torch.ops.fastforward.quantize_dynamic_by_tile(  # type: ignore[no-any-return]
            data, tile_size, num_bits, quant_dtype
        )

    @staticmethod
    @override
    @torch.autograd.function.once_differentiable  # type: ignore[misc]
    def backward(
        ctx: Any, output_grad: torch.Tensor, scale_grad: torch.Tensor, offset_grad: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        del scale_grad, offset_grad
        grad = output_grad
        return (grad, None, None, None)


class DequantizeAffine(torch.autograd.Function):
    @staticmethod
    @override
    def forward(
        ctx: Any,
        data: torch.Tensor,
        scale: torch.Tensor,
        offset: torch.Tensor | None,
        tile_size: torch.Size | Literal["data_shape"],
        dtype: torch.dtype | None,
    ) -> torch.Tensor:
        tile_size = data.shape if tile_size == "data_shape" else tile_size
        return torch.ops.fastforward.dequantize_by_tile(data, scale, tile_size, offset, dtype)  # type: ignore[no-any-return]

    @staticmethod
    @override
    @torch.autograd.function.once_differentiable  # type: ignore[misc]
    def backward(
        ctx: Any, output_grad: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None]:
        return (output_grad, None, None, None, None)
