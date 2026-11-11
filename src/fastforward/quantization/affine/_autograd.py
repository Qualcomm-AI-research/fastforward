# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""This file contains autograd functions for quantization.

We following the following autograd semantics: all gradients/gradient
approximations are implemented in the 'quantize' functions. As a result, the
backward pass of the dequantize functions is always an identity function. By
doing this the value of a quantized tensor and its dequantized tensor are only
different in representation, not in value. These semantics are not strictly
enforced but implementers of new quantization functions are encouraged to
follow the same semantics.

`static_affine_quantize_fn`, `affine_dequantize_fn` and
`dynamic_affine_quantize_fn` are typed interfaces to their corresponding
autograd functions.
"""

from typing import Any, Literal

import torch

from typing_extensions import override

import fastforward.quantization.affine._ops as ops

from fastforward.common import tensor_or_none


def affine_static_quantize_fn(
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
    encoded_tile_size = _encode_tile_size(tile_size, data.shape)
    return AffineStaticQuantizeFn.apply(  # type: ignore[no-any-return]
        data, scale, offset, encoded_tile_size, num_bits, quantized_dtype
    )


def affine_dynamic_quantize_fn(
    data: torch.Tensor,
    tile_size: torch.Size | Literal["data_shape"],
    num_bits: int,
    symmetric: bool,
    allow_one_sided: bool,
    quantized_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    encoded_tile_size = _encode_tile_size(tile_size, data.shape)
    return AffineDynamicQuantizeFn.apply(  # type: ignore[no-any-return]
        data, encoded_tile_size, num_bits, symmetric, allow_one_sided, quantized_dtype
    )


def affine_dequantize_fn(
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
    encoded_tile_size = _encode_tile_size(tile_size, data.shape)
    return AffineDequantizeFn.apply(  # type: ignore[no-any-return]
        data, scale, offset, encoded_tile_size, dtype
    )


def affine_static_qdq_fn(
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
    encoded_tile_size = _encode_tile_size(tile_size, data.shape)
    return AffineStaticQdqFn.apply(  # type: ignore[no-any-return]
        data, scale, offset, encoded_tile_size, num_bits, quantized_dtype
    )


class AffineStaticQuantizeFn(torch.autograd.Function):
    @staticmethod
    @override
    def forward(
        ctx: Any,
        data: torch.Tensor,
        scale: torch.Tensor,
        offset: torch.Tensor | None,
        tile_size: torch.Size | Literal["data_shape"],
        num_bits: int,
        quantized_dtype: torch.dtype | None,
    ) -> torch.Tensor:
        quant_dtype = quantized_dtype or data.dtype
        ctx.save_for_backward(data, scale, offset)
        tile_size = _decode_tile_size(tile_size, data.shape)
        ctx.tile_size = tile_size
        ctx.num_bits = num_bits

        return ops.affine_static_quantize_op(  # type: ignore[no-any-return]
            data, scale, tile_size, num_bits, quant_dtype, offset
        )

    @staticmethod
    @override
    @torch.autograd.function.once_differentiable
    def backward(
        ctx: Any, output_grad: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None, None, None]:
        data, scale, offset = ctx.saved_tensors
        tile_size = ctx.tile_size
        num_bits = ctx.num_bits
        grads = ops.affine_quantize_backward_op(
            data, output_grad, scale, tile_size, num_bits, offset
        )
        data_grad, scale_grad, offset_grad_ = grads
        offset_grad = offset_grad_ if offset is not None else None
        return (data_grad, scale_grad, offset_grad, None, None, None)


class AffineDynamicQuantizeFn(torch.autograd.Function):
    @staticmethod
    @override
    def forward(
        ctx: Any,
        data: torch.Tensor,
        tile_size: torch.Size | Literal["data_shape"],
        num_bits: int,
        symmetric: bool,
        allow_one_sided: bool,
        quantized_dtype: torch.dtype | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        quant_dtype = quantized_dtype or data.dtype
        tile_size = _decode_tile_size(tile_size, data.shape)
        return ops.affine_dynamic_quantize_op(  # type: ignore[no-any-return]
            data, tile_size, num_bits, symmetric, allow_one_sided, quant_dtype
        )

    @staticmethod
    @override
    @torch.autograd.function.once_differentiable
    def backward(
        ctx: Any, output_grad: torch.Tensor, scale_grad: torch.Tensor, offset_grad: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None, None]:
        del scale_grad, offset_grad
        grad = output_grad
        return (grad, None, None, None, None, None)


class AffineDequantizeFn(torch.autograd.Function):
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
        tile_size = _decode_tile_size(tile_size, data.shape)
        return ops.affine_dequantize_op(  # type: ignore[no-any-return]
            data, scale, tile_size, offset, dtype
        )

    @staticmethod
    @override
    @torch.autograd.function.once_differentiable
    def backward(
        ctx: Any, output_grad: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None]:
        return (output_grad, None, None, None, None)


class AffineStaticQdqFn(torch.autograd.Function):
    @staticmethod
    @override
    def forward(
        ctx: Any,
        data: torch.Tensor,
        scale: torch.Tensor,
        offset: torch.Tensor | None,
        tile_size: torch.Size | Literal["data_shape"],
        num_bits: int,
        quantized_dtype: torch.dtype | None,
    ) -> torch.Tensor:
        quant_dtype = quantized_dtype or data.dtype
        ctx.save_for_backward(data, scale, offset)
        tile_size = _decode_tile_size(tile_size, data.shape)
        ctx.tile_size = tile_size
        ctx.num_bits = num_bits
        return ops.affine_static_qdq_op(data, scale, tile_size, num_bits, quant_dtype, offset)  # type: ignore[no-any-return]

    @staticmethod
    @override
    @torch.autograd.function.once_differentiable
    def backward(
        ctx: Any, output_grad: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None, None, None]:
        data, scale, offset = ctx.saved_tensors
        tile_size = ctx.tile_size
        num_bits = ctx.num_bits
        grads = ops.affine_quantize_backward_op(
            data, output_grad, scale, tile_size, num_bits, offset
        )
        data_grad, scale_grad, offset_grad_ = grads
        offset_grad = offset_grad_ if offset is not None else None
        return (data_grad, scale_grad, offset_grad, None, None, None)


def _encode_tile_size(
    tile_size: torch.Size | Literal["data_shape"], data_shape: torch.Size
) -> torch.Size | Literal["data_shape"]:
    """Replace tile dimensions that span the whole data dimension by -1 (full-dim).

    When dynamo compiles with dynamic shapes, `tile_size` contains symbolic sizes
    (e.g., `(s0, s1, 1)`).

    Dynamo (torch<2.11) can not pass these as non-tensor arguments to
    `autograd.Function.apply`. Dimensions that cover the data dimension are the
    ones that are symbolic, so they are encoded as a constant and recovered from
    the data shape in `_decode_tile_size`.
    """
    if isinstance(tile_size, str):
        return tile_size
    return torch.Size(-1 if tile == dim else tile for tile, dim in zip(tile_size, data_shape))


def _decode_tile_size(
    tile_size: torch.Size | Literal["data_shape"], data_shape: torch.Size
) -> torch.Size:
    """Inverse of `_encode_tile_size`."""
    if isinstance(tile_size, str):
        return data_shape
    return torch.Size(dim if tile == -1 else tile for tile, dim in zip(tile_size, data_shape))
