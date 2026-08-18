# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import List, Sequence

import torch

import fastforward as ff

from fastforward.library import conditional_compile, custom_quant_op, register_quant_fake
from fastforward.quantization import tiled_tensor
from fastforward.quantization.ste import round_ste

from . import dtypes, range

SizeT = Sequence[int]


@custom_quant_op("affine_static_quantize")
@conditional_compile
def affine_static_quantize_op(
    data: torch.Tensor,
    scale: torch.Tensor,
    tile_size: SizeT,
    num_bits: float,
    output_dtype: torch.dtype | None,
    offset: torch.Tensor | None = None,
) -> torch.Tensor:
    scale = scale.reshape(-1)
    offset = _infer_offset(offset, scale)
    tile_size = torch.Size(tile_size)

    min_threshold = -(2 ** (num_bits - 1))
    max_threshold = -min_threshold - 1
    row_representation = tiled_tensor.tiles_to_rows(data, tile_size)
    quantized = round_ste(row_representation / scale[:, None] - offset[:, None])
    quantized = torch.clamp(quantized, min_threshold, max_threshold)
    result = tiled_tensor.rows_to_tiles(quantized, data.shape, tile_size)
    output_dtype = output_dtype or result.dtype
    if not dtypes.can_support_bitwidth(output_dtype, num_bits):
        msg = f"Provided dtype ({output_dtype}) is not enough to store {num_bits} bits quantized values."
        raise RuntimeError(msg)
    result = result.to(output_dtype)
    return result


@custom_quant_op("affine_dequantize")
@conditional_compile
def affine_dequantize_op(
    data: torch.Tensor,
    scale: torch.Tensor,
    tile_size: SizeT,
    offset: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    scale = scale.reshape(-1)
    offset = _infer_offset(offset, scale)
    tile_size = torch.Size(tile_size)

    row_representation = tiled_tensor.tiles_to_rows(data, tile_size)
    dequantized = (row_representation + offset[:, None]) * scale[:, None]
    dequantized = tiled_tensor.rows_to_tiles(dequantized, data.shape, tile_size)
    if output_dtype:
        dequantized = dequantized.to(output_dtype)
    return dequantized


@custom_quant_op("affine_quantize_backward")
@conditional_compile
def affine_quantize_backward_op(
    data: torch.Tensor,
    output_grad: torch.Tensor,
    scale: torch.Tensor,
    tile_size: SizeT,
    num_bits: float,
    offset: torch.Tensor | None = None,
) -> List[torch.Tensor]:  # noqa: UP006
    param_shape = scale.shape
    scale = scale.reshape(-1)
    offset_is_none = offset is None
    offset = _infer_offset(offset, scale)
    tile_size = torch.Size(tile_size)

    min_threshold = -(2 ** (num_bits - 1))
    max_threshold = -min_threshold - 1

    data_as_rows = tiled_tensor.tiles_to_rows(data, tile_size)
    grad_as_rows = tiled_tensor.tiles_to_rows(output_grad, tile_size)

    pre_round = (data_as_rows / scale[:, None]) - round_ste(offset[:, None])
    quantized = torch.round(pre_round)
    clip_mask = torch.logical_or(quantized < min_threshold, quantized > max_threshold)

    dinput = torch.where(clip_mask, 0, grad_as_rows)

    if offset_is_none:
        doffset = torch.Tensor()
    else:
        doffset = torch.where(clip_mask, scale[:, None] * grad_as_rows, 0)
        doffset = doffset.sum(1).reshape(param_shape)

    dscale = torch.empty(quantized.shape, dtype=scale.dtype, device=scale.device)
    min_thresh = scale.new_tensor([min_threshold])
    max_thresh = scale.new_tensor([max_threshold])
    torch.where(quantized < min_threshold, min_thresh, max_thresh, out=dscale)
    dscale.add_(offset[:, None].to(dscale.dtype))
    torch.where(clip_mask, dscale, (quantized - pre_round).to(dscale.dtype), out=dscale)
    dscale.mul_(grad_as_rows)

    dinput = tiled_tensor.rows_to_tiles(dinput, data.shape, tile_size)
    dscale = dscale.sum(1).reshape(param_shape)
    return [dinput, dscale, doffset]


@custom_quant_op("affine_dynamic_quantize")
@conditional_compile
def affine_dynamic_quantize_op(
    data: torch.Tensor,
    tile_size: SizeT,
    num_bits: float,
    symmetric: bool,
    allow_one_sided: bool,
    output_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tile_size = torch.Size(tile_size)

    min_threshold = -(2 ** (num_bits - 1))
    max_threshold = -min_threshold - 1
    row_representation = tiled_tensor.tiles_to_rows(data, tile_size)

    try:
        min_range = torch.min(row_representation, dim=1).values
        max_range = torch.max(row_representation, dim=1).values
    except IndexError as e:
        msg = f"Cannot dynamically quantize an empty tensor of shape {data.shape}"
        raise ff.exceptions.QuantizationError(msg) from e

    scale, offset = range.parameters_for_range(
        min_range,
        max_range,
        num_bits,
        symmetric=symmetric,
        allow_one_sided=allow_one_sided,
    )
    if offset is None:
        offset = torch.zeros_like(scale)
    offset = torch.round(offset)

    quantized = torch.round(row_representation / scale[:, None] - offset[:, None])
    quantized = torch.clamp(quantized, min_threshold, max_threshold)
    result = tiled_tensor.rows_to_tiles(quantized, data.shape, tile_size)
    output_dtype = output_dtype or result.dtype
    if not dtypes.can_support_bitwidth(output_dtype, num_bits):
        msg = f"Provided dtype ({output_dtype}) is not enough to store {num_bits} bits quantized values."
        raise RuntimeError(msg)
    result = result.to(output_dtype)
    return result, scale, offset


@register_quant_fake("affine_static_quantize")
def affine_static_quantize_meta(
    input: torch.Tensor,
    scale: torch.Tensor,
    tile_size: SizeT,
    num_bits: float,
    output_dtype: torch.dtype | None,
    offset: torch.Tensor | None = None,
) -> torch.Tensor:
    del scale, tile_size, num_bits, output_dtype, offset
    return torch.empty_like(input)


@register_quant_fake("affine_dequantize")
def affine_dequantize_meta(
    input: torch.Tensor,
    scale: torch.Tensor,
    tile_size: SizeT,
    offset: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    del scale, tile_size, offset, output_dtype
    return torch.empty_like(input)


@register_quant_fake("affine_quantize_backward")
def affine_quantize_backward_meta(
    input: torch.Tensor,
    output_grad: torch.Tensor,
    scale: torch.Tensor,
    tile_size: SizeT,
    num_bits: float,
    offset: torch.Tensor | None = None,
) -> List[torch.Tensor]:  # noqa: UP006
    del output_grad, tile_size, num_bits, offset
    return [torch.empty_like(input), torch.empty_like(scale), torch.empty_like(scale)]


@register_quant_fake("affine_dynamic_quantize")
def affine_dynamic_quantize_meta(
    input: torch.Tensor,
    tile_size: SizeT,
    num_bits: float,
    symmetric: bool,
    allow_one_sided: bool,
    output_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del num_bits, symmetric, allow_one_sided, output_dtype
    num_params = int(input.numel() / torch.Size(tile_size).numel())
    scale = torch.empty(num_params)
    offset = torch.empty(num_params)
    return torch.empty_like(input), scale, offset


def _infer_offset(offset: torch.Tensor | None, scale: torch.Tensor) -> torch.Tensor:
    return torch.round(offset.reshape(-1)) if offset is not None else torch.zeros_like(scale)
