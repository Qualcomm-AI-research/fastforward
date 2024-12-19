# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import TYPE_CHECKING

import torch

import fastforward as ff

from fastforward.quantization import granularity as granularities
from fastforward.quantization.function import QuantizationContext

from .function import AffineQuantizationFunction, StaticAffineQuantParams

if TYPE_CHECKING:
    from fastforward import QuantizedTensor


def quantization_context(
    scale: torch.Tensor | float,
    offset: torch.Tensor | float | None,
    granularity: granularities.Granularity | None = None,
    num_bits: int = 8,
    output_dtype: torch.dtype | None = None,
    dequantize_dtype: torch.dtype | None = None,
) -> QuantizationContext[StaticAffineQuantParams]:
    """
    Create quantization context for static linear quantization

    Args:
        scale: Scale parameters to use for quantization. Its dimensionality must match the number
            of tiles as inferred based on the input data and `granularity`
        offset: Offset parameters to use for quantization. Its dimensionality must match the number
            of tiles as inferred based on the input data and `granularity`
        granularity: The granularity to use for quantization
        num_bits: Bitwidth for quantization
        output_dtype: The data type in which the quantized data is stored
        dequantize_dtype: The datatype used by the dequantized tensor. If none is provided
            the datatype of the input tensor is used.

    Returns:
        Quantization context for static quantization.
    """
    granularity = granularity or granularities.PerTensor()

    params = StaticAffineQuantParams(
        scale=scale,
        offset=offset,
        num_bits=num_bits,
        granularity=granularity,
        quantized_dtype=output_dtype,
        dequantize_dtype=dequantize_dtype,
    )

    return QuantizationContext(AffineQuantizationFunction, params)


def quantize_per_granularity(
    input: torch.Tensor,
    scale: torch.Tensor | float,
    offset: torch.Tensor | float | None,
    granularity: granularities.Granularity,
    num_bits: int = 8,
    output_dtype: torch.dtype | None = None,
) -> "QuantizedTensor":
    """
    Quantize `input` following `granularity` and the given
    quantization parameters.

    Args:
        input: The data to be quantized
        scale: Scale parameters to use for quantization. Its dimensionality must match the number
            of tiles as inferred based on `input` data and `granularity`
        offset: Offset parameters to use for quantization. Its dimensionality must match the number
            of tiles as inferred based on the `input` data and `granularity`
        granularity: The granularity to use for quantization
        num_bits: Bitwidth for quantization
        output_dtype: The data type in which the quantized data is stored

    Returns:
        Quantized tensor
    """
    match granularity:
        case granularities.PerTensor():
            return quantize_per_tensor(input, scale, offset, num_bits, output_dtype)
        case granularities.PerChannel(axis):
            return quantize_per_channel(input, scale, offset, axis, num_bits, output_dtype)
        case _:
            tile_size = granularity.tile_size(input.shape)
            assert not isinstance(tile_size, str)
            return quantize_by_tile(input, scale, offset, tile_size, num_bits, output_dtype)


def quantize_by_tile(
    input: torch.Tensor,
    scale: torch.Tensor | float,
    offset: torch.Tensor | float | None,
    tile_size: torch.Size,
    num_bits: int = 8,
    output_dtype: torch.dtype | None = None,
) -> "QuantizedTensor":
    """
    Quantize `input` by tile and the given quantization parameters.

    Args:
        input: The data to be quantized
        scale: Scale parameters to use for quantization. Its dimensionality must match the number
            of tiles as inferred based on `input` data and `granularity`
        offset: Offset parameters to use for quantization. Its dimensionality must match the number
            of tiles as inferred based on the `input` data and `granularity`
        tile_size: The tile_size to 'break' up the input tensor quantization
        num_bits: Bitwidth for quantization
        output_dtype: The data type in which the quantized data is stored

    Returns:
        Quantized tensor
    """
    params = StaticAffineQuantParams(
        scale=scale,
        offset=offset,
        num_bits=num_bits,
        granularity=ff.PerTile(tile_shape=tile_size),
        quantized_dtype=output_dtype,
    )
    return AffineQuantizationFunction.quantize(input, params)


def quantize_per_tensor(
    input: torch.Tensor,
    scale: torch.Tensor | float,
    offset: torch.Tensor | float | None = None,
    num_bits: int = 8,
    output_dtype: torch.dtype | None = None,
) -> "QuantizedTensor":
    """
    Quantize `input` per tensor.

    Args:
        input: The data to be quantized
        scale: Scale parameters to use for quantization.
        offset: Offset parameters to use for quantization.
        num_bits: Bitwidth for quantization
        output_dtype: The data type in which the quantized data is stored

    Returns:
        Quantized tensor
    """
    params = StaticAffineQuantParams(
        scale=scale,
        offset=offset,
        num_bits=num_bits,
        granularity=ff.PerTensor(),
        quantized_dtype=output_dtype,
    )
    return AffineQuantizationFunction.quantize(input, params)


def quantize_per_channel(
    input: torch.Tensor,
    scale: torch.Tensor | float,
    offset: torch.Tensor | float | None = None,
    axis: int | tuple[int, ...] = -1,
    num_bits: int = 8,
    output_dtype: torch.dtype | None = None,
) -> "QuantizedTensor":
    """
    Quantize `input` per channel.

    Args:
        input: The data to be quantized
        scale: Scale parameters to use for quantization.
        offset: Offset parameters to use for quantization.
        axis: The channel (or channels) to share parameters over
        num_bits: Bitwidth for quantization
        output_dtype: The data type in which the quantized data is stored

    Returns:
        Quantized tensor
    """
    params = StaticAffineQuantParams(
        scale=scale,
        offset=offset,
        num_bits=num_bits,
        granularity=ff.PerChannel(axis),
        quantized_dtype=output_dtype,
    )
    return AffineQuantizationFunction.quantize(input, params)


def quantize_per_block(
    input: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
    channel_axis: int,
    block_axis: int,
    block_size: int,
    num_bits: int = 8,
    output_dtype: torch.dtype | None = None,
) -> "QuantizedTensor":
    """
    Quantize `input` per block,

    Args:
        input: The data to be quantized
        scale: Scale parameters to use for quantization.
        offset: Offset parameters to use for quantization.
        channel_axis: The axis along which blocks are repeated
        block_axis: The axis along which blocks are created
        block_size: The number of blocks in `block_axis`
        num_bits: Bitwidth for quantization
        output_dtype: The data type in which the quantized data is stored

    Returns:
        Quantized tensor
    """
    input_shape = list(input.shape)
    input_shape[channel_axis] = 1
    input_shape[block_axis] = block_size
    tile_size = torch.Size(input_shape)
    return quantize_by_tile(input, scale, offset, tile_size, num_bits, output_dtype)
