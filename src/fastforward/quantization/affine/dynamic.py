# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


from typing import TYPE_CHECKING

import torch

import fastforward as ff

from fastforward.quantization import granularity as granularities
from fastforward.quantization.affine.function import (
    AffineQuantizationFunction,
    DynamicAffineQuantParams,
)
from fastforward.quantization.function import QuantizationContext

if TYPE_CHECKING:
    from fastforward.quantized_tensor import QuantizedTensor


def quantization_context(
    granularity: granularities.Granularity,
    num_bits: int,
    quantized_dtype: torch.dtype | None = None,
    dequantize_dtype: torch.dtype | None = None,
) -> QuantizationContext[DynamicAffineQuantParams]:
    """Create quantization context for dynamic linear quantization.

    Args:
        granularity: The granuliarty to use for quantization
        num_bits: Bitwidth for quantization
        quantized_dtype: The data type in which the quantized data is stored
        dequantize_dtype: The datatype used by the dequantized tensor. If none is provided
            the datatype of the input tensor is used

    Returns:
        Quantization context for dynamic quantization.
    """
    params = DynamicAffineQuantParams(
        num_bits=num_bits,
        granularity=granularity,
        quantized_dtype=quantized_dtype,
        dequantize_dtype=dequantize_dtype,
    )
    return QuantizationContext(AffineQuantizationFunction, params)


def quantize_per_granularity(
    input: torch.Tensor,
    granularity: granularities.Granularity,
    num_bits: int = 8,
    output_dtype: torch.dtype | None = None,
) -> "QuantizedTensor":
    """Dynamically quantize `input` following `granularity`.

    Args:
        input: The data to be quantized
        granularity: The granularity to use for quantization
        num_bits: Bitwidth for quantization
        output_dtype: The data type in which the quantized data is stored

    Returns:
        Quantized tensor
    """
    match granularity:
        case granularities.PerTensor():
            return quantize_per_tensor(input, num_bits, output_dtype)
        case granularities.PerChannel(axis):
            return quantize_per_channel(input, axis, num_bits, output_dtype)
        case _:
            tile_size = granularity.tile_size(input.shape)
            assert not isinstance(tile_size, str)
            return quantize_by_tile(input, tile_size, num_bits, output_dtype)


def quantize_by_tile(
    input: torch.Tensor,
    tile_size: torch.Size,
    num_bits: int = 8,
    output_dtype: torch.dtype | None = None,
) -> "QuantizedTensor":
    """Dynamically quantize `input` by tile.

    Args:
        input: The data to be quantized
        tile_size: The size of 'sub-tensors' to which quantization is applied separately
        num_bits: Bitwidth for quantization
        output_dtype: The data type in which the quantized data is stored

    Returns:
        Quantized tensor
    """
    params = DynamicAffineQuantParams(
        num_bits=num_bits,
        granularity=ff.PerTile(tile_size),
        quantized_dtype=output_dtype,
    )
    return AffineQuantizationFunction.quantize(input, params)


def quantize_per_tensor(
    input: torch.Tensor,
    num_bits: int = 8,
    output_dtype: torch.dtype | None = None,
) -> "QuantizedTensor":
    """Dynamically quantize `input` per tensor.

    Args:
        input: The data to be quantized
        num_bits: Bitwidth for quantization
        output_dtype: The data type in which the quantized data is stored

    Returns:
        Quantized tensor
    """
    params = DynamicAffineQuantParams(
        num_bits=num_bits,
        granularity=ff.PerTensor(),
        quantized_dtype=output_dtype,
    )
    return AffineQuantizationFunction.quantize(input, params)


def quantize_per_channel(
    input: torch.Tensor,
    axis: int | tuple[int, ...],
    num_bits: int = 8,
    output_dtype: torch.dtype | None = None,
) -> "QuantizedTensor":
    """Dynamically quantize `input` per channel.

    Args:
        input: The data to be quantized
        axis: The channel (or channels) to share parameters over
        num_bits: Bitwidth for quantization
        output_dtype: The data type in which the quantized data is stored

    Returns:
        Quantized tensor
    """
    params = DynamicAffineQuantParams(
        num_bits=num_bits,
        granularity=ff.PerChannel(axis),
        quantized_dtype=output_dtype,
    )
    return AffineQuantizationFunction.quantize(input, params)


def quantize_per_block(
    input: torch.Tensor,
    channel_axis: int,
    block_axis: int,
    block_size: int,
    num_bits: int = 8,
    output_dtype: torch.dtype | None = None,
) -> "QuantizedTensor":
    """Dynamically quantize `input` per block.

    Args:
        input: The data to be quantized
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

    params = DynamicAffineQuantParams(
        num_bits=num_bits,
        granularity=ff.PerTile(tile_size),
        quantized_dtype=output_dtype,
    )
    return AffineQuantizationFunction.quantize(input, params)
