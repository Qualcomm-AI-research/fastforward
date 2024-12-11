# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Optional,
)

import torch

# Import linear_quantized_ops to register linear_quantized_op impls
import fastforward.nn._linear_quantized_ops  # noqa: F401

from fastforward.quantization.function import (
    BoundQuantizationFunction,
    QuantizationAutogradFunction,
)
from fastforward.type_common import SizeT

if TYPE_CHECKING:
    from fastforward.quantized_tensor import QuantizedTensor


class TiledDynamicAffineQuantizationFunction(QuantizationAutogradFunction):
    """
    QuantizationFunction that implements dynamic affine quantization.
    """

    @classmethod
    def bind(
        cls,
        tile_size: SizeT | Literal["data_shape"],
        num_bits: int,
        output_dtype: torch.dtype | None,
    ) -> BoundQuantizationFunction:
        """
        Construct a `BoundQuantizationFunction` for TiledDynamicAffineQuantizationFunction.
        """
        if num_bits < 1:
            raise ValueError(f"num_bits must be >= 1, got {num_bits}")

        return BoundQuantizationFunction(cls, tile_size, num_bits, output_dtype)

    @staticmethod
    def quantize(  # type: ignore[override]
        ctx: Any,
        data: torch.Tensor,
        tile_size: SizeT | Literal["data_shape"],
        num_bits: int,
        output_dtype: torch.dtype | None,
    ) -> torch.Tensor:
        """
        Quantize tensor unsing tiled linear operator.
        """
        ctx.save_for_backward(data)
        ctx.tile_size = data.shape if tile_size == "data_shape" else tile_size
        ctx.num_bits = num_bits
        output_dtype = output_dtype or torch.float32
        quantized_data, scale, offset = torch.ops.fastforward.quantize_dynamic_by_tile(
            data, ctx.tile_size, num_bits, output_dtype
        )
        ctx.save_for_backward(data, scale, offset)
        return quantized_data, scale, offset, tile_size, num_bits  # type: ignore[return-value]

    @staticmethod
    def dequantize(  # type: ignore[override]
        quant_data: torch.Tensor,
        scale: torch.Tensor,
        offset: torch.Tensor,
        tile_size: SizeT | Literal["data_shape"],
        num_bits: float,
    ) -> torch.Tensor:
        """
        Dequantize quantized data.

        Dequantize tensor using a tiled linear operator. I.e., take integer
        representation of quantized data and return real-valued (float)
        quantized data.

        Args:
            quant_data: Integer data to dequantize

        Returns:
            Tensor: Quantized real-valued representation of quant_data

        Note:
            Dequantization is used colloquially, i.e., it is not the inverse of
            the quantize operation, but takes a quantized integer representation
            and represent the quantized data as real-valued. The returned data
            still folows a quantization grid.
        """
        tile_size_to_use = quant_data.shape if tile_size == "data_shape" else tile_size
        return torch.ops.fastforward.dequantize_by_tile(quant_data, scale, tile_size_to_use, offset)  # type: ignore[no-any-return]

    @staticmethod
    def quant_dequant_backward(
        ctx: Any,
        grad: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor | None, ...]:
        """
        Combined backward pass for dequantize(quantize(x)).

        Since the quantization parameters are selected such to no clipping
        occurs, the entire backward pass is equivalent to a STE backward.
        """
        return (grad, None, None, None)


def quantize_by_tile_function(
    tile_size: torch.Size | Literal["data_shape"],
    num_bits: int = 8,
    output_dtype: Optional[torch.dtype] = None,
) -> "BoundQuantizationFunction":
    """
    Quantize data  using a tiled linear operator.

    The input variables are bound to a Function class, which can then be
    invoked multiple times.
    """
    return TiledDynamicAffineQuantizationFunction.bind(
        tile_size=tile_size,
        num_bits=num_bits,
        output_dtype=output_dtype,
    )


def quantize_by_tile(
    input: torch.Tensor,
    tile_size: torch.Size | Literal["data_shape"],
    num_bits: int = 8,
    output_dtype: Optional[torch.dtype] = None,
) -> "QuantizedTensor":
    """
    Quantize data  using a tiled linear operator.

    The input variables are passed to a Function class, which immediately
    returns the quantized tensor.
    """
    return TiledDynamicAffineQuantizationFunction.apply(
        data=input,
        tile_size=tile_size,
        num_bits=num_bits,
        output_dtype=output_dtype,
    )


def quantize_per_tensor(
    input: torch.Tensor,
    num_bits: int = 8,
    output_dtype: Optional[torch.dtype] = None,
) -> "QuantizedTensor":
    """
    Quantize data  using a tiled linear operator.

    The tile size is automatically assigned to the data shape, so the function
    applies the provided scale and offset on the entirety of the tensor.
    """
    return quantize_by_tile(input, "data_shape", num_bits, output_dtype)


def quantize_per_channel(
    input: torch.Tensor,
    axis: int,
    num_bits: int = 8,
    output_dtype: Optional[torch.dtype] = None,
) -> "QuantizedTensor":
    """
    Quantize data  using a tiled linear operator.

    The tile size is calculated based on the input axis, so the quantization
    parameters are project per channel.
    """
    input_shape = list(input.shape)
    input_shape[axis] = 1
    tile_size = torch.Size(input_shape)
    return quantize_by_tile(input, tile_size, num_bits, output_dtype)


def quantize_per_block(
    input: torch.Tensor,
    channel_axis: int,
    block_axis: int,
    block_size: int,
    num_bits: int = 8,
    output_dtype: Optional[torch.dtype] = None,
) -> "QuantizedTensor":
    """
    Quantize data  using a tiled linear operator.

    The tile size is calculated based on the input channel axis, block axis and
    block size, so the quantization parameters are project per user defined
    block.
    """
    input_shape = list(input.shape)
    input_shape[channel_axis] = 1
    input_shape[block_axis] = block_size
    tile_size = torch.Size(input_shape)
    return quantize_by_tile(input, tile_size, num_bits, output_dtype)
