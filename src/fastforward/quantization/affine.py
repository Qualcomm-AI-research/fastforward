# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import TYPE_CHECKING, Any, Literal, Optional, cast

import torch

# Import linear_quantized_ops to register linear_quantized_op impls
import fastforward.nn._linear_quantized_ops  # noqa: F401

from fastforward.common import ensure_tensor
from fastforward.quantization.function import (
    BoundQuantizationFunction,
    QuantizationAutogradFunction,
)
from fastforward.type_common import SizeT

if TYPE_CHECKING:
    from fastforward.quantized_tensor import QuantizedTensor


def integer_minimum(num_bits: float) -> float:
    """
    Return the minimum integer value given num_bits.

    Args:
        num_bits: Number of bits
    Returns:
        float: Minimum integer value supporting the quantization range
    """
    return -(2 ** (num_bits - 1))


def integer_maximum(num_bits: float) -> float:
    """
    Return the maximum integer value given num_bits.

    Args:
        num_bits: Number of bits
    Returns:
        float: Maximum integer value supporting the quantization range
    """
    return -integer_minimum(num_bits) - 1


def quantization_range(
    scale: torch.Tensor | float, offset: torch.Tensor | float | None, num_bits: float
) -> tuple[torch.Tensor | float, torch.Tensor | float]:
    """
    Compute quantization range for a set of quantization parameters.

    If both scale and offset are tensors, their dimensions must match.

    Args:
        scale: Scale for quantization
        offset: Offset for quantization
        num_bits: Number of bits used for quantization

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            `Tuple` of tensor representing the minimum and maximum thresholds of the
            quantization range, respectively.
    """
    offset = 0.0 if offset is None else offset
    range_min = (integer_minimum(num_bits) + offset) * scale
    range_max = (integer_maximum(num_bits) + offset) * scale
    return range_min, range_max


def parameters_for_range(
    min_range: torch.Tensor,
    max_range: torch.Tensor,
    num_bits: float,
    symmetric: bool,
    allow_one_sided: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Compute affine quantization parameters for a range.

    Given a range or ranges (if min_range and max_range are multidimensional),
    compute the scale and offset parameters that best represent that the given
    range(s)

    Args:
        min_range: `Tensor` representing the minimum range threshold
        max_range: `Tensor` representing the maximum range threshold

    Returns:
        tuple[torch.Tensor, torch.Tensor | None]:
            scale and offset values that best represent the given range. Offset
            may be None in the non-onesided symmetric case.

    Notes::
        - The dimensionality of min_range and max_range must match.
        - If symmetric == True, not every range can be represented exactly,
            in that case, the scale and offset parameters are selected such
            that the entire given range is within bounds, i.e., the range
            used by `LinearQuantizer` may be wider and no assumptions on using
            the given range exactly must be made.
    """
    min_range, max_range = ensure_tensor(min_range), ensure_tensor(max_range)

    # Here we check if the minimum range threshold is all zero. In that case,
    # we assume a one-sided (or unsigned following Nagel et al., 2021). LinearQuantizerOp
    # still uses a signed integer representation, hence, we treat the unsigned symmetric case
    # from Nagel et al. as the asymmetric case where the offset is equal to minimum integer
    # value.
    #
    # NB: Theoretically, it is possible for non positive data to be assigned a zero minimum
    # threshold (e.g., when using an l2 quantization error minimization).
    one_sided = min_range.min() >= 0 and allow_one_sided

    int_min = integer_minimum(num_bits)

    if symmetric and one_sided:
        min_range = torch.zeros_like(min_range)

    if symmetric and not one_sided:
        # Choose scale such that entire range falls within bounds
        neg_scale = torch.abs(min_range) / abs(int_min)
        pos_scale = torch.abs(max_range) / abs(integer_maximum(num_bits))
        return torch.max(neg_scale, pos_scale), None
    else:
        # Choose scale and offset such that integer_minimum and integer_maximum
        # map to min and max, respectively.
        # NB: this results in a quantization grid in which min and max lie exactly
        # half a bin from the maximum/minimum values.
        num_steps = 2**num_bits - 1
        interval_length = max_range - min_range
        scale = interval_length / num_steps
        scale = scale.clamp(torch.finfo(scale.dtype).eps)
        offset = min_range / scale - int_min
        return scale, offset


class TiledAffineQuantizationFunction(QuantizationAutogradFunction):
    """
    QuantizationFunction for affine quantization.
    """

    @classmethod
    def bind(
        cls,
        scale: torch.Tensor,
        tile_size: SizeT | Literal["data_shape"],
        num_bits: int,
        output_dtype: torch.dtype | None,
        offset: Optional[torch.Tensor] = None,
    ) -> BoundQuantizationFunction:
        """
        Create a bound quantization function for affine quantization given the provided arguments.

        Args:
            scale: Quantization scale
            tile_size: Tile size that determines quantizations grouping
            num_bits: Number of bits for quantization
            output_dtype: Dtype used for quantized representation
            offset: Quantization offset

        Returns:
            `BoundQuantizationFunction` that captures the quantization using the
            provided arguments.
        """
        if num_bits < 1:
            raise ValueError(f"num_bits must be >= 1, got {num_bits}")

        return super().bind(
            scale=scale,
            offset=offset,
            tile_size=tile_size,
            num_bits=num_bits,
            output_dtype=output_dtype,
        )

    @staticmethod
    def quantize(  # type: ignore[override]
        ctx: Any,
        data: torch.Tensor,
        scale: torch.Tensor,
        tile_size: SizeT | Literal["data_shape"],
        num_bits: int,
        output_dtype: torch.dtype | None,
        offset: Optional[torch.Tensor] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        SizeT | Literal["data_shape"],
        int,
        torch.dtype,
        torch.Tensor | None,
    ]:
        """
        Quantize tensor unsing tiled linear operator.
        """
        ctx.save_for_backward(data, scale, offset)
        ctx.tile_size = data.shape if tile_size == "data_shape" else tile_size
        ctx.num_bits = num_bits
        output_dtype = output_dtype or data.dtype
        quantized_data = torch.ops.fastforward.quantize_by_tile(
            data, scale, ctx.tile_size, num_bits, output_dtype, offset
        )
        quantized_data = cast(torch.Tensor, quantized_data)
        return quantized_data, scale, tile_size, num_bits, data.dtype, offset

    @staticmethod
    def dequantize(  # type: ignore[override]
        quant_data: torch.Tensor,
        scale: torch.Tensor,
        tile_size: SizeT | Literal["data_shape"],
        num_bits: int,
        output_dtype: torch.dtype | None,
        offset: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Dequantize tensor using a tiled linear operator.

        Take integer representation of quantized data and return
        real-valued (float) quantized data.

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
        return torch.ops.fastforward.dequantize_by_tile(  # type: ignore[no-any-return]
            quant_data, scale, tile_size_to_use, offset, output_dtype
        )

    @staticmethod
    def quant_dequant_backward(
        ctx: Any, grad: torch.Tensor, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor | None, ...]:
        """
        Combined quantize and dequantize backward implementation.

        Quantize tensor using a tiled linear operator, for performing the backward
        pass. The function returns the gradient values and graph settings.
        """
        data, scale, offset = ctx.saved_tensors
        tile_size = ctx.tile_size
        num_bits = ctx.num_bits
        grads = torch.ops.fastforward.quantize_by_tile_backward(
            data, grad, scale, tile_size, num_bits, offset
        )
        data_grad, scale_grad, offset_grad_ = grads
        offset_grad = offset_grad_ if offset is not None else None
        return (data_grad, scale_grad, None, None, None, offset_grad)


def quantize_by_tile_function(
    scale: torch.Tensor | float,
    offset: torch.Tensor | float | None,
    tile_size: torch.Size | Literal["data_shape"],
    num_bits: int = 8,
    output_dtype: Optional[torch.dtype] = None,
) -> "BoundQuantizationFunction":
    """
    Quantize data  using a tiled linear operator.

    The input variables are bound to a Function class, which can then be
    invoked multiple times.
    """
    scale = ensure_tensor(scale)
    if offset is not None:
        offset = ensure_tensor(offset)
    return TiledAffineQuantizationFunction.bind(
        scale=scale,
        offset=offset,
        tile_size=tile_size,
        num_bits=num_bits,
        output_dtype=output_dtype,
    )


def quantize_by_tile(
    input: torch.Tensor,
    scale: torch.Tensor | float,
    offset: torch.Tensor | float | None,
    tile_size: torch.Size | Literal["data_shape"],
    num_bits: int = 8,
    output_dtype: Optional[torch.dtype] = None,
) -> "QuantizedTensor":
    """
    Quantize data  using a tiled linear operator.

    The input variables are passed to a Function class, which immediately
    returns the quantized tensor.
    """
    scale = ensure_tensor(scale)
    if offset is not None:
        offset = ensure_tensor(offset)
    return TiledAffineQuantizationFunction.apply(
        data=input,
        scale=scale,
        offset=offset,
        tile_size=tile_size,
        num_bits=num_bits,
        output_dtype=output_dtype,
    )


def quantize_per_tensor(
    input: torch.Tensor,
    scale: torch.Tensor | float,
    offset: torch.Tensor | float | None = None,
    num_bits: int = 8,
    output_dtype: Optional[torch.dtype] = None,
) -> "QuantizedTensor":
    """
    Quantize data  using a tiled linear operator.

    The tile size is automatically assigned to the data shape, so the function
    applies the provided scale and offset on the entirety of the tensor.
    """
    scale = ensure_tensor(scale)
    if offset is not None:
        offset = ensure_tensor(offset)

    tile_size = input.size()
    return quantize_by_tile(input, scale, offset, tile_size, num_bits, output_dtype)


def quantize_per_channel(
    input: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
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
    return quantize_by_tile(input, scale, offset, tile_size, num_bits, output_dtype)


def quantize_per_block(
    input: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
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
    return quantize_by_tile(input, scale, offset, tile_size, num_bits, output_dtype)
