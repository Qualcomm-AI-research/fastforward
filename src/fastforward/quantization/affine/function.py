# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import dataclasses

from typing import TYPE_CHECKING, Any, Callable, TypeAlias, TypeVar

import torch

from typing_extensions import override

import fastforward as ff
import fastforward.quantization._linear_quantized_ops  # noqa: F401


# Import linear_quantized_ops to register linear_quantized_op impls
from fastforward.exceptions import ExportError
from fastforward.quantization import granularity as granularities
from fastforward.quantization.function import (
    QuantizationContext,
    QuantizationFunction,
    QuantizationParameters,
)

from ._autograd import dequantize_affine, quantize_affine, quantize_dynamic_affine

if TYPE_CHECKING:
    from fastforward.quantized_tensor import QuantizedTensor


@dataclasses.dataclass
class StaticAffineQuantParams(QuantizationParameters):
    """
    Quantization parameters for static affine quantization.
    """

    scale: float | torch.Tensor
    offset: float | torch.Tensor | None
    num_bits: int
    granularity: granularities.Granularity
    quantized_dtype: torch.dtype | None = None
    dequantize_dtype: torch.dtype | None = None


_T = TypeVar("_T")
_ScaleOffset = tuple[torch.Tensor, torch.Tensor | None]
DynamicParamInferenceFn: TypeAlias = Callable[
    ["DynamicAffineQuantParams", torch.Tensor], _ScaleOffset
]


@dataclasses.dataclass
class DynamicAffineQuantParams(QuantizationParameters):
    """
    Quantization parameters for dynamic affine quantization.
    """

    num_bits: int
    granularity: granularities.Granularity
    quantized_dtype: torch.dtype | None = None
    dequantize_dtype: torch.dtype | None = None
    parameter_inference_fn: DynamicParamInferenceFn | None = None


AffineQuantParams = TypeVar("AffineQuantParams", StaticAffineQuantParams, DynamicAffineQuantParams)


class AffineQuantizationFunction(QuantizationFunction[AffineQuantParams]):
    @classmethod
    @override
    def quantize(cls, data: torch.Tensor, params: AffineQuantParams) -> "QuantizedTensor":
        # The check is performed here with an if statement instead
        # of using the `match` function because dynamo does not
        # support this custom case.
        if ff.get_export_mode():
            # In the export case this function will return a standard torch.Tensor instead
            # of a QuantizedTensor. We ignore the type error due to how entangled the QuantizedTensor
            # return is with the rest of the codebase.
            return cls._export_quantize(data, params)  # type: ignore[return-value]

        match params:
            case StaticAffineQuantParams():
                return cls._static_quantize(data, params)
            case DynamicAffineQuantParams():
                return cls._dynamic_quantize(data, params)

        raise TypeError(f"Unsupported type for argument 'params': '{type(params)}'")

    @classmethod
    def _export_quantize(cls, data: torch.Tensor, params: AffineQuantParams) -> torch.Tensor:
        """
        Dedicated quantization function for export.

        Torch dynamo does not currently support custom tensor objects, such
        as the QuantizedTensor. For this reason this function performs
        quantization, followed immediately by dequantization.
        """
        if isinstance(params, DynamicAffineQuantParams):
            raise ExportError("Export does not support dynamic quantization.")

        tile_size = params.granularity.tile_size(data.shape)
        quantized_data = quantize_affine(
            data,
            params.scale,
            params.offset,
            tile_size,
            params.num_bits,
            params.quantized_dtype or data.dtype,
        )

        dequantized_data = dequantize_affine(
            quantized_data,
            params.scale,
            params.offset,
            tile_size,
            params.quantized_dtype or data.dtype,
        )
        return dequantized_data

    @classmethod
    def _static_quantize(
        cls, data: torch.Tensor, params: StaticAffineQuantParams
    ) -> "QuantizedTensor":
        tile_size = params.granularity.tile_size(data.shape)
        quantized_data = quantize_affine(
            data,
            params.scale,
            params.offset,
            tile_size,
            params.num_bits,
            params.quantized_dtype or data.dtype,
        )

        params = params.with_changes(dequantize_dtype=params.dequantize_dtype or data.dtype)
        context = QuantizationContext(cls, params)
        return ff.QuantizedTensor(quantized_data, context)

    @classmethod
    def _dynamic_quantize(
        cls, data: torch.Tensor, params: DynamicAffineQuantParams
    ) -> "QuantizedTensor":
        if params.parameter_inference_fn is None:
            return cls._dynamic_minmax_quantize(data, params)
        else:
            scale, offset = params.parameter_inference_fn(params, data)
            static_params = _static_from_dynamic(
                params,
                scale,
                offset,
                dequantize_dtype=params.dequantize_dtype or data.dtype,
            )
            return cls._static_quantize(data, static_params)

    @classmethod
    def _dynamic_minmax_quantize(
        cls, data: torch.Tensor, params: DynamicAffineQuantParams
    ) -> "QuantizedTensor":
        tile_size = params.granularity.tile_size(data.shape)
        tile_size = data.shape if tile_size == "data_shape" else tile_size
        output_dtype = params.quantized_dtype or data.dtype
        quantized_data, scale, offset = quantize_dynamic_affine(
            data, tile_size, params.num_bits, output_dtype
        )

        static_params = _static_from_dynamic(
            params, scale, offset, dequantize_dtype=params.dequantize_dtype or data.dtype
        )
        context = QuantizationContext(AffineQuantizationFunction, static_params)
        return ff.QuantizedTensor(quantized_data, context)

    @classmethod
    @override
    def dequantize(cls, data: torch.Tensor, params: AffineQuantParams) -> torch.Tensor:
        if isinstance(params, DynamicAffineQuantParams):
            raise TypeError("Cannot dequantize a QuantizedTensor with dynamic parameters.")

        tile_size = params.granularity.tile_size(data.shape)
        return dequantize_affine(
            data, params.scale, params.offset, tile_size, params.dequantize_dtype
        )


def _static_from_dynamic(
    params: DynamicAffineQuantParams,
    scale: torch.Tensor,
    offset: torch.Tensor | None,
    **changes: Any,
) -> StaticAffineQuantParams:
    args = dataclasses.asdict(params)

    # Remove dynamic only fields
    static_fields = StaticAffineQuantParams.__dataclass_fields__.keys()
    for key in list(args.keys()):
        if key not in static_fields:
            del args[key]

    # Set static only fields
    args["scale"] = scale
    args["offset"] = offset

    # Overwrite other changes
    args.update(**changes)

    return StaticAffineQuantParams(**args)
