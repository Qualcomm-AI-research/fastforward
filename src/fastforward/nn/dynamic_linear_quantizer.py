# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.


import torch

from typing_extensions import override

import fastforward.quantization.granularity as granularities

from fastforward.nn.linear_quantizer import AbstractAffineQuantizer
from fastforward.quantization.affine import (
    AffineQuantizationFunction,
    DynamicAffineQuantParams,
    DynamicParamInferenceFn,
)
from fastforward.quantization.function import QuantizationFunction


class DynamicLinearQuantizer(AbstractAffineQuantizer[DynamicAffineQuantParams]):
    """Dynamic Linear quantizer.

    Support multiple quantization granularities. A granularity
    defines which parts of the input tensor are quantized using the same
    quantization parameter. E.g., per-tensor, per-channel, or per-block
    quantization. The granularity details are implemented in Granularity
    subclasses and are passed into LinearQuantizer.

    Args:
        num_bits: Bitwidth of quantization output granularity
        granularity: Granularity object that specifies the
            quantization granularity
        quantized_dtype: datatype in which the quantized data is stored
        parameter_inference_fn: function that determines scale and offset
            dynamically. If omitted, a min/max inference method is used.
    """

    def __init__(
        self,
        num_bits: int,
        *,
        granularity: granularities.Granularity | None = None,
        quantized_dtype: torch.dtype | None = None,
        parameter_inference_fn: DynamicParamInferenceFn | None = None,
    ):
        super().__init__(
            num_bits=num_bits, granularity=granularity, quantized_dtype=quantized_dtype
        )
        self.num_bits = num_bits
        self.granularity = granularity or granularities.PerTensor()
        self.quantized_dtype = quantized_dtype
        self.parameter_inference_fn = parameter_inference_fn

    @property
    def symmetric(self) -> bool:
        """True if symmetric quantization, False otherwise.

        Part of fastforward.range_setting.SupportsRangeBasedOperator Protocol
        """
        return False

    @override
    def quantization_parameters(self) -> DynamicAffineQuantParams:
        """Quantization parameters of this quantizer.

        Returns:
            `DynamicAffineQuantParams` specific to this quantizer
        """
        return DynamicAffineQuantParams(
            granularity=self.granularity,
            num_bits=self.num_bits,
            quantized_dtype=self.quantized_dtype,
            parameter_inference_fn=self.parameter_inference_fn,
        )

    @property
    @override
    def quantization_function(self) -> type[QuantizationFunction[DynamicAffineQuantParams]]:
        """Quantization function associated with this quantizer.

        Returns:
            `QuantizationFunction` that implements the quantization operator
            specific to this quantizer.
        """
        return AffineQuantizationFunction
