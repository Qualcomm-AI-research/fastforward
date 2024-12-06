# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import Literal

import torch

from fastforward.nn.quantizer import Quantizer
from fastforward.quantization import granularity
from fastforward.quantization.dynamic import quantize_by_tile_function
from fastforward.quantization.function import BoundQuantizationFunction


class DynamicLinearQuantizer(Quantizer):
    """
    Dynamic Linear quantizer.

    Support multiple quantization granularities. A granularity
    defines which parts of the input tensor are quantized using the same
    quantization parameter. E.g., per-tensor, per-channel, or per-block
    quantization. The granularity details are implemented in Granularity
    subclasses and are passed into LinearQuantizer.

    Args:
        num_bits: Bitwidh of quantization output granularity
        granularity: Granularity object that specifies the
            quantization granularity
    """

    def __init__(
        self,
        num_bits: int,
        granularity: granularity.Granularity = granularity.PerTensor(),
    ) -> None:
        super().__init__()
        self.num_bits = num_bits
        self.granularity = granularity

    @property
    def symmetric(self) -> bool:
        """True if symmetric quantization, False otherwise.

        Part of fastforward.range_setting.SupportsRangeBasedOperator Protocol
        """
        return False

    @property
    def per_channel(self) -> bool:
        """Boolean indicating whether quantizer uses PerChannel quantization."""
        return granularity.is_per_channel(self.granularity)

    @property
    def per_tensor(self) -> bool:
        """Boolean indicating whether quantizer uses PerTensor quantization."""
        return granularity.is_per_tensor(self.granularity)

    def extra_repr(self) -> str:
        """
        Provide extra repr information on num_bits and granularities.
        """
        extra_repr = f"num_bits={self.num_bits}, granularity={self.granularity}"
        return super().extra_repr() + extra_repr

    def bound_quantization_function(
        self, tile_size: torch.Size | Literal["data_shape"]
    ) -> BoundQuantizationFunction:
        """
        Creates a `BoundQuantizationFunction` using parameters on this module.
        """
        return quantize_by_tile_function(tile_size=tile_size, num_bits=self.num_bits)

    def quantize(self, data: torch.Tensor) -> torch.Tensor:
        """Quantizer data using dynamic linear quantizer.

        Args:
            data: Tensor to quantize

        Returns:
            torch.Tensor:
                Quantized data
        """
        tile_size = self.granularity.tile_size(data.shape)
        quant_func = self.bound_quantization_function(tile_size)
        return quant_func(data)
