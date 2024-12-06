# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import Callable, Literal

import torch

import fastforward.quantization.affine

from fastforward.common import ensure_tensor
from fastforward.nn.quantizer import Quantizer
from fastforward.quantization import granularity
from fastforward.quantization.function import BoundQuantizationFunction
from fastforward.quantized_tensor import QuantizedTensor


class LinearQuantizer(Quantizer):
    """
    Linear quantizer.

    Support multiple quantization granularities. A granularity
    defines which part of the input tensor are quantized using the same
    quantization parameter. E.g., per-tensor, per-channel, or per-block
    quantization. The granularity details are implemented in Granularity
    subclasses and are passed into LinearQuantizer.

    Args:
        num_bits: Bitwidh of quantization output granularity
        symmetric: Use symmetric quantization if True, asymmetric otherwise.
            (Default: True)
        allow_one_sided: If symmetric and allow_one_sided, the quantizer may fall
            back to a one-sided (or unsigned) quantizer if all lower quantizations thresholds
            are above zero. (Default: True)
        granularity: Granularity object that specifies the quantization
            granularity
    """

    offset: torch.Tensor | torch.nn.Parameter | None

    def __init__(
        self,
        num_bits: int,
        symmetric: bool = True,
        allow_one_sided: bool = True,
        granularity: granularity.Granularity = granularity.PerTensor(),
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.num_bits = num_bits
        self.granularity = granularity
        self.scale = torch.nn.UninitializedParameter(device=device)  # type: ignore[call-arg]
        self.allow_one_sided = allow_one_sided
        if symmetric and not allow_one_sided:
            self.register_parameter("offset", None)
        elif symmetric and allow_one_sided:
            self.register_buffer("offset", torch.nn.UninitializedBuffer(device=device))  # type: ignore[call-arg]
        else:
            self.offset = torch.nn.UninitializedParameter(device=device)  # type: ignore[call-arg]

    @property
    def symmetric(self) -> bool:
        """
        True if symmetric quantization, False otherwise.

        Part of fastforward.range_setting.SupportsRangeBasedOperator Protocol
        """
        # The quantizer is 'symmetric' (this includes the unsigned case,
        # following [1]) If no offset is set, or  if the offset is set to a
        # buffer that equals 2 ** (num_bits -1)
        #
        # Here we only use a buffer in the 'unsigned' case, hence we do not
        # check for the buffer's values.
        #
        # [1] Nagel, Markus, et al. "A white paper on neural network quantization."
        #     arXiv preprint arXiv:2106.08295 (2021).
        return "offset" in self._buffers or self.offset is None

    @property
    def per_channel(self) -> bool:
        """
        Boolean indicating whether quantizer uses PerChannel quantization.
        """
        return granularity.is_per_channel(self.granularity)

    @property
    def per_tensor(self) -> bool:
        """
        Boolean indicating whether quantizer uses PerTensor quantization.
        """
        return granularity.is_per_tensor(self.granularity)

    @property
    def integer_minimum(self) -> float:
        """
        The minimum integer value that quantized data takes, based on bitwidth.
        """
        return fastforward.quantization.affine.integer_minimum(self.num_bits)

    @property
    def integer_maximum(self) -> float:
        """
        The maximum integer value that quantized data takes, based on bitwidth.
        """
        return fastforward.quantization.affine.integer_maximum(self.num_bits)

    @property
    def has_uninitialized_params(self) -> bool:
        """
        Check if there any quantization parameters that are unitialized.
        """
        for parameter in self.parameters():
            if isinstance(parameter, torch.nn.parameter.UninitializedParameter):
                return True
        return False

    def reset_parameters(self) -> None:
        """
        Reset parameters to scale=1, offset=0.
        """
        with torch.no_grad():
            self.scale.fill_(1.0)
            if self.offset is not None:
                self.offset.fill_(0.0)

    def _initialize_parameters(self, parameter_dimensionality: int) -> None:  # pylint: disable=arguments-differ
        """
        Internal method to materialize the uninitialized parameters of the quantizer.
        """
        if self.has_uninitialized_params:  # type: ignore[unused-ignore]
            scale_shape = torch.Size([parameter_dimensionality])
            offset_shape = scale_shape if self.offset is not None else None

            with torch.no_grad():
                self.scale.materialize(scale_shape)
                if offset_shape and self.offset is not None:
                    self.offset.materialize(offset_shape)  # type: ignore[attr-defined]
            self.reset_parameters()

    def extra_repr(self) -> str:
        """
        Provide extra repr information on num_bits, symmetric flag and granularities.
        """
        extra_repr = (
            f"num_bits={self.num_bits}, symmetric={self.symmetric}, "
            f"granularity={self.granularity}"
        )
        return super().extra_repr() + extra_repr

    def bound_quantization_function(
        self, tile_size: torch.Size | Literal["data_shape"]
    ) -> BoundQuantizationFunction:
        """
        Creates a `BoundQuantizationFunction` using parameters on this module.
        """
        return fastforward.quantization.affine.quantize_by_tile_function(
            tile_size=tile_size, scale=self.scale, offset=self.offset, num_bits=self.num_bits
        )

    def quantize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Quantize data using linear quantizer.

        Args:
            data: Tensor to quantize

        Returns:
            torch.Tensor:
                Quantized data
        """
        tile_size = self.granularity.tile_size(data.shape)
        quant_func = self.bound_quantization_function(tile_size)
        try:
            return quant_func(data)
        except ValueError as e:
            # If the quantizer has no uninitialized params anymore, reraise the caught
            # ValueError
            # If the quantizer still has uninitialized params, the ValueError was likely
            # raised by UninitializedParameter. Raise a more descriptive ValueError instead.
            if not self.has_uninitialized_params:
                raise e
            else:
                raise ValueError(
                    f"Tried to quantize a tensor using an uninitialized quantizer (of type "
                    f"{type(self).__name__}). This quantizer is initialized after its "
                    f"quantization_range is specified. This can be done explicitly by using "
                    f"the {type(self).__name__}.quantization_range setter or using a range "
                    "setting method."
                ) from e

    def operator_for_range(
        self, min_range: torch.Tensor, max_range: torch.Tensor, data_shape: torch.Size
    ) -> Callable[[torch.Tensor], QuantizedTensor]:
        """
        Part of fastforward.range_setting.SupportsRangeBasedOperator Protocol.
        """
        scale, offset = self._parameters_for_range(min_range, max_range)
        tile_size = self.granularity.tile_size(data_shape)
        return fastforward.quantization.affine.quantize_by_tile_function(
            tile_size=tile_size, scale=scale, offset=offset, num_bits=self.num_bits
        )

    def _parameters_for_range(
        self, min_range: torch.Tensor, max_range: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return fastforward.quantization.affine.parameters_for_range(
            min_range=min_range,
            max_range=max_range,
            num_bits=self.num_bits,
            symmetric=self.symmetric,
            allow_one_sided=self.allow_one_sided,
        )

    @property
    def quantization_range(self) -> tuple[torch.Tensor | float | None, torch.Tensor | float | None]:
        """
        Getter for quantization range.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                `Tuple` of tensor representing the minimum and maximum thresholds of the
                quantization range, respectively.

        Part of fastforward.range_setting.RangeSettable Protocol
        """
        if self.has_uninitialized_params:
            return None, None
        return fastforward.quantization.affine.quantization_range(
            self.scale, self.offset, self.num_bits
        )

    @quantization_range.setter
    def quantization_range(
        self, quant_range: tuple[torch.Tensor | float, torch.Tensor | float]
    ) -> None:
        """
        Setter for quantization range.

        The quantization range is set using Part of
        fastforward.range_setting.RangeSettable Protocol
        """
        try:
            min, max = (ensure_tensor(t, device=self.scale.device) for t in quant_range)
            if self.has_uninitialized_params:
                self._initialize_parameters(min.numel())
        except ValueError as e:
            raise ValueError(
                f"Tried to set quantization range with {len(quant_range)}-tuple. "
                "A 2-tuple is expected"
            ) from e
        except TypeError as e:
            raise ValueError(
                "Tried to set quantization range with a single value. A 2-tuple is expected"
            ) from e

        with torch.no_grad():
            scale, offset = self._parameters_for_range(min, max)
            self.scale.copy_(scale)
            if self.offset is not None:
                if offset is not None:
                    self.offset.copy_(offset)
                else:
                    self.offset.fill_(0.0)
