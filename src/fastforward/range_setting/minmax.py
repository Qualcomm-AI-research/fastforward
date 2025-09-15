# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import logging

from typing import Iterator, TypeVar

import torch

import fastforward as ff

from fastforward.forward_override import OverrideHandle
from fastforward.nn.quantized_module import named_quantizers
from fastforward.nn.quantizer import Quantizer
from fastforward.quantization.tiled_tensor import tiles_to_rows

from .common import RangeEstimator, RangeSettable, SimpleEstimatorStep

T = TypeVar("T")


logger = logging.getLogger(__name__)
logger.addFilter(ff.logging_utils.DuplicateLogFilter(levels=(logging.WARNING,)))


class SmoothedMinMaxEstimator(SimpleEstimatorStep[RangeSettable], torch.nn.Module):
    """Estimates the quantization range using exponential smoothing of the minimum and maximum values.

    Attributes:
        min: The minimum values observed.
        max: The maximum values observed.
        gamma: The smoothing factor.
    """

    min: torch.Tensor | None
    max: torch.Tensor | None

    def __init__(
        self, quantizer: RangeSettable, gamma: float = 1.0, disable_quantization: bool = False
    ) -> None:
        """Initialize the SmoothedMinMaxEstimator.

        Args:
            quantizer: The quantizer to estimate ranges for.
            gamma: The smoothing factor.
            disable_quantization: Flag to disable quantization.
        """
        super().__init__(disable_quantization=disable_quantization)
        min, max = quantizer.quantization_range
        self.gamma = gamma
        self.register_buffer("min", min)
        self.register_buffer("max", max)

    def initialize_parameters(self, quantizer: RangeSettable, data: torch.Tensor) -> None:
        """Initialize the min and max parameters.

        Args:
            quantizer: The quantizer to estimate ranges for.
            data: The input data tensor.
        """
        shape = (quantizer.granularity.parameter_dimensionality(data.shape),)
        if self.min is None:
            self.min = torch.full(shape, float("inf"), dtype=data.dtype, device=data.device)
        if self.max is None:
            self.max = torch.full(shape, float("-inf"), dtype=data.dtype, device=data.device)

    def estimate_step(self, quantizer: RangeSettable, data: torch.Tensor) -> None:
        """Perform a single estimation step.

        Args:
            quantizer: The quantizer to estimate ranges for.
            data: The input data tensor.
        """
        self.initialize_parameters(quantizer, data)
        assert self.min is not None
        assert self.max is not None

        with torch.no_grad():
            tile_size = quantizer.granularity.tile_size(data.shape)
            reshaped_data = tiles_to_rows(data, tile_size)
            data_min = torch.min(reshaped_data, -1).values
            data_max = torch.max(reshaped_data, -1).values

            # If min or max is infinity, reset everything to new batch
            if self.min.isinf().any() or self.max.isinf().any():
                self.min = data_min
                self.max = data_max
            else:
                self.min = self.gamma * data_min + (1 - self.gamma) * self.min
                self.max = self.gamma * data_max + (1 - self.gamma) * self.max

        quantizer.quantization_range = (self.min, self.max)

    def extra_repr(self) -> str:
        """Return a string representation of the estimator.

        Returns:
            str: A string representation of the estimator.
        """
        return f"min={self.min}, max={self.max}"


class SmoothedMinMaxRangeEstimator(RangeEstimator[OverrideHandle, Quantizer]):
    """Exponential moving average range estimator.

    Estimates the ranges based on a exponential moving average over batches of
    the minimum and maximum for each quantizer.

    Attributes:
        gamma: The smoothing factor.
        disable_quantization: Flag to disable quantization.
        skip_unsupported_quantizers: If True, ignore any quantizer that does
            not not implement `fastforward.range_setting.RangeSettable`. If
            False, an error `TypeError` is thrown when an unsupported quantizer
            is encountered.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        disable_quantization: bool = False,
        skip_unsupported_quantizers: bool = False,
    ) -> None:
        self.gamma = gamma
        self.disable_quantization = disable_quantization
        self.skip_unsupported_quantizers = skip_unsupported_quantizers

    def prepare(self, module: Quantizer) -> OverrideHandle:
        """Prepare the module for range estimation.

        Args:
            module: The quantizer module.

        Returns:
            OverrideHandle: The override handle for the module.
        """
        if not isinstance(module, RangeSettable):
            proto_name = f"{RangeSettable.__module__}.{RangeSettable.__qualname__}"
            msg = f"{type(module).__name__} does not implement {proto_name}."
            raise TypeError(msg)
        return module.register_override(
            SmoothedMinMaxEstimator(
                module, gamma=self.gamma, disable_quantization=self.disable_quantization
            )
        )

    def cleanup(self, module: Quantizer, metadata: OverrideHandle) -> None:
        """Clean up the module after range estimation.

        Args:
            module: The quantizer module.
            metadata: The override handle for the module.
        """
        del module
        metadata.remove()

    def split_module(self, module: torch.nn.Module) -> Iterator[Quantizer]:
        """Split the module into individual quantizers.

        Args:
            module: The module to split.

        Yields:
            Iterator[Quantizer]: An iterator over the quantizers.
        """
        for _, quantizer in named_quantizers(module, recurse=True):
            if isinstance(quantizer, RangeSettable) or not self.skip_unsupported_quantizers:
                yield quantizer
            else:
                logger.warning(
                    f"{type(quantizer).__name__} does not implement RangeSettable. Therefore "
                    f"it is not included in {type(self).__name__} range setting."
                )


smoothed_minmax = SmoothedMinMaxRangeEstimator


class RunningMinMaxEstimator(SimpleEstimatorStep[RangeSettable], torch.nn.Module):
    """Estimates the quantization range using the running minimum and maximum values.

    Attributes:
        min: The minimum values observed.
        max: The maximum values observed.
    """

    min: torch.Tensor | None
    max: torch.Tensor | None

    def __init__(self, quantizer: RangeSettable, disable_quantization: bool = False) -> None:
        """Initialize the RunningMinMaxEstimator.

        Args:
            quantizer: The quantizer to estimate ranges for.
            disable_quantization: Flag to disable quantization.
        """
        super().__init__(disable_quantization=disable_quantization)
        min, max = quantizer.quantization_range
        self.register_buffer("min", min)
        self.register_buffer("max", max)

    def initialize_parameters(self, quantizer: RangeSettable, data: torch.Tensor) -> None:
        """Initialize the min and max parameters.

        Args:
            quantizer: The quantizer to estimate ranges for.
            data: The input data tensor.
        """
        shape = (quantizer.granularity.parameter_dimensionality(data.shape),)
        if self.min is None:
            self.min = torch.full(shape, float("inf"), dtype=data.dtype, device=data.device)
        if self.max is None:
            self.max = torch.full(shape, float("-inf"), dtype=data.dtype, device=data.device)

    def estimate_step(self, quantizer: RangeSettable, data: torch.Tensor) -> None:
        """Perform a single estimation step.

        Args:
            quantizer: The quantizer to estimate ranges for.
            data: The input data tensor.
        """
        self.initialize_parameters(quantizer, data)
        assert self.min is not None
        assert self.max is not None

        with torch.no_grad():
            tile_size = quantizer.granularity.tile_size(data.shape)
            reshaped_data = tiles_to_rows(data, tile_size)
            data_min = torch.min(reshaped_data, -1).values
            data_max = torch.max(reshaped_data, -1).values

            # Look for -inf in data_min and inf in data_max and replace. See issue #167
            if data_min.isinf().any() or data_max.isinf().any():
                raise NotImplementedError("Infinite")

            self.min = torch.min(self.min, data_min)
            self.max = torch.max(self.max, data_max)

        quantizer.quantization_range = (self.min, self.max)

    def extra_repr(self) -> str:
        """Return a string representation of the estimator.

        Returns:
            str: A string representation of the estimator.
        """
        return f"min={self.min}, max={self.max}"


class RunningMinMaxRangeEstimator(RangeEstimator[OverrideHandle, Quantizer]):
    """Running min-max range estimator.

    Estimates the quantization ranges based on the minimum and maximum over all
    data seen for each quantizer.

    Attributes:
        disable_quantization: Flag to disable quantization.
        skip_unsupported_quantizers: If True, ignore any quantizer that does
            not not implement `fastforward.range_setting.RangeSettable`. If
            False, an error `TypeError` is thrown when an unsupported quantizer
            is encountered.
    """

    def __init__(
        self,
        disable_quantization: bool = False,
        skip_unsupported_quantizers: bool = False,
    ) -> None:
        self.disable_quantization = disable_quantization
        self.skip_unsupported_quantizers = skip_unsupported_quantizers

    def prepare(self, module: Quantizer) -> OverrideHandle:
        """Prepare the module for range estimation.

        Args:
            module: The quantizer module.

        Returns:
            OverrideHandle: The override handle for the module.
        """
        if not isinstance(module, RangeSettable):
            proto_name = f"{RangeSettable.__module__}.{RangeSettable.__qualname__}"
            msg = f"{type(module).__name__} does not implement {proto_name}."
            raise TypeError(msg)
        return module.register_override(
            RunningMinMaxEstimator(module, disable_quantization=self.disable_quantization)
        )

    def cleanup(self, module: Quantizer, metadata: OverrideHandle) -> None:
        """Cleanup after range estimation has concluded."""
        del module
        metadata.remove()

    def split_module(self, module: torch.nn.Module) -> Iterator[Quantizer]:
        """Split module up into separate quantizers."""
        for _, quantizer in named_quantizers(module, recurse=True):
            if isinstance(quantizer, RangeSettable) or not self.skip_unsupported_quantizers:
                yield quantizer
            else:
                logger.warning(
                    f"{type(quantizer).__name__} does not implement RangeSettable. Therefore "
                    f"it is not included in {type(self).__name__} range setting."
                )


running_minmax = RunningMinMaxRangeEstimator
