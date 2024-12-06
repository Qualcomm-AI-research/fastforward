# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

"""
Minimum error range estimators.

This module contains implementations for range estimators that perform a search to determine
the (near) optimal quantization grid that minimizes a specified error.

Attributes:
    min_error_grid: Alias of `MinErrorGridRangeEstimator`
    mse_grid: Alias of `MinErrorGridRangeEstimator`

"""

from math import floor, sqrt
from typing import Callable, Iterator, Optional, Protocol, TypedDict

import torch

from typing_extensions import NotRequired

from fastforward.forward_override import OverrideHandle
from fastforward.nn.quantized_module import named_quantizers
from fastforward.nn.quantizer import Quantizer
from fastforward.quantization.tiled_tensor import tiles_to_rows
from fastforward.range_setting.common import (
    RangeEstimator,
    SimpleEstimatorStep,
    SupportsRangeBasedOperator,
)


class _TensorKwargs(TypedDict):
    dtype: NotRequired[torch.dtype]
    device: NotRequired[torch.device]


class _ErrorFn(Protocol):
    def __call__(
        self, __quantized_data: torch.Tensor, __original_data: torch.Tensor
    ) -> torch.Tensor | float:
        raise NotImplementedError


class _SearchGridGenerator(Protocol):
    def __call__(
        self,
        __data_sample: torch.Tensor,
        __symmetric: bool,
        __parameter_dimensionality: int,
        __num_candidates: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


def mse_error(quantized_data: torch.Tensor, unquantized_data: torch.Tensor) -> torch.Tensor:
    """
    Mean Squared Error error function for min_error_grid method.

    Args:
        quantized_data: Data after quantization
        unquantized_data: Data before quantization

    Returns:
        mean squared error between `quantized_data` and `unquantized_data`
    """
    return torch.mean((quantized_data - unquantized_data) ** 2, dim=1)


def _default_search_grid(
    tiled_data_sample: torch.Tensor,
    symmetric: bool,
    parameter_dimensionality: int,
    num_candidates: int,
    range_margin: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a default grid for grid search.

    Returns
        (Torch.Tensor, Torch.Tensor): min_threshold and max_threshold tensors of dimension
            (num_candidates, parameter_dimensionality)
    """
    assert tiled_data_sample.ndim == 2
    assert tiled_data_sample.shape[0] == parameter_dimensionality

    min_data = tiled_data_sample.min(dim=1).values - range_margin
    max_data = tiled_data_sample.max(dim=1).values + range_margin
    negative_data = min_data.min() < 0

    tkwargs: _TensorKwargs = {
        "dtype": tiled_data_sample.dtype,
        "device": tiled_data_sample.device,
    }
    if not negative_data:
        # Case, search from 0ish - max for upper min_threshold
        # Potentially, also search for lower threshold 0ish-max?
        min_threshold = torch.zeros((num_candidates, parameter_dimensionality), **tkwargs)
        steps = torch.linspace(1 / num_candidates, 1, num_candidates, **tkwargs)
        max_threshold = steps.unsqueeze(1) * max_data.unsqueeze(0)

    elif not symmetric:  # and negative_data
        # Asymmetric, search a combinatorial search space of:
        #   min_range: [min_range, margin * min_range] x [max_range, margin * max_range]
        margin = 0.6

        num_candidates_min_threshold = floor(sqrt(num_candidates))
        num_candidates_max_threshold = (
            num_candidates_min_threshold + num_candidates - num_candidates_min_threshold**2
        )

        steps_min_threshold = torch.linspace(1, margin, num_candidates_min_threshold, **tkwargs)
        steps_max_threshold = torch.linspace(margin, 1, num_candidates_max_threshold, **tkwargs)

        min_threshold = steps_min_threshold.unsqueeze(1) * min_data.unsqueeze(0)
        max_threshold = steps_max_threshold.unsqueeze(1) * max_data.unsqueeze(0)

        # Make combinatorial optimization grid
        min_threshold = min_threshold.repeat(num_candidates_max_threshold, 1)
        max_threshold = max_threshold.repeat_interleave(num_candidates_min_threshold, dim=0)

    else:  # symmetric and negative_data
        # search [(-delta, delta), ... (-max, max)]
        steps = torch.linspace(1 / num_candidates, 1, num_candidates, **tkwargs)
        max_abs_data = torch.max(torch.abs(min_data), torch.abs(max_data))
        max_threshold = steps.unsqueeze(1) * max_abs_data.unsqueeze(0)
        min_threshold = -max_threshold

    return min_threshold, max_threshold


class _MinAvgErrorGridEstimator(SimpleEstimatorStep[SupportsRangeBasedOperator], torch.nn.Module):
    min_threshold: torch.Tensor
    max_threshold: torch.Tensor
    cummulative_error: torch.Tensor

    def __init__(
        self,
        quantizer: SupportsRangeBasedOperator,
        error_fn: _ErrorFn = mse_error,
        num_candidates: int = 100,
        search_grid_generator: _SearchGridGenerator = _default_search_grid,
        update_range_policy: Optional[Callable[["_MinAvgErrorGridEstimator", int], bool]] = None,
        disable_quantization: bool = False,
    ):
        super().__init__(disable_quantization=disable_quantization)
        self._quantizer = quantizer
        self.error_fn = error_fn
        self.num_candidates = num_candidates
        self.nonnegative_data = True
        self.search_grid_generator = search_grid_generator
        self._estimation_steps = 0
        self.update_range_policy = update_range_policy

    def setup_estimator(self, data: torch.Tensor) -> None:
        self._estimation_steps = 0
        self._initialize_search_grid(data)

    def _initialize_search_grid(self, data: torch.Tensor) -> None:
        parameter_dimensionality = self._quantizer.granularity.parameter_dimensionality(data.shape)
        tile_size = self._quantizer.granularity.tile_size(data.shape)
        tiled_data = tiles_to_rows(data, tile_size)

        self.min_threshold, self.max_threshold = self.search_grid_generator(
            tiled_data, self._quantizer.symmetric, parameter_dimensionality, self.num_candidates
        )
        self.cumulative_error = torch.zeros(
            self.min_threshold.shape, device=data.device, dtype=data.dtype
        )

    def _update_quantizer_ranges(self, quantizer: SupportsRangeBasedOperator) -> None:
        best_grid = self.cumulative_error.min(dim=0).indices
        parameter_dimensionality = self.min_threshold.shape[1]
        quant_idx = torch.arange(parameter_dimensionality)
        min_threshold = self.min_threshold[best_grid, quant_idx]
        max_threshold = self.max_threshold[best_grid, quant_idx]
        quantizer.quantization_range = (min_threshold, max_threshold)

    def estimate_step(self, quantizer: SupportsRangeBasedOperator, data: torch.Tensor) -> None:
        for i in range(self.num_candidates):
            quant_op = quantizer.operator_for_range(
                self.min_threshold[i], self.max_threshold[i], data.shape
            )
            quant_data = quant_op(data).dequantize()
            tile_size = self._quantizer.granularity.tile_size(data.shape)
            tiled_data = tiles_to_rows(data, tile_size)
            tiled_quant_data = tiles_to_rows(quant_data, tile_size)
            err = self.error_fn(tiled_quant_data, tiled_data)
            self.cumulative_error[i] += err

        self._estimation_steps += 1
        if not self.update_range_policy or self.update_range_policy(self, self._estimation_steps):
            self._update_quantizer_ranges(quantizer)


class MinErrorGridRangeEstimator(RangeEstimator[OverrideHandle, Quantizer]):
    """
    Grid range estimator for error minimization.

    Range Estimator that searches for quantization range that minimizes
    `error_fn` between quantized and non-quantized value.

    A grid search as defined by `search_grid_generator` is performed to find the candidate
    that minimizes the given error.

    `update_range_policy` specifies how often the the quantization grids should
    be updated. If no such policy is provided, the ranges are updated after
    every step.

    Args:
        error_fn: The error function `(quantized_data, non_quantized_data)
            -> real-valued error` that is minimized
        num_canidates: The size of the search grid
        search_grid_generator: Callable that defines search grid
        update_range_policy: Callable that defines whether quantizers
            should be updated per step.

    """

    def __init__(
        self,
        error_fn: _ErrorFn = mse_error,
        num_candidates: int = 100,
        search_grid_generator: _SearchGridGenerator = _default_search_grid,
        update_range_policy: Optional[Callable[["_MinAvgErrorGridEstimator", int], bool]] = None,
    ):
        self._error_fn = error_fn
        self._num_candidates = num_candidates
        self._search_grid_generator = search_grid_generator
        self._update_range_policy = update_range_policy

    def prepare(self, module: Quantizer) -> OverrideHandle:
        """
        Prepare `module` for min error range estimation.
        """
        if not isinstance(module, SupportsRangeBasedOperator):
            proto_name = (
                f"{SupportsRangeBasedOperator.__module__}.{SupportsRangeBasedOperator.__qualname__}"
            )
            raise TypeError(f"{type(module).__name__} does not implement {proto_name}.")
        return module.register_override(
            _MinAvgErrorGridEstimator(
                module,
                error_fn=self._error_fn,
                num_candidates=self._num_candidates,
                search_grid_generator=self._search_grid_generator,
                update_range_policy=self._update_range_policy,
            )
        )

    def cleanup(self, module: Quantizer, metadata: OverrideHandle) -> None:
        """
        Cleanup `module` after min error range estimation.
        """
        del module
        metadata.remove()

    @classmethod
    def split_module(cls, module: torch.nn.Module) -> Iterator[Quantizer]:
        """
        Yields all quantizers in `module`.

        Each is set up for min error range estimation seperately.
        """
        for _, quantizer in named_quantizers(module, recurse=True):
            yield quantizer


min_error_grid = MinErrorGridRangeEstimator
mse_grid = MinErrorGridRangeEstimator
