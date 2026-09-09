# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Minimum error range estimators.

This module contains implementations for range estimators that perform a search to determine
the (near) optimal quantization grid that minimizes a specified error.

Attributes:
    min_error_grid: Alias of `MinErrorGridRangeEstimator`
    mse_grid: Alias of `MinErrorGridRangeEstimator`

"""

import dataclasses
import logging

from math import floor, sqrt
from typing import Callable, Iterator, Literal, Protocol, TypedDict

import torch

from typing_extensions import NotRequired

import fastforward as ff

from fastforward.forward_override import OverrideHandle
from fastforward.nn.linear_quantizer import LinearQuantizer
from fastforward.nn.quantized_module import named_quantizers
from fastforward.nn.quantizer import Quantizer
from fastforward.quantization import affine
from fastforward.quantization.tiled_tensor import tiles_to_rows
from fastforward.range_setting.common import (
    RangeEstimator,
    SimpleEstimatorStep,
    SupportsRangeBasedOperator,
)

logger = logging.getLogger(__name__)
logger.addFilter(ff.logging_utils.DuplicateLogFilter(levels=(logging.WARNING,)))


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
    """Mean Squared Error error function for min_error_grid method.

    Args:
        quantized_data: Data after quantization
        unquantized_data: Data before quantization

    Returns:
        mean squared error between `quantized_data` and `unquantized_data`
    """
    return torch.mean((quantized_data - unquantized_data) ** 2, dim=1)


@dataclasses.dataclass
class _UniformSearchGrid:
    absolute_margin: float = 0.5
    relative_margin: float = 1.0

    def __call__(
        self,
        tiled_data_sample: torch.Tensor,
        symmetric: bool,
        parameter_dimensionality: int,
        num_candidates: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a uniform grid for grid search.

        Returns
            (Torch.Tensor, Torch.Tensor): min_threshold and max_threshold tensors of dimension
                (num_candidates, parameter_dimensionality)
        """
        assert tiled_data_sample.ndim == 2
        assert tiled_data_sample.shape[0] == parameter_dimensionality
        rel_margin = self.relative_margin
        abs_margin = self.absolute_margin

        max_data = rel_margin * tiled_data_sample.max(dim=1).values + abs_margin
        min_data = rel_margin * tiled_data_sample.min(dim=1).values - abs_margin
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

            min_threshold = steps_min_threshold.unsqueeze(1) * (
                rel_margin * min_data.unsqueeze(0) + abs_margin
            )
            max_threshold = steps_max_threshold.unsqueeze(1) * (
                rel_margin * max_data.unsqueeze(0) + abs_margin
            )

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


def uniform_search_grid(
    absolute_margin: float = 0.5, relative_margin: float = 1.0
) -> _UniformSearchGrid:
    """Uniform search grid generator.

    Generates a search grid with a margin based on the provided data.

    The search grid will contain ranges within `(r * min + a, r * max + a)`
    where `r` denotes `relative_margin`, `a` denotes `absolute_margin` and
    `min` and `max` denote    the minimum an maximum value of the observed
    batch.

    Args:
        absolute_margin: The absolute margin to use
        relative_margin: The absolute margin to use

    Return:
        Instance of `_UniformSearchGrid`.
    """
    return _UniformSearchGrid(absolute_margin=absolute_margin, relative_margin=relative_margin)


class _MinAvgErrorGridEstimator(SimpleEstimatorStep[SupportsRangeBasedOperator], torch.nn.Module):
    min_threshold: torch.Tensor
    max_threshold: torch.Tensor
    cumulative_error: torch.Tensor

    # Upper bound, in bytes, on the copy that `_candidate_errors_batched` forces
    # per chunk.
    max_chunk_bytes = 64 * 1024 * 1024

    def __init__(
        self,
        quantizer: SupportsRangeBasedOperator,
        error_fn: _ErrorFn = mse_error,
        num_candidates: int = 100,
        search_grid_generator: _SearchGridGenerator = _UniformSearchGrid(),
        update_range_policy: Callable[["_MinAvgErrorGridEstimator", int], bool] | None = None,
        disable_quantization: bool = False,
        chunk_size: int | None = None,
    ):
        super().__init__(disable_quantization=disable_quantization)
        self._quantizer = quantizer
        self.error_fn = error_fn
        self.num_candidates = num_candidates
        self.nonnegative_data = True
        self.search_grid_generator = search_grid_generator
        self._estimation_steps = 0
        self.update_range_policy = update_range_policy
        if chunk_size is not None and chunk_size < 1:
            msg = "'chunk_size' must be a positive integer or None."
            raise ValueError(msg)
        self._chunk_size = chunk_size

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
        self.cumulative_error = torch.zeros_like(self.min_threshold)

    def _update_quantizer_ranges(self, quantizer: SupportsRangeBasedOperator) -> None:
        best_grid = self.cumulative_error.min(dim=0).indices
        parameter_dimensionality = self.min_threshold.shape[1]
        quant_idx = torch.arange(parameter_dimensionality)
        min_threshold = self.min_threshold[best_grid, quant_idx]
        max_threshold = self.max_threshold[best_grid, quant_idx]
        quantizer.quantization_range = (min_threshold, max_threshold)

    def estimate_step(self, quantizer: SupportsRangeBasedOperator, data: torch.Tensor) -> None:
        with torch.no_grad():
            self.cumulative_error += self._candidate_errors(quantizer, data)

        self._estimation_steps += 1
        if not self.update_range_policy or self.update_range_policy(self, self._estimation_steps):
            self._update_quantizer_ranges(quantizer)

    def _candidate_errors(
        self, quantizer: SupportsRangeBasedOperator, data: torch.Tensor
    ) -> torch.Tensor:
        """Return the per-candidate, per-tile error for `data`.

        The result has the same shape as `cumulative_error`, i.e. one row per
        search grid candidate. Uses a single batched quantization when
        `quantizer` is a plain `LinearQuantizer`, and falls back to one
        `operator_for_range` call per candidate otherwise.
        """
        tile_size = self._quantizer.granularity.tile_size(data.shape)
        tiled_data = tiles_to_rows(data, tile_size)

        # Restricted to exactly `LinearQuantizer`: the batched path reproduces its
        # `operator_for_range` parameterization, which subclasses may override.
        # Narrowed via `object` because `LinearQuantizer` and
        # `SupportsRangeBasedOperator` are not statically compatible.
        candidate: object = quantizer
        if type(candidate) is LinearQuantizer:
            return self._candidate_errors_batched(candidate, tiled_data)
        return self._candidate_errors_per_candidate(quantizer, data, tiled_data, tile_size)

    def _candidate_errors_batched(
        self, quantizer: LinearQuantizer, tiled_data: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate every candidate in a small number of batched quantization calls.

        Each (candidate, tile) pair becomes an independent row of the tiled
        representation, so a `(1, tile_numel)`-tiled quantization applies a
        different scale/offset per row. This replaces one kernel launch per
        candidate with one launch per chunk of candidates.

        `parameters_for_range` is elementwise in the thresholds except for the
        `one_sided` decision, which is derived from `min_range.min()`. Every
        candidate's `min_threshold` in a search grid is a positive multiple of a
        single per-tile vector, so the sign of that minimum -- and hence
        `one_sided` -- is the same for all candidates. Evaluating the whole
        grid at once therefore yields exactly the per-candidate parameters.

        Candidates are processed in chunks rather than all at once: expanding
        `tiled_data` over the candidate dimension and flattening it forces a
        real copy (a broadcast/stride-0 dimension cannot be merged into a
        contiguous reshape without materializing it), so batching every
        candidate together scales memory and allocator traffic with
        `num_candidates` and can be slower than the original per-candidate
        loop for large tensors. Chunking bounds that copy to a fixed byte
        budget regardless of tensor size. On CPU, `_candidate_chunk_size`
        always chunks by a single candidate -- see its docstring for why.
        """
        if quantizer.allow_one_sided:
            candidate_is_one_sided = self.min_threshold.amin(dim=1) >= 0
            if not torch.all(candidate_is_one_sided == candidate_is_one_sided[0]):
                msg = (
                    "Batched candidate evaluation requires all candidates to agree on "
                    "whether their range is one-sided."
                )
                raise ValueError(msg)

        scale, offset = affine.parameters_for_range(
            min_range=self.min_threshold,
            max_range=self.max_threshold,
            num_bits=quantizer.num_bits,
            symmetric=quantizer.symmetric,
            allow_one_sided=quantizer.allow_one_sided,
        )

        num_grid_rows, parameter_dimensionality = self.min_threshold.shape
        tile_numel = tiled_data.shape[1]
        chunk_size = self._candidate_chunk_size(
            parameter_dimensionality, tile_numel, tiled_data.element_size(), tiled_data.device
        )

        error = torch.empty_like(self.cumulative_error)
        for start in range(0, num_grid_rows, chunk_size):
            stop = min(start + chunk_size, num_grid_rows)
            num_chunk_rows = stop - start
            rows = tiled_data.expand(num_chunk_rows, parameter_dimensionality, tile_numel).reshape(
                num_chunk_rows * parameter_dimensionality, tile_numel
            )
            quantized_rows = affine.quantize_per_tile(
                rows,
                scale[start:stop].reshape(-1),
                None if offset is None else offset[start:stop].reshape(-1),
                torch.Size((1, tile_numel)),
                quantizer.num_bits,
                quantizer.quantized_dtype,
            ).dequantize()

            chunk_error = self.error_fn(quantized_rows, rows)
            if not isinstance(chunk_error, torch.Tensor):
                msg = (
                    f"{type(self).__name__} requires 'error_fn' to return a Tensor with one "
                    f"error per row, but it returned {type(chunk_error).__name__}."
                )
                raise TypeError(msg)
            expected_numel = num_chunk_rows * parameter_dimensionality
            if chunk_error.numel() != expected_numel:
                msg = (
                    f"'error_fn' must return {expected_numel} errors, "
                    f"but returned {chunk_error.numel()}."
                )
                raise ValueError(msg)
            error[start:stop] = chunk_error.reshape(num_chunk_rows, parameter_dimensionality)

        return error.to(self.cumulative_error)

    def _candidate_chunk_size(
        self,
        parameter_dimensionality: int,
        tile_numel: int,
        element_size: int,
        device: torch.device,
    ) -> int:
        """Return how many search-grid candidates to batch into one quantization call.

        Bounds the size of the copy that batching forces (see
        `_candidate_errors_batched`) to roughly `max_chunk_bytes`, independent
        of `num_candidates` or tensor size.

        On CPU, batching brings no benefit: unlike a CUDA kernel launch, a CPU
        op dispatch has near-zero fixed overhead, so there is no launch cost
        to amortize, while the forced copy still grows with chunk size and can
        push working set past cache. One candidate per chunk keeps that copy
        minimal, bounding the CPU regression instead of eliminating it.
        """
        if self._chunk_size is not None:
            return self._chunk_size
        if device.type == "cpu":
            return 1
        row_bytes = max(parameter_dimensionality * tile_numel * element_size, 1)
        return max(1, self.max_chunk_bytes // row_bytes)

    def _candidate_errors_per_candidate(
        self,
        quantizer: SupportsRangeBasedOperator,
        data: torch.Tensor,
        tiled_data: torch.Tensor,
        tile_size: torch.Size | Literal["data_shape"],
    ) -> torch.Tensor:
        """Evaluate every candidate with its own `operator_for_range` call."""
        error = torch.empty_like(self.cumulative_error)
        for i in range(self.min_threshold.shape[0]):
            quant_op = quantizer.operator_for_range(
                self.min_threshold[i], self.max_threshold[i], data.shape
            )
            quant_data = quant_op(data).dequantize()
            tiled_quant_data = tiles_to_rows(quant_data, tile_size)
            error[i] = self.error_fn(tiled_quant_data, tiled_data)
        return error


class MinErrorGridRangeEstimator(RangeEstimator[OverrideHandle, Quantizer]):
    """Grid range estimator for error minimization.

    Range Estimator that searches for quantization range that minimizes
    `error_fn` between quantized and non-quantized value.

    A grid search as defined by `search_grid_generator` is performed to find the candidate
    that minimizes the given error.

    `update_range_policy` specifies how often the quantization grids should
    be updated. If no such policy is provided, the ranges are updated after
    every step.

    Args:
        error_fn: The error function `(quantized_data, non_quantized_data)
            -> real-valued error` that is minimized
        num_candidates: The size of the search grid
        search_grid_generator: Callable that defines search grid
        update_range_policy: Callable that defines whether quantizers
            should be updated per step.
        chunk_size: Number of candidates to evaluate in each batch. If `None`,
            a device-dependent chunk size is selected automatically.
        skip_unsupported_quantizers: If True, ignore any quantizer that does
            not not implement
            `fastforward.range_setting.SupportsRangeBasedOperator`. If False,
            a `TypeError` is raised when an unsupported quantizer is
            encountered.

    """

    def __init__(
        self,
        error_fn: _ErrorFn = mse_error,
        num_candidates: int = 100,
        search_grid_generator: _SearchGridGenerator = _UniformSearchGrid(),
        update_range_policy: Callable[["_MinAvgErrorGridEstimator", int], bool] | None = None,
        chunk_size: int | None = None,
        skip_unsupported_quantizers: bool = False,
    ):
        self._error_fn = error_fn
        self._num_candidates = num_candidates
        self._search_grid_generator = search_grid_generator
        self._update_range_policy = update_range_policy
        self._chunk_size = chunk_size
        self._skip_unsupported_quantizers = skip_unsupported_quantizers

    def prepare(self, module: Quantizer) -> OverrideHandle:
        """Prepare `module` for min error range estimation."""
        if not isinstance(module, SupportsRangeBasedOperator):
            proto_name = (
                f"{SupportsRangeBasedOperator.__module__}.{SupportsRangeBasedOperator.__qualname__}"
            )
            msg = f"{type(module).__name__} does not implement {proto_name}."
            raise TypeError(msg)
        return module.register_override(
            _MinAvgErrorGridEstimator(
                module,
                error_fn=self._error_fn,
                num_candidates=self._num_candidates,
                search_grid_generator=self._search_grid_generator,
                update_range_policy=self._update_range_policy,
                chunk_size=self._chunk_size,
            )
        )

    def cleanup(self, module: Quantizer, metadata: OverrideHandle) -> None:
        """Cleanup `module` after min error range estimation."""
        del module
        metadata.remove()

    def split_module(self, module: torch.nn.Module) -> Iterator[Quantizer]:
        """Yields all quantizers in `module`.

        Each is set up for min error range estimation separately.
        """
        for _, quantizer in named_quantizers(module, recurse=True):
            if (
                isinstance(quantizer, SupportsRangeBasedOperator)
                or not self._skip_unsupported_quantizers
            ):
                yield quantizer
            else:
                logger.warning(
                    f"{type(quantizer).__name__} does not implement SupportsRangeBasedOperator."
                    f" Therefore it is not included in {type(self).__name__} range setting."
                )


min_error_grid = MinErrorGridRangeEstimator
mse_grid = MinErrorGridRangeEstimator
