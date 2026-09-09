# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Callable, cast

import pytest
import torch

from fastforward.nn.linear_quantizer import LinearQuantizer
from fastforward.quantization import granularity
from fastforward.quantization.granularity import Granularity
from fastforward.quantization.tiled_tensor import tiles_to_rows
from fastforward.quantized_tensor import QuantizedTensor
from fastforward.range_setting import estimate_ranges
from fastforward.range_setting.common import SupportsRangeBasedOperator
from fastforward.range_setting.min_error import (
    MinErrorGridRangeEstimator,
    _MinAvgErrorGridEstimator,
    _UniformSearchGrid,
    mse_error,
    mse_grid,
)
from typing_extensions import override


@pytest.mark.slow
@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("negative_data", [True, False])
@pytest.mark.parametrize(
    "quant_granularity",
    [
        granularity.PerChannel(0),
        granularity.PerChannel(-1),
        granularity.PerTensor(),
        granularity.PerTile((12, 16)),
    ],
)
def test_mse_grid_estimator_decreasing_error_by_num_candidates(
    symmetric: bool,
    negative_data: bool,
    quant_granularity: Granularity,
    _seed_prngs: int,
) -> None:
    # The search grids (the size of which is defined by the num_cadidates argument)
    # need to display coherence, ie the larger grid needs to contain all the points
    # of the smaller grid. If that condition is not fulfillied there is no
    # guarantee that the performance will be better for the larger grid parameter space.
    # For the symmetric case that is simple, bur the asymmetric case separates the space
    # based on the sqrt(num_candidates). For this reason we choose the value 9 and 81, because
    # they display this coherence for both cases.
    data = torch.randn(24, 16)
    if not negative_data:
        data = data.abs()
    quantizer = LinearQuantizer(8, granularity=quant_granularity, symmetric=symmetric)

    with estimate_ranges(quantizer, mse_grid, num_candidates=9):
        quantizer(data)

    err1 = torch.sum((quantizer(data).dequantize() - data) ** 2)

    with estimate_ranges(quantizer, mse_grid, num_candidates=81):
        quantizer(data)

    err2 = torch.sum((quantizer(data).dequantize() - data) ** 2)

    assert err1 >= err2, (
        "if grid is ^2 times bigger, the selected range must result in lower or equal error"
    )


@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("negative_data", [True, False])
def test__UniformSearchGrid(symmetric: bool, negative_data: bool) -> None:
    num_candidates = 3
    parameter_dimensionality = 5

    data = torch.rand((parameter_dimensionality, 7))
    if not negative_data:
        data = data.abs()

    search_grid_generator = _UniformSearchGrid()
    min_threshold, max_threshold = search_grid_generator(
        data,
        symmetric=symmetric,
        parameter_dimensionality=parameter_dimensionality,
        num_candidates=num_candidates,
    )

    assert min_threshold.shape == (num_candidates, parameter_dimensionality)
    assert max_threshold.shape == (num_candidates, parameter_dimensionality)
    assert (min_threshold < max_threshold).all()


def _l3_error(quantized_data: torch.Tensor, unquantized_data: torch.Tensor) -> torch.Tensor:
    """Non-MSE error function, to exercise the custom `error_fn` path."""
    return (quantized_data - unquantized_data).abs().pow(3).mean(dim=1)


def _too_few_errors(quantized_data: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
    """Return an invalid error vector to exercise batch-result validation."""
    return torch.zeros(1, device=quantized_data.device)


class _CustomLinearQuantizer(LinearQuantizer):
    """`LinearQuantizer` subclass with an overridden `operator_for_range`.

    Used to confirm that the batched fast path -- which reimplements
    `LinearQuantizer.operator_for_range`'s parameterization rather than calling
    it -- is only taken for plain `LinearQuantizer` instances, never for
    subclasses that may override that method.
    """

    @override
    def operator_for_range(
        self, min_range: torch.Tensor, max_range: torch.Tensor, data_shape: torch.Size
    ) -> Callable[[torch.Tensor], QuantizedTensor]:
        base_operator = super().operator_for_range(min_range, max_range, data_shape)
        return lambda data: base_operator(data + 1000.0)


@pytest.mark.parametrize("error_fn", [mse_error, _l3_error])
@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("negative_data", [True, False])
@pytest.mark.parametrize(
    "quant_granularity",
    [
        granularity.PerChannel(0),
        granularity.PerChannel(-1),
        granularity.PerTensor(),
        granularity.PerTile((4, 4)),
    ],
)
def test_candidate_errors_batched_matches_per_candidate(
    quant_granularity: Granularity,
    negative_data: bool,
    symmetric: bool,
    error_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    _seed_prngs: int,
) -> None:
    # GIVEN a quantizer and an estimator with an initialized search grid
    data = torch.randn(8, 8)
    if not negative_data:
        data = data.abs()
    quantizer = LinearQuantizer(4, granularity=quant_granularity, symmetric=symmetric)
    range_based_quantizer = cast(SupportsRangeBasedOperator, quantizer)
    estimator = _MinAvgErrorGridEstimator(
        range_based_quantizer, error_fn=error_fn, num_candidates=5
    )
    estimator.setup_estimator(data)
    tile_size = quantizer.granularity.tile_size(data.shape)
    tiled_data = tiles_to_rows(data, tile_size)

    # WHEN computing the per-candidate error surface via the batched fast path
    # and via the reference per-candidate loop
    batched = estimator._candidate_errors_batched(quantizer, tiled_data)
    per_candidate = estimator._candidate_errors_per_candidate(
        range_based_quantizer, data, tiled_data, tile_size
    )

    # THEN the two must agree exactly: batching only changes how candidates are
    # grouped into quantization calls, not the arithmetic performed per candidate
    torch.testing.assert_close(batched, per_candidate)


def test_candidate_errors_batched_chunking_does_not_change_result(_seed_prngs: int) -> None:
    # GIVEN two estimators configured with different batch sizes
    data = torch.randn(6, 6)
    quantizer = LinearQuantizer(4, granularity=granularity.PerChannel(0), symmetric=False)
    range_based_quantizer = cast(SupportsRangeBasedOperator, quantizer)
    unchunked_estimator = _MinAvgErrorGridEstimator(
        range_based_quantizer, num_candidates=7, chunk_size=7
    )
    chunked_estimator = _MinAvgErrorGridEstimator(
        range_based_quantizer, num_candidates=7, chunk_size=1
    )
    unchunked_estimator.setup_estimator(data)
    chunked_estimator.setup_estimator(data)
    tile_size = quantizer.granularity.tile_size(data.shape)
    tiled_data = tiles_to_rows(data, tile_size)

    # WHEN each estimator computes the candidate error surface
    unchunked = unchunked_estimator._candidate_errors_batched(quantizer, tiled_data)
    chunked = chunked_estimator._candidate_errors_batched(quantizer, tiled_data)

    # THEN the chunk boundaries must not affect the result
    torch.testing.assert_close(chunked, unchunked)


def test_candidate_errors_batched_rejects_wrong_number_of_errors() -> None:
    # GIVEN an error function that does not return one error per batched row
    data = torch.randn(4, 4)
    quantizer = LinearQuantizer(4, granularity=granularity.PerTensor())
    estimator = _MinAvgErrorGridEstimator(
        cast(SupportsRangeBasedOperator, quantizer),
        error_fn=_too_few_errors,
        num_candidates=2,
        chunk_size=2,
    )
    estimator.setup_estimator(data)
    tile_size = quantizer.granularity.tile_size(data.shape)
    tiled_data = tiles_to_rows(data, tile_size)

    # WHEN evaluating candidates in a batch
    with pytest.raises(ValueError):
        estimator._candidate_errors_batched(quantizer, tiled_data)


def test_candidate_chunk_size_bounds_the_forced_copy() -> None:
    # GIVEN a row size that divides the byte budget evenly, on a non-CPU device
    max_chunk_bytes = _MinAvgErrorGridEstimator.max_chunk_bytes
    element_size = 4
    parameter_dimensionality = 4
    tile_numel = max_chunk_bytes // (4 * element_size * parameter_dimensionality)
    row_bytes = parameter_dimensionality * tile_numel * element_size
    quantizer = LinearQuantizer(4, granularity=granularity.PerTensor())
    estimator = _MinAvgErrorGridEstimator(cast(SupportsRangeBasedOperator, quantizer))

    # WHEN computing the chunk size for that row size
    chunk_size = estimator._candidate_chunk_size(
        parameter_dimensionality, tile_numel, element_size, torch.device("cuda")
    )

    # THEN the chunk must fit the budget, and be the largest one that does
    assert chunk_size * row_bytes <= max_chunk_bytes
    assert (chunk_size + 1) * row_bytes > max_chunk_bytes


def test_candidate_chunk_size_never_returns_zero() -> None:
    # GIVEN a single row far larger than the byte budget, on a non-CPU device
    quantizer = LinearQuantizer(4, granularity=granularity.PerTensor())
    estimator = _MinAvgErrorGridEstimator(cast(SupportsRangeBasedOperator, quantizer))

    # WHEN computing the chunk size
    chunk_size = estimator._candidate_chunk_size(
        parameter_dimensionality=1,
        tile_numel=_MinAvgErrorGridEstimator.max_chunk_bytes,
        element_size=4,
        device=torch.device("cuda"),
    )

    # THEN it must floor at 1 rather than 0, which would loop forever in the caller
    assert chunk_size == 1


def test_candidate_chunk_size_is_one_on_cpu() -> None:
    # GIVEN a row size that would batch many candidates per chunk on a GPU
    quantizer = LinearQuantizer(4, granularity=granularity.PerTensor())
    estimator = _MinAvgErrorGridEstimator(cast(SupportsRangeBasedOperator, quantizer))

    # WHEN computing the chunk size for a CPU tensor
    chunk_size = estimator._candidate_chunk_size(
        parameter_dimensionality=1, tile_numel=1, element_size=4, device=torch.device("cpu")
    )

    # THEN CPU always processes one candidate per chunk: there is no launch
    # overhead to amortize there, so batching only adds a bigger forced copy
    assert chunk_size == 1


def test_min_error_grid_estimator_passes_chunk_size_to_override() -> None:
    # GIVEN a public estimator with a manual chunk size
    quantizer = LinearQuantizer(4, granularity=granularity.PerTensor())
    range_estimator = MinErrorGridRangeEstimator(chunk_size=3)

    # WHEN preparing a quantizer for range estimation
    handle = range_estimator.prepare(quantizer)
    override_fn = next(quantizer.overrides)

    # THEN the registered estimator receives the configured chunk size
    assert isinstance(override_fn, _MinAvgErrorGridEstimator)
    assert override_fn._chunk_size == 3
    range_estimator.cleanup(quantizer, handle)


def test_candidate_errors_batched_rejects_mixed_one_sided_ranges(_seed_prngs: int) -> None:
    # GIVEN candidates that disagree on whether their range is one-sided
    data = torch.randn(4, 4)
    quantizer = LinearQuantizer(4, granularity=granularity.PerTensor())
    estimator = _MinAvgErrorGridEstimator(
        cast(SupportsRangeBasedOperator, quantizer), num_candidates=2
    )
    estimator.setup_estimator(data)
    estimator.min_threshold[0].fill_(-1)
    estimator.min_threshold[1].fill_(1)
    tile_size = quantizer.granularity.tile_size(data.shape)
    tiled_data = tiles_to_rows(data, tile_size)

    # WHEN evaluating candidates in a batch
    with pytest.raises(ValueError, match="one-sided"):
        estimator._candidate_errors_batched(quantizer, tiled_data)


def test_candidate_errors_uses_fallback_for_quantizer_subclasses(_seed_prngs: int) -> None:
    # GIVEN a LinearQuantizer subclass that overrides `operator_for_range`
    data = torch.randn(6, 6)
    quantizer = _CustomLinearQuantizer(4, granularity=granularity.PerTensor(), symmetric=False)
    range_based_quantizer = cast(SupportsRangeBasedOperator, quantizer)
    estimator = _MinAvgErrorGridEstimator(range_based_quantizer, num_candidates=4)
    estimator.setup_estimator(data)
    tile_size = quantizer.granularity.tile_size(data.shape)
    tiled_data = tiles_to_rows(data, tile_size)

    # WHEN the dispatcher picks an error computation path for this quantizer
    dispatched = estimator._candidate_errors(range_based_quantizer, data)
    expected = estimator._candidate_errors_per_candidate(
        range_based_quantizer, data, tiled_data, tile_size
    )

    # THEN it must use the per-candidate path (and hence the override): the
    # batched path reimplements the base `LinearQuantizer` parameterization and
    # would silently ignore the subclass's override if used here
    torch.testing.assert_close(dispatched, expected)
