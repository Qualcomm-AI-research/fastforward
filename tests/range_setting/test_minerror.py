# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import pytest
import torch

from fastforward.nn.linear_quantizer import LinearQuantizer
from fastforward.quantization import granularity
from fastforward.quantization.granularity import Granularity
from fastforward.range_setting import estimate_ranges
from fastforward.range_setting.min_error import _default_search_grid, mse_grid


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
def test__default_search_grid(symmetric: bool, negative_data: bool) -> None:
    num_candidates = 3
    parameter_dimensionality = 5

    data = torch.rand((parameter_dimensionality, 7))
    if not negative_data:
        data = data.abs()

    min_threshold, max_threshold = _default_search_grid(
        data,
        symmetric=symmetric,
        parameter_dimensionality=parameter_dimensionality,
        num_candidates=num_candidates,
    )

    assert min_threshold.shape == (num_candidates, parameter_dimensionality)
    assert max_threshold.shape == (num_candidates, parameter_dimensionality)
    assert (min_threshold < max_threshold).all()
