# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import torch

from fastforward.nn.linear_quantizer import LinearQuantizer
from fastforward.quantization import granularity
from fastforward.range_setting import estimate_ranges, running_minmax, smoothed_minmax


def test_smoothed_minmax_range_setting_per_tensor(_seed_prngs: int) -> None:
    quantizer = LinearQuantizer(num_bits=3)

    data = torch.randn(10, 8)
    data_min, data_max = data.min(), data.max()

    with estimate_ranges(quantizer, smoothed_minmax):
        estimator = list(quantizer._quantizer_overrides.values())[0]
        quantizer(data)

    assert estimator.min == data_min  # type: ignore[attr-defined]
    assert estimator.max == data_max  # type: ignore[attr-defined]


def test_smoothed_minmax_range_setting_per_channel(_seed_prngs: int) -> None:
    quantizer = LinearQuantizer(num_bits=3, granularity=granularity.PerChannel(0))

    data = torch.randn(10, 8)
    data_min, data_max = data.min(-1).values, data.max(-1).values

    with estimate_ranges(quantizer, smoothed_minmax):
        estimator = list(quantizer._quantizer_overrides.values())[0]
        quantizer(data)

    estimator_min = estimator.min  # type: ignore[attr-defined]
    estimator_max = estimator.max  # type: ignore[attr-defined]
    torch.testing.assert_close(estimator_min, data_min)
    torch.testing.assert_close(estimator_max, data_max)


def test_running_minmax_range_setting_per_tensor(_seed_prngs: int) -> None:
    quantizer = LinearQuantizer(num_bits=3)

    batch = torch.randn(10, 8)
    batches = torch.stack([
        batch * 0.3,
        batch,
        batch * 5,
        batch * 0.5,
        batch,
    ])

    data_min, data_max = batches.min(), batches.max()

    with estimate_ranges(quantizer, running_minmax):
        estimator = list(quantizer._quantizer_overrides.values())[0]
        for batch in batches:
            quantizer(batch)

    assert estimator.min == data_min  # type: ignore[attr-defined]
    assert estimator.max == data_max  # type: ignore[attr-defined]


def test_running_minmax_range_setting_per_channel(_seed_prngs: int) -> None:
    quantizer = LinearQuantizer(num_bits=3, granularity=granularity.PerChannel(0))

    batch = torch.randn(10, 8)
    batches = torch.stack([
        batch * 0.3,
        batch,
        batch * 5,
        batch * 0.5,
        batch,
    ])

    data_min = batches.min(-1).values.min(0).values
    data_max = batches.max(-1).values.max(0).values

    with estimate_ranges(quantizer, running_minmax):
        estimator = list(quantizer._quantizer_overrides.values())[0]
        for batch in batches:
            quantizer(batch)

    estimator_min = estimator.min  # type: ignore[attr-defined]
    estimator_max = estimator.max  # type: ignore[attr-defined]
    torch.testing.assert_close(estimator_min, data_min)
    torch.testing.assert_close(estimator_max, data_max)
