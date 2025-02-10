# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import math

import pytest
import torch

from fastforward.quantization import granularity


@pytest.mark.parametrize(
    "data_shape",
    [
        (2,),
        (
            2,
            10,
        ),
        (2, 10, 30),
    ],
)
def test_per_tensor_granularity(data_shape):
    data_shape = torch.Size(data_shape)
    gran = granularity.PerTensor()
    tile_size = gran.tile_size(data_shape)

    assert tile_size == "data_shape"
    assert gran.parameter_dimensionality(data_shape) == 1


@pytest.mark.parametrize(
    "data_shape",
    [
        (2,),
        (
            2,
            10,
        ),
        (2, 10, 30),
    ],
)
def test_per_channel_granularity_single_channel(data_shape):
    data_shape = torch.Size(data_shape)

    for channel_dim in range(len(data_shape)):
        gran = granularity.PerChannel(channel_dim)
        tile_size = gran.tile_size(data_shape)

        expected_data_shape = [dim for dim in data_shape]
        expected_data_shape[channel_dim] = 1

        assert tile_size == torch.Size(expected_data_shape)
        assert gran.parameter_dimensionality(data_shape) == data_shape[channel_dim]


@pytest.mark.parametrize(
    "data_shape,channels",
    [
        (
            (
                2,
                10,
            ),
            (0, 1),
        ),
        ((2, 10, 30), (0, 1)),
        ((2, 10, 30), (0, 2)),
        ((2, 10, 30), (1, 2)),
    ],
)
def test_per_channel_granularity_multiple_channels(data_shape, channels):
    data_shape = torch.Size(data_shape)

    gran = granularity.PerChannel(channels)
    tile_size = gran.tile_size(data_shape)
    expected_data_shape = [dim for dim in data_shape]

    for channel_dim in channels:
        expected_data_shape[channel_dim] = 1

    assert tile_size == torch.Size(expected_data_shape)
    assert gran.parameter_dimensionality(data_shape) == math.prod([
        data_shape[channel_dim] for channel_dim in channels
    ])


@pytest.mark.parametrize(
    "data_shape,tile_size,passes",
    [
        ((2,), (1,), True),
        (
            (
                2,
                10,
            ),
            (2, 5),
            True,
        ),
        ((2, 10, 30), (1, 5, 10), True),
        ((2,), (3,), False),
        (
            (
                2,
                10,
            ),
            (3, 6),
            False,
        ),
        ((2, 10, 30), (1, 5, 12), False),
    ],
)
def test_per_tile_granularity(data_shape, tile_size, passes):
    data_shape = torch.Size(data_shape)
    gran = granularity.PerTile(tile_size)
    if passes:
        assert gran.tile_size(data_shape) == tile_size
        assert gran.parameter_dimensionality(data_shape) == (
            math.prod(data_shape) // math.prod(tile_size)
        )
    else:
        with pytest.raises(ValueError):
            gran.tile_size(data_shape)
