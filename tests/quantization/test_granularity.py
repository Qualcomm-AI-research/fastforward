# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import math

from collections.abc import Sequence

import fastforward as ff
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
def test_per_tensor_granularity(data_shape: torch.Size | Sequence[int]) -> None:
    data_shape = torch.Size(data_shape)
    gran = granularity.PerTensor()
    tile_size = gran.tile_size(data_shape)

    assert tile_size == "data_shape"
    assert gran.parameter_dimensionality(data_shape) == 1


@pytest.mark.parametrize(
    "data_shape",
    [
        (2,),
        (2, 10),
        (2, 10, 30),
    ],
)
def test_per_channel_granularity_single_channel(data_shape: Sequence[int]) -> None:
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
        ((2, 10), (0, 1)),
        ((2, 10, 30), (0, 1)),
        ((2, 10, 30), (0, 2)),
        ((2, 10, 30), (1, 2)),
    ],
)
def test_per_channel_granularity_multiple_channels(
    data_shape: Sequence[int], channels: tuple[int, ...]
) -> None:
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
    "data_shape,block_dims,block_sizes,per_channel_dims,expected_tile_size,strict_blocks,passes",
    [
        # Basic per-block quantization
        ([16, 32], [0], [4], [], [4, 32], True, True),
        ([16, 32], [1], [8], [], [16, 8], True, True),
        ([64, 128], [0, 1], [8, 16], [], [8, 16], True, True),
        # Per-channel quantization
        ([16, 32], [], [], [0], [1, 32], True, True),
        ([16, 32], [], [], [1], [16, 1], True, True),
        ([16, 32, 64], [], [], [0, 2], [1, 32, 1], True, True),
        # Mixed per-block and per-channel
        ([16, 32, 64], [1], [8], [0], [1, 8, 64], True, True),
        ([32, 64, 128], [0, 2], [4, 16], [1], [4, 1, 16], True, True),
        # Non-strict blocks (strict_blocks=False)
        ([15, 32], [0], [4], [], [4, 32], False, True),
        ([16, 30], [1], [8], [], [16, 8], False, True),
        ([17, 33], [0, 1], [4, 8], [], [4, 8], False, True),
        # Single dimension tensors
        ([64], [0], [8], [], [8], True, True),
        ([64], [], [], [0], [1], True, True),
        # Larger tensors
        ([128, 256, 512], [0, 1], [16, 32], [2], [16, 32, 1], True, True),
        ([100, 200, 300], [1], [25], [0, 2], [1, 25, 1], True, True),
        # Edge case: block size equals dimension size
        ([16, 32], [0], [16], [], [16, 32], True, True),
        ([16, 32], [1], [32], [], [16, 32], True, True),
        # Multiple blocks with different sizes
        ([24, 48, 72], [0, 1, 2], [6, 12, 18], [], [6, 12, 18], True, True),
        # FAILURE CASES - Block size larger than dimension size
        ([16, 32], [0], [20], [], [], True, False),  # block_size 20 > dim 16
        # Multiple block dims where at least one block size is too large
        ([16, 32], [0, 1], [8, 40], [], [], True, False),  # second block_size 40 > dim 32
        ([16, 32], [0, 1], [20, 16], [], [], True, False),  # first block_size 20 > dim 16
        # FAILURE CASES - Non-divisible dimensions with strict_blocks=True
        ([15, 32], [0], [4], [], [], True, False),  # 15 % 4 != 0
        ([15, 30], [0, 1], [4, 7], [], [], True, False),  # 15 % 4 != 0 and 30 % 7 != 0
        # Edge cases - block size of 1 larger than dimension
        ([0], [0], [1], [], [], True, False),  # empty dimension
        ([1], [0], [2], [], [], True, False),  # block_size 2 > dim 1
        ([2, 3], [0], [3], [], [], True, False),  # block_size 3 > dim 2
    ],
)
def test_per_block_granularity(
    data_shape: Sequence[int],
    block_dims: Sequence[int],
    block_sizes: Sequence[int],
    per_channel_dims: Sequence[int],
    expected_tile_size: Sequence[int],
    strict_blocks: bool,
    passes: bool,
) -> None:
    # GIVEN data_shape and a per block granularity
    data_shape = torch.Size(data_shape)
    gran = granularity.PerBlock(block_dims, block_sizes, per_channel_dims, strict_blocks)
    # WHEN the parametrization is valid
    if passes:
        # THEN tile size and parameter_per_dimensionality mutch match expectations
        assert gran.tile_size(data_shape) == torch.Size(expected_tile_size)
        assert gran.parameter_dimensionality(data_shape) == (
            math.prod(data_shape) // math.prod(expected_tile_size)
        )
    # WHEN the parametrization is not valid
    else:
        # THEN tile_size must raise an error
        with pytest.raises(ValueError):
            gran.tile_size(data_shape)


@pytest.mark.parametrize(
    "data_shape,tile_size,passes",
    [
        ((2,), (1,), True),
        ((2, 10), (2, 5), True),
        ((2, 10, 30), (1, 5, 10), True),
        ((2,), (3,), False),
        ((2, 10), (3, 6), False),
        ((2, 10, 30), (1, 5, 12), False),
    ],
)
def test_per_tile_granularity(
    data_shape: Sequence[int], tile_size: tuple[int, ...], passes: bool
) -> None:
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


@pytest.mark.parametrize(
    "gran,expected,data_size",
    [
        # If no expected is given, reproduced granularity equals the original
        (ff.PerTensor(), None, torch.Size([16, 16])),
        (ff.PerChannel(0), None, torch.Size([16, 16])),
        (ff.PerChannel(1), None, torch.Size([16, 16])),
        (ff.PerChannel((0, 1)), None, torch.Size([16, 16])),
        (ff.PerChannel((1, 0)), ff.PerChannel((0, 1)), torch.Size([16, 16])),
        (ff.PerChannel(()), ff.PerTensor(), torch.Size([16, 16])),
        (ff.PerBlock(block_dims=(), block_sizes=()), ff.PerTensor(), torch.Size([16, 16])),
        (ff.PerBlock(block_dims=(0), block_sizes=(4)), None, torch.Size([16, 16])),
        (ff.PerBlock(block_dims=(1), block_sizes=(2)), None, torch.Size([16, 16])),
        (ff.PerBlock(block_dims=(0, 1), block_sizes=(4, 4)), None, torch.Size([16, 16])),
        (
            ff.PerBlock(block_dims=(1), block_sizes=(4), per_channel_dims=(0,)),
            None,
            torch.Size([16, 16]),
        ),
        (
            ff.PerBlock(
                block_dims=(0), block_sizes=(5), per_channel_dims=(1,), strict_blocks=False
            ),
            None,
            torch.Size([16, 16]),
        ),
        (ff.PerChannel((0, 2, 3)), ff.PerChannel((0,)), torch.Size([16, 16, 1, 1])),
        (ff.PerChannel((1, 2)), ff.PerTensor(), torch.Size([16, 1, 1])),
        (ff.PerChannel((0, 1, 2)), ff.PerChannel((0, 1)), torch.Size([32, 64, 1])),
    ],
)
def test_granularity_from_sizes(
    gran: granularity.Granularity, expected: granularity.Granularity | None, data_size: torch.Size
) -> None:
    # GIVEN a granularity
    expected = expected or gran
    gran_tile_size = gran.tile_size(data_size)
    if gran_tile_size == "data_shape":
        gran_tile_size = data_size

    # WHEN the granularity is reproduced using tile and data size
    repro_granularity = granularity.granularity_from_sizes(data_size, gran_tile_size)

    # THEN the granularity must match the expected granularity
    assert repro_granularity == expected
