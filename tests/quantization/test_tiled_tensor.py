# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import pytest
import torch

from fastforward.quantization import tiled_tensor


def test_tiles_to_rows_and_rows_to_tiles() -> None:
    a1 = torch.ones(2, 2)
    a2 = torch.ones(2, 2) * 2
    a3 = torch.ones(2, 2) * 3
    a4 = torch.ones(2, 2) * 4

    a5 = torch.ones(2, 2) * 5
    a6 = torch.ones(2, 2) * 6
    a7 = torch.ones(2, 2) * 7
    a8 = torch.ones(2, 2) * 8

    partial1 = torch.stack([torch.hstack([a1, a2]), torch.hstack([a3, a4])]).reshape(4, 4)
    partial2 = torch.stack([torch.hstack([a5, a6]), torch.hstack([a7, a8])]).reshape(4, 4)
    input = torch.stack([partial1, partial2])

    tile_size = (1, 2, 2)
    row_representation = tiled_tensor.tiles_to_rows(input, tile_size)
    expected = torch.ones(8, 4) * torch.arange(1, 9)[:, None]
    torch.testing.assert_close(row_representation, expected)

    tiled_representation = tiled_tensor.rows_to_tiles(row_representation, input.shape, tile_size)
    torch.testing.assert_close(tiled_representation, input)

    with pytest.raises(ValueError):
        tiled_tensor.tiles_to_rows(input, (1, 2, 3, 4))

    with pytest.raises(ValueError):
        tiled_tensor.tiles_to_rows(input, (3, 2, 2))

    with pytest.raises(ValueError):
        tiled_tensor.rows_to_tiles(row_representation[:, :-1], input.shape, tile_size)

    with pytest.raises(ValueError):
        tiled_tensor.rows_to_tiles(row_representation[:, :-1], input.shape, (2, 2, 2))


@pytest.mark.parametrize(
    "seq1,seq2,expected",
    [
        ([1, 2], [3, 4], [1, 3, 2, 4]),
        ([1, 2, 3], 0, [1, 0, 2, 0, 3, 0]),
        ([1], [3, 4], [1, 3, 1, 4]),
        ([1, 2, 3, 4], [5, 6], [1, 5, 2, 6, 3, 5, 4, 6]),
        ([], [], []),
    ],
)
def test_interleave(seq1: list[int], seq2: list[int] | int, expected: list[int]) -> None:
    # GIVEN two sequences (or a sequence and a scalar)
    # WHEN they are interleaved
    actual = list(tiled_tensor._interleave(seq1, seq2))

    # THEN the elements alternate, the second sequence repeating cyclically
    assert actual == expected
