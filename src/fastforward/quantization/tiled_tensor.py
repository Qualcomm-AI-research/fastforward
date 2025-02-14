# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import itertools

from typing import (
    Iterable,
    Literal,
    Sequence,
    TypeVar,
    overload,
)

import torch

from fastforward.type_common import SizeT


def check_tile_compatibility(input_size: SizeT, tile_size: SizeT) -> None:
    """Helper function to assess whether a given tile size can be used with a given data shape.

    The function will raise an error if not, otherwise return None.
    """
    if len(input_size) != len(tile_size):
        raise ValueError(
            f"Input dimensionality must match tile_size dimensionality got "
            f"{len(input_size)} and {len(tile_size)}"
        )

    mismatched = []
    for i, (input_dim, tile_dim) in enumerate(zip(input_size, tile_size)):
        if tile_dim > 0 and input_dim % tile_dim != 0:
            mismatched.append(i)

    if mismatched:
        errors = [f"{input_size[i]} and {tile_size[i]} for dimension {i}" for i in mismatched]
        raise ValueError(
            "Each dimension of tile_size must divide the corresponding input dimension. Got "
            + ", ".join(errors)
            + "."
        )


T = TypeVar("T")


@overload
def _interleave(seq1: Sequence[T], seq2: Sequence[T]) -> Iterable[T]: ...


@overload
def _interleave(seq1: Sequence[T], seq2: T) -> Iterable[T]: ...


def _interleave(seq1: Sequence[T], seq2: Sequence[T] | T) -> Iterable[T]:
    left = list(seq1)
    if not isinstance(seq2, Sequence):
        right = [seq2]
    else:
        right = list(seq2)

    if len(left) == 1 and len(right) > 1:
        left = left * len(right)

    for a, b in zip(left, itertools.cycle(right)):
        yield a
        yield b


def tiles_to_rows(data: torch.Tensor, tile_size: SizeT | Literal["data_shape"]) -> torch.Tensor:
    """Reshape and permute data.

    Reshape and permute data to a tensor in which the elements per tile are
    laid out per row, following tile_size.

    Args:
    ----
        data: Data to reshape and permute
        tile_size: Tile size to use for row collection

    Returns:
    -------
        Tensor: The reshaped tensor.
    """
    if data.numel() == 0:
        return data.reshape(1, 0)

    tile_size = data.shape if tile_size == "data_shape" else torch.Size(tile_size)
    check_tile_compatibility(data.size(), tile_size)
    num_dim_blocks = [a // b for a, b in zip(data.size(), tile_size)]

    data_shape = torch.Size(_interleave(num_dim_blocks, tile_size))
    num_blocks = data.numel() // tile_size.numel()

    tiled_data = data.reshape(data_shape)
    permutation = list(range(0, tiled_data.ndim, 2)) + list(range(1, tiled_data.ndim, 2))
    return tiled_data.permute(permutation).reshape(num_blocks, -1)


def rows_to_tiles(
    tiled_data: torch.Tensor, data_size: SizeT, tile_size: SizeT | Literal["data_shape"]
) -> torch.Tensor:
    """Reshape and permute `tiled_data`.

    Reshape and permute `tiled_data` to a tensor of `data_size` tiled by `tile_size` where each row
    in tiled_data corresponds to a single tile.

    Args:
    ----
        tiled_data: Data to reshape and permute
        data_size: Size of the output
        tile_size: Tile size to use for row collection

    Returns:
    -------
        Tensor: The reshaped tensor.

    Raises:
    ------
        ValueError: `tiled_data`'s size does not correspond to `data_size` and `tile_size`.
    """
    if tiled_data.numel() == 0:
        return tiled_data.reshape(data_size)

    tile_size_to_use = torch.Size(data_size) if tile_size == "data_shape" else torch.Size(tile_size)
    data_size = torch.Size(data_size)
    check_tile_compatibility(data_size, tile_size_to_use)

    expected_rows = data_size.numel() // tile_size_to_use.numel()
    expected_size = torch.Size((expected_rows, tile_size_to_use.numel()))
    if tiled_data.size() != expected_size:
        raise ValueError(
            f"tiled_data is expected to be of size {expected_size} but found {tiled_data.size()}"
        )

    num_dim_blocks = [a // b for a, b in zip(data_size, tile_size_to_use)]
    tiled_intermediate_shape = torch.Size(_interleave(num_dim_blocks, tile_size_to_use))

    ndims = len(tiled_intermediate_shape)
    permutation = list(range(0, ndims, 2)) + list(range(1, ndims, 2))
    tiled_shape = [tiled_intermediate_shape[i] for i in permutation]
    reverse_permutation = list(_interleave(range(0, ndims // 2), range(ndims // 2, ndims)))
    data = tiled_data.reshape(tiled_shape).permute(reverse_permutation)
    return data.reshape(data_size)
