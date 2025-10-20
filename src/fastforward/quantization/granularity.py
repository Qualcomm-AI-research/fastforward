# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import abc
import logging

from typing import Any, Literal, Sequence

import torch

from typing_extensions import override

from fastforward.quantization.tiled_tensor import check_tile_compatibility
from fastforward.serialization import yamlable

logger = logging.getLogger(__name__)


@yamlable
class Granularity(abc.ABC):
    """Granularity represents how parameters are shared during quantization.

    Granularities provide an abstraction used for element-wise operations with
    shared parameters. These are most prominently used in quantizers. For
    example, when using per-channel quantization, the same quantization
    parameters are used for each element in the input tensor that are in the
    same channel.

    Because FastForward performs quantization based on input tiles, granularities
    are just used to provide the user with a simple class-based approach of retrieving
    the appropriate tile sizes for performing per tensor and per channel quantization.

    The usage for the granularities subclass is as follows:

        > data = torch.randn(2, 4, 6)
        > gr = some_granularity(...)
        > tile_size = gr.tile_size(data.shape)

    The found tile size can then be used as input for the `quantize_by_tile` and `quantize_by_tile_function`
    methods from `fastforward.quantization.affine` like so:

        > scale = 1.4
        > offset = 0.
        > num_bits = 3
        > quantize_by_tile(scale, offset, tile_size, num_bits)

    A more involved usage of the Granularity class can be found in the `LinearQuantizer` class in
    FastForward (`fastforward.nn.linear_quantizer`).

    """

    @abc.abstractmethod
    def tile_size(self, data_shape: torch.Size) -> torch.Size | Literal["data_shape"]:
        """Retrieve the tile size that will be applied over the input data.

        The tile functions (fastforward.quantization.affine) broadcast the
        dimensions with a size of 1 over the data shape. For example, given an
        input data shape of (2, 4, 6) and the requested granularity is per
        channel quantization over dimension zero, the tile size will be (1, 4,
        6) setting data_shape[0] = 1.
        """
        raise NotImplementedError

    def parameter_dimensionality(self, data_shape: torch.Size) -> int:
        """The dimensionality of the quantizer parameters  to process the input data.

        This is computed by dividing the number of elements of the original
        data shape over the number of elements of the tile size. For example,
        for PerTensor quantization the tile size is the same as the data shape,
        applying parameters with a dimensionality of 1.
        """
        tile_size = self.tile_size(data_shape)
        if isinstance(tile_size, str):
            return 1
        return data_shape.numel() // tile_size.numel()

    def repr_args(self) -> dict[str, Any]:
        """Return a dictionary of arguments for the __repr__ method.

        Returns:
            dict[str, Any]: A dictionary of arguments.
        """
        return {}

    @override
    def __repr__(self) -> str:
        args_str = ", ".join([f"{k}={v}" for k, v in self.repr_args().items()])
        return f"{type(self).__name__}({args_str})"

    @override
    def __eq__(self, other: object) -> bool:
        # If the types don't match, there is no equality
        if type(self) is not type(other):
            return False
        args = getattr(type(self), "__match_args__", ())
        for key in args:
            if getattr(self, key) != getattr(other, key):
                return False
        return True


class PerTensor(Granularity):
    """Granularity class for per-tensor quantization."""

    def __init__(self) -> None:
        super().__init__()

    @override
    def tile_size(self, data_shape: torch.Size) -> Literal["data_shape"]:
        """Return the tile size for per-tensor quantization.

        Args:
            data_shape: The shape of the input data.

        Returns:
            The tile size, which is the same as the data shape.
        """
        return "data_shape"


class PerChannel(Granularity):
    """Granularity class for per-channel quantization.

    Attributes:
        channel_dims: The dimensions to apply per-channel quantization.
    """

    __match_args__ = ("channel_dims",)

    def __init__(self, channel_dim: int | tuple[int, ...] = 0) -> None:
        if isinstance(channel_dim, int):
            channel_dim = (channel_dim,)
        self.channel_dims = channel_dim
        super().__init__()

    @override
    def tile_size(self, data_shape: torch.Size) -> torch.Size:
        """Return the tile size for per-channel quantization.

        Args:
            data_shape: The shape of the input data.

        Returns:
            The tile size for per-channel quantization.
        """
        tile_size = list(data_shape)
        for dim in self.channel_dims:
            tile_size[dim] = 1

        return torch.Size(tile_size)

    @override
    def repr_args(self) -> dict[str, Any]:
        """Return a dictionary of arguments for the __repr__ method."""
        dim = self.channel_dims[0] if len(self.channel_dims) == 1 else self.channel_dims
        return {"channel": dim}


class PerBlock(Granularity):
    """Granularity class for per-block quantization.

    Attributes:
        block_dims: The dimensions to quantize per-block
        block_sizes: The block sizes corresponding to `block_dims`. The length
            of block_dims and block_sizes must match.
        per_channel_dims: The dimensions to quantize per-channel
        strict_blocks: If true, `block_sizes` must divide `data_size` at
            corresponding dimensions.
    """

    def __init__(
        self,
        block_dims: int | Sequence[int],
        block_sizes: int | Sequence[int],
        per_channel_dims: int | Sequence[int] = (),
        strict_blocks: bool = True,
    ):
        self.block_dims = _as_tuple(block_dims)
        self.block_sizes = _as_tuple(block_sizes)
        self.per_channel_dims = _as_tuple(per_channel_dims)
        self.strict_blocks = strict_blocks

        if len(self.block_dims) != len(self.block_sizes):
            msg = "block_sizes and block_dims must be of equal length"
            raise ValueError(msg)

        duplicate_dims = [str(dim) for dim in self.per_channel_dims if dim in self.block_dims]
        if duplicate_dims:
            msg = (
                f"Dimensions {', '.join(duplicate_dims)} are in both 'block_dims' and "
                "'per_channel_dims'. They will be quantized as per-block following 'block_sizes'"
            )
            logger.warning(msg)

    @override
    def tile_size(self, data_shape: torch.Size) -> torch.Size:
        tile_size = list(data_shape)
        for dim in self.per_channel_dims:
            tile_size[dim] = 1

        for block_dim, block_size in zip(self.block_dims, self.block_sizes):
            if block_size > data_shape[block_dim]:
                msg = (
                    f"Can't apply per block quantization using block-size={block_size}"
                    f" over dimension {block_dim} for a tensor with shape {data_shape}. "
                )
                raise ValueError(msg)
            if self.strict_blocks and data_shape[block_dim] % block_size != 0:
                msg = (
                    f"Block dim {block_dim} of size {block_size} does not divide the data dim "
                    f"{data_shape[block_dim]} exactly. This is required because strict_blocks=True"
                )
                raise ValueError(msg)
            tile_size[block_dim] = block_size

        return torch.Size(tile_size)

    @override
    def repr_args(self) -> dict[str, Any]:
        return {
            "block_dims": self.block_dims,
            "block_sizes": self.block_sizes,
            "per_channel_dims": self.per_channel_dims,
            "strict_blocks": self.strict_blocks,
        }


class PerTile(Granularity):
    """Granularity class for per-tile quantization.

    Attributes:
        tile_shape: The shape of the tile.
    """

    __match_args__ = ("tile_shape",)

    def __init__(self, tile_shape: tuple[int, ...]):
        super().__init__()
        self.tile_shape = torch.Size(tile_shape)

    @override
    def tile_size(self, data_shape: torch.Size) -> torch.Size:
        """Return the tile size for per-tile quantization.

        Args:
            data_shape: The shape of the input data.

        Returns:
            The tile size for per-tile quantization.

        Raises:
            ValueError: If the tile shape is not compatible with the data shape.
        """
        try:
            check_tile_compatibility(data_shape, self.tile_shape)
            return self.tile_shape
        except ValueError as e:
            raise e

    @override
    def repr_args(self) -> dict[str, Any]:
        """Return a dictionary of arguments for the __repr__ method."""
        return {"tile_shape": self.tile_shape}


def is_per_tensor(granularity: Granularity) -> bool:
    """Check if the granularity is per-tensor.

    Args:
        granularity: The granularity instance.

    Returns:
        bool: True if the granularity is per-tensor, False otherwise.
    """
    return isinstance(granularity, PerTensor)


def is_per_channel(granularity: Granularity) -> bool:
    """Check if the granularity is per-channel.

    Args:
        granularity: The granularity instance.

    Returns:
        bool: True if the granularity is per-channel, False otherwise.
    """
    return isinstance(granularity, PerChannel)


def is_per_block(granularity: Granularity) -> bool:
    """Check if the granularity is per-block.

    Args:
        granularity: The granularity instance.

    Returns:
        bool: True if the granularity is per-block, False otherwise.
    """
    return isinstance(granularity, PerBlock)


def _as_tuple(value: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,)
    return tuple(value)


def granularity_from_sizes(data_size: torch.Size, tile_size: torch.Size) -> Granularity:
    """Infer granularity from `data_size` and `tile_size`.

    Return granularity such that `granularity.tile_size(data_size) == tile_size`.

    Note:
        Tilings can be represented using multiple granularities. For example,
        `PerTensor()` represents the same tiling as `PerChannel(())`. This function
        returns the 'simplest' option. I.e., `PerTensor()` in the previous example.

    """
    if data_size == tile_size:
        return PerTensor()

    divs = torch.tensor(data_size) / torch.tensor(tile_size)
    dims = list(range(len(data_size)))
    if all(div == 1 or div == data_dim for div, data_dim in zip(divs, data_size)):
        indices = tuple([i for i in dims if tile_size[i] == 1 and data_size[i] > 1])
        return PerChannel(indices)

    block_dims = tuple([i for i in dims if tile_size[i] not in (1, data_size[i])])
    block_sizes = tuple([tile_size[i] for i in block_dims])
    per_channel_dims = tuple([i for i in dims if tile_size[i] == 1 and data_size[i] > 1])
    strict_blocks = all(data_dim % div == 0 for div, data_dim in zip(divs, data_size))
    return PerBlock(block_dims, block_sizes, per_channel_dims, strict_blocks=strict_blocks)
