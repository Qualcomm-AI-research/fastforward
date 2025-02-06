# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import abc

from typing import Any, Literal

import torch

from typing_extensions import override

from fastforward.quantization.tiled_tensor import check_tile_compatibility


class Granularity(abc.ABC):
    """Granularity represents how paraameters are shared during quantization.

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

        The tile functions (fastforward.quantization.affine) brodcast the
        dimensions with a size of 1 over the data shape. For example, given an
        input data shape of (2, 4, 6) and the requested granularity is per
        channel quantization over dimension zero, the tile size will be (1, 4,
        6) setting data_shape[0] = 1.
        """
        raise NotImplementedError

    def parameter_dimensionality(self, data_shape: torch.Size) -> int:
        """The dimensionality of the quantizer parameters  to porcess the input data.

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
