# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Places where activations and weights can live.

Activations and weights can live on a device, such as `"cpu"`, `"cuda:0"` or any
`torch.device`.
"""

from __future__ import annotations

import abc
import dataclasses

from typing import Any, TypeAlias

import torch
import torch.utils._pytree as pytree


class Location(abc.ABC):
    """A place where data can be held."""

    @abc.abstractmethod
    def receive(self, tensor: torch.Tensor) -> Any:
        """Bring `tensor` to this location.

        Args:
            tensor: The tensor to place here.

        Returns:
            The value that stands for `tensor` at this location. For a location
            backed by a device this is a tensor on that device.
        """

    def place(self, value: Any) -> Any:
        """Bring every tensor in `value` to this location.

        Args:
            value: A tensor, or a structure that holds tensors, such as a tuple, a list, a
                dict, a named tuple, or a model output. Every container that torch knows
                keeps its type. A container that torch does not know counts as a leaf and
                comes back as it is.

        Returns:
            The value with every tensor placed here. Other leaves are returned as they are.
        """
        return pytree.tree_map_only(torch.Tensor, self.receive, value)


@dataclasses.dataclass(frozen=True)
class DeviceLocation(Location):
    """A `torch.device` that holds data directly.

    Args:
        device: The device data is held on.
    """

    device: torch.device

    def receive(self, tensor: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return tensor.to(device=self.device)

    def __repr__(self) -> str:
        return f"DeviceLocation({str(self.device)!r})"


DeviceLike: TypeAlias = torch.device | str
LocationLike: TypeAlias = Location | DeviceLike


def as_location(value: LocationLike) -> Location:
    """Read `value` as a place where data can rest.

    A `Location` is already such a place and comes back as it is. A `torch.device`, or a
    device name such as `"cpu"` or `"cuda:0"`, becomes a `DeviceLocation`.

    Args:
        value: A location, a device, or the name of a device.

    Returns:
        The place `value` names.

    Raises:
        ValueError: If a string does not name a device.
    """
    match value:
        case Location():
            return value
        case torch.device():
            return DeviceLocation(value)
    try:
        device = torch.device(value)
    except (RuntimeError, ValueError) as error:
        msg = f"{value!r} does not name a device. Give a device name such as 'cpu' or 'cuda:0'."
        raise ValueError(msg) from error
    return DeviceLocation(device)
