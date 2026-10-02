# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Places where activations and weights can live.

Activations and weights can live on a device, such as `"cpu"`, `"cuda:0"` or any
`torch.device`, and in storage, which means files in a directory.
"""

from __future__ import annotations

import abc
import dataclasses
import os
import pathlib
import tempfile
import weakref

from typing import Any, TypeAlias

import torch
import torch.utils._pytree as pytree

from fastforward.quantized_tensor import QuantizedTensor


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
        received = tensor.to(device=self.device)
        if received.device.type == "cpu":
            data = received.raw_data if isinstance(received, QuantizedTensor) else received
            # File-based tensors also have "cpu" type, but will have a `filename`.
            # Copy these into CPU mem to remove the mmap dependency.
            if data.untyped_storage().filename is not None:
                received = received.clone()
        return received

    def __repr__(self) -> str:
        return f"DeviceLocation({str(self.device)!r})"


@dataclasses.dataclass(frozen=True)
class DiskLocation(Location):
    """Stores the raw data of a tensor on disk, and keeps its metadata on the CPU.

    A tensor is returned that points to the file on disk. The data is memory-mapped, so it stays
    on disk until used, and the file is deleted when the tensor is no longer referenced.

    Note:
        For a `QuantizedTensor`, the metadata on the CPU includes the quantization parameters.

        The tensor is returned detached, so a store also drops the gradient graph that held it.

    Args:
        directory: Where to write files.
    """

    directory: pathlib.Path

    def receive(self, tensor: torch.Tensor) -> torch.Tensor:  # noqa: D102
        if isinstance(tensor, QuantizedTensor):
            context = tensor.quantization_context.detach_parameters().to("cpu")
            return context.attach(self._store(tensor.raw_data))
        return self._store(tensor)

    def _store(self, tensor: torch.Tensor) -> torch.Tensor:
        """Copy `tensor` into a file in `directory`, and return a tensor that reads that file.

        Args:
            tensor: The tensor to copy. It holds plain data, not quantized data.

        Returns:
            A CPU tensor that holds the same values in a file.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        source = tensor.detach().cpu().contiguous()
        descriptor, name = tempfile.mkstemp(dir=self.directory, suffix=".bin")
        os.close(descriptor)
        # `shared=True` maps the file, so the copy below writes into the file itself.
        mapped = torch.from_file(name, shared=True, size=source.numel(), dtype=source.dtype).view(
            source.shape
        )
        mapped.copy_(source)
        # A weakref on the storage, so the file goes away with the data, and not with this
        # tensor. A caller may keep a view of the data, or give the data to a parameter.
        weakref.finalize(mapped.untyped_storage(), pathlib.Path(name).unlink, True)
        return mapped

    def __repr__(self) -> str:
        return f"DiskLocation({str(self.directory)!r})"


DeviceLike: TypeAlias = torch.device | str
LocationLike: TypeAlias = Location | DeviceLike | os.PathLike[str]


def as_location(value: LocationLike) -> Location:
    """Read `value` as a place where data can rest.

    A `Location` is already such a place and comes back as it is. A `pathlib.Path`, or
    anything else that names a file path, becomes a `DiskLocation`. A `torch.device`, or a
    device name such as `"cpu"` or `"cuda:0"`, becomes a `DeviceLocation`.

    Args:
        value: A location, a directory, a device, or the name of a device.

    Returns:
        The place `value` names.

    Raises:
        ValueError: If a string does not name a device.
    """
    match value:
        case Location():
            return value
        case os.PathLike():
            return DiskLocation(pathlib.Path(value))
        case torch.device():
            return DeviceLocation(value)
    try:
        device = torch.device(value)
    except (RuntimeError, ValueError) as error:
        hint = "Give a device name such as 'cpu', or a pathlib.Path to rest data in files."
        msg = f"{value!r} does not name a device. {hint}"
        raise ValueError(msg) from error
    return DeviceLocation(device)
