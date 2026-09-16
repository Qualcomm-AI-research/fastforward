# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import collections
import gc

from pathlib import Path

import pytest
import torch

from fastforward._orchestration.location import (
    DeviceLocation,
    DiskLocation,
    Location,
    as_location,
)
from fastforward.quantization.random import random_quantized
from fastforward.quantized_tensor import QuantizedTensor


class _Tagging(Location):
    """A location that adds one to every tensor it receives, so a test sees the visit."""

    def receive(self, tensor: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return tensor + 1


def test_a_device_name_reads_as_that_device() -> None:
    # GIVEN the name of a device
    # WHEN we read it as a location
    # THEN we get that device, so a caller never has to name a location itself
    assert as_location("cpu") == DeviceLocation(torch.device("cpu"))
    assert as_location(torch.device("cuda:0")) == DeviceLocation(torch.device("cuda:0"))


def test_a_path_reads_as_a_directory_that_holds_files(tmp_path: Path) -> None:
    # GIVEN a path
    # WHEN we read it as a location
    # THEN we get a place that stores tensors in files in that directory
    assert as_location(tmp_path) == DiskLocation(tmp_path)


def test_a_location_reads_as_itself() -> None:
    # GIVEN a location that a caller made
    location = DeviceLocation(torch.device("cpu"))

    # WHEN we read it
    # THEN it comes back as it is
    assert as_location(location) is location


def test_a_name_that_is_no_device_is_rejected() -> None:
    # GIVEN a string that looks like a device name, but names no device
    # WHEN we read it as a location
    # THEN it is refused, because a string never names anything but a device
    with pytest.raises(ValueError, match="does not name a device"):
        as_location("gpu0")


def test_device_locations_for_the_same_device_are_interchangeable() -> None:
    # GIVEN two locations that name the same device
    left = DeviceLocation(torch.device("cpu"))
    right = DeviceLocation(torch.device("cpu"))

    # THEN they are equal and hash alike, so the offloading passes can use them as keys
    assert left == right
    assert {left, right} == {left}


def test_device_location_receives_a_tensor_on_its_device() -> None:
    # GIVEN a location that names the CPU
    location = DeviceLocation(torch.device("cpu"))

    # WHEN it receives a tensor
    received = location.receive(torch.randn(2, 3))

    # THEN the tensor is on that device
    assert received.device == torch.device("cpu")


def test_place_reaches_tensors_inside_nested_containers() -> None:
    # GIVEN a value that mixes nested tuples and lists of tensors with a non-tensor leaf
    value = ([torch.randn(2, 3), (torch.randn(4), "scalar")], torch.randn(1))

    # WHEN we place it
    placed = DeviceLocation(torch.device("cpu")).place(value)

    # THEN the containers keep their shape and every tensor is on the device
    inner_list, outer_tensor = placed
    assert isinstance(inner_list, list)
    assert inner_list[0].device == torch.device("cpu")
    inner_tuple = inner_list[1]
    assert isinstance(inner_tuple, tuple)
    assert inner_tuple[0].device == torch.device("cpu")
    assert outer_tensor.device == torch.device("cpu")

    # THEN the non-tensor leaf passes through untouched
    assert inner_tuple[1] == "scalar"


def test_place_returns_non_tensor_leaves_unchanged() -> None:
    # GIVEN a value with leaves that are not tensors
    location = DeviceLocation(torch.device("cpu"))

    # WHEN we place it
    placed = location.place({"n": 3, "s": "text", "none": None})

    # THEN those leaves come back as they were
    assert placed == {"n": 3, "s": "text", "none": None}


def test_place_keeps_the_type_of_a_named_tuple() -> None:
    # GIVEN a named tuple of tensors, as `torch.max` and many model outputs return
    values, indices = torch.max(torch.randn(3, 3), dim=0)

    # WHEN we place it
    placed = _Tagging().place(torch.return_types.max((values, indices)))

    # THEN the fields survive, and both tensors were placed
    assert bool(placed.values.eq(values + 1).all())
    assert bool(placed.indices.eq(indices + 1).all())


def test_place_reaches_tensors_in_a_deque() -> None:
    # GIVEN a deque of tensors
    value = collections.deque([torch.randn(2, 3)])

    # WHEN we place it
    placed = _Tagging().place(value)

    # THEN the deque survives and its tensor was placed
    assert type(placed) is collections.deque
    assert bool(placed[0].eq(value[0] + 1).all())


def test_place_leaves_a_container_that_torch_does_not_know_alone() -> None:
    # GIVEN a home-made dict subclass, which torch has no rule for
    class Bespoke(dict[str, object]):
        pass

    value = Bespoke(hidden=torch.randn(2, 3))

    # WHEN we place it
    placed = _Tagging().place(value)

    # THEN it counts as one leaf and comes back as it is, tensor and all
    assert placed is value


def test_place_keeps_a_plain_dict_plain() -> None:
    # GIVEN a plain dict of tensors
    value = {"a": torch.randn(2, 3), "b": torch.randn(4)}

    # WHEN we place it
    placed = DeviceLocation(torch.device("cpu")).place(value)

    # THEN it stays a plain dict with the same contents
    assert type(placed) is dict
    assert placed["a"].shape == (2, 3)
    assert placed["b"].shape == (4,)


def test_disk_locations_for_the_same_directory_are_interchangeable(tmp_path: Path) -> None:
    # GIVEN two locations that name the same directory
    left = DiskLocation(tmp_path)
    right = DiskLocation(tmp_path)

    # THEN they are equal and hash alike, so the offloading passes can use them as keys
    assert left == right
    assert {left, right} == {left}


def test_a_stored_tensor_reads_back_as_it_was(tmp_path: Path) -> None:
    # GIVEN a tensor and a location that stores tensors in files
    original = torch.randn(4, 8)

    # WHEN the location receives it
    stored = DiskLocation(tmp_path).receive(original)

    # THEN an ordinary CPU tensor comes back that holds the same values
    assert isinstance(stored, torch.Tensor)
    assert stored.device == torch.device("cpu")
    assert torch.equal(stored, original)


def test_every_stored_tensor_keeps_a_file_of_its_own(tmp_path: Path) -> None:
    # GIVEN a location that stores tensors in files
    location = DiskLocation(tmp_path)
    originals = [torch.randn(64, 64) for _ in range(4)]

    # WHEN it receives them
    stored = [location.receive(original) for original in originals]

    # THEN the directory names one file per tensor, so a run shows what it holds
    assert len(list(tmp_path.iterdir())) == len(originals)

    # AND every tensor still reads the values it was given
    assert all(torch.equal(*pair) for pair in zip(stored, originals))


def test_a_file_goes_away_with_the_tensor_that_reads_it(tmp_path: Path) -> None:
    # GIVEN tensors that a location stored in files
    location = DiskLocation(tmp_path)
    stored = [location.receive(torch.randn(64, 64)) for _ in range(4)]

    # WHEN the last tensor that reads a file goes away
    del stored
    gc.collect()

    # THEN the file goes with it, so a run that ends leaves nothing behind
    assert list(tmp_path.iterdir()) == []


def test_a_file_lives_as_long_as_the_data_and_not_as_long_as_one_tensor(tmp_path: Path) -> None:
    # GIVEN a parameter that takes the data of a stored tensor, the way a weight move does
    parameter = torch.nn.Parameter(torch.randn(4, 8))
    parameter.data = DiskLocation(tmp_path).receive(parameter.data)

    # WHEN the tensor that the store returned goes away, and only the data is held
    gc.collect()

    # THEN the file is still there, because the data still reads it
    assert len(list(tmp_path.iterdir())) == 1
    assert parameter.is_shared()


def test_a_store_creates_its_directory(tmp_path: Path) -> None:
    # GIVEN a directory that does not exist yet
    directory = tmp_path / "absent" / "deeper"

    # WHEN a location stores a tensor in it
    DiskLocation(directory).receive(torch.randn(4))

    # THEN the directory was made along the way
    assert directory.is_dir()


def test_a_stored_tensor_carries_no_gradient_history(tmp_path: Path) -> None:
    # GIVEN a tensor that needs gradients
    original = torch.randn(4, requires_grad=True)

    # WHEN it is stored
    stored = DiskLocation(tmp_path).receive(original)

    # THEN it comes back without them, because a file cannot hold a gradient graph
    assert not stored.requires_grad


def test_place_stores_tensors_inside_containers(tmp_path: Path) -> None:
    # GIVEN a nested value and a location that stores tensors in files
    value = {"hidden": [torch.randn(2, 3)], "count": 7}

    # WHEN we place it
    placed = DiskLocation(tmp_path).place(value)

    # THEN the tensor is stored in a file, and the other leaf is untouched
    assert torch.equal(placed["hidden"][0], value["hidden"][0])  # type: ignore[index]
    assert placed["count"] == 7
    assert len(list(tmp_path.iterdir())) == 1


def test_a_store_writes_the_bytes_of_a_view_only(tmp_path: Path) -> None:
    # GIVEN one row of a large tensor, which still holds the storage of the whole tensor
    row = torch.randn(1024, 1024)[:1]

    # WHEN we store it
    stored = DiskLocation(tmp_path).receive(row)

    # THEN the file holds the bytes of the row only, not those of the whole tensor
    assert torch.equal(stored, row)
    assert stored.untyped_storage().nbytes() == row.numel() * row.element_size()


def test_a_stored_tensor_keeps_its_values_in_a_file(tmp_path: Path) -> None:
    # GIVEN a location that stores tensors in files
    # WHEN it receives a tensor
    stored = DiskLocation(tmp_path).receive(torch.randn(4, 8))

    # THEN the tensor reads from the file, so it holds no memory of its own
    assert stored.is_shared()
    assert stored.untyped_storage().filename is not None


def test_a_stored_quantized_tensor_keeps_its_quantization(tmp_path: Path) -> None:
    # GIVEN a quantized tensor
    original = random_quantized((4, 8))

    # WHEN it is stored
    stored = DiskLocation(tmp_path).receive(original)

    # THEN a quantized tensor comes back that dequantizes to the same values
    assert isinstance(stored, QuantizedTensor)
    assert torch.equal(stored.dequantize(), original.dequantize())

    # AND its raw data is the part that went to the file
    assert stored.raw_data.is_shared()
