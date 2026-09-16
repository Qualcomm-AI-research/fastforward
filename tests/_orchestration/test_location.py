# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import collections

import pytest
import torch

from fastforward._orchestration.location import DeviceLocation, Location, as_location


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


def test_a_location_reads_as_itself() -> None:
    # GIVEN a location that a caller made
    location = DeviceLocation(torch.device("cpu"))

    # WHEN we read it
    # THEN it comes back as it is
    assert as_location(location) is location


def test_a_name_that_is_no_device_is_rejected() -> None:
    # GIVEN a string that names no device
    # WHEN we read it as a location
    # THEN it is refused, and the message says what a device name looks like
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
