# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import collections
import gc

from pathlib import Path

import pytest
import torch

from fastforward._orchestration.location import (
    BudgetLocation,
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


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_only_a_cpu_can_hold_pinned_data() -> None:
    # GIVEN a device that is not a CPU
    device = torch.device("cuda:0")

    # WHEN we ask for pinned data on it
    # THEN the location is rejected
    with pytest.raises(ValueError, match="Only CPU memory can be pinned"):
        DeviceLocation(device, pinned=True)


def test_pinned_and_unpinned_device_locations_are_not_interchangeable() -> None:
    # GIVEN two locations for the CPU, of which one pins
    plain = DeviceLocation(torch.device("cpu"))
    pinned = DeviceLocation(torch.device("cpu"), pinned=True)

    # THEN they differ, so the offloading passes do not mistake one for the other
    assert plain != pinned
    assert len({plain, pinned}) == 2
    assert "pinned=True" in repr(pinned)
    assert "pinned" not in repr(plain)


def test_pinned_location_leaves_a_cpu_tensor_alone() -> None:
    # GIVEN a pinned location and a tensor that already is on the CPU
    location = DeviceLocation(torch.device("cpu"), pinned=True)
    tensor = torch.randn(2, 3)

    # WHEN the location receives the tensor
    received = location.receive(tensor)

    # THEN no copy is made
    assert received is tensor


@requires_cuda
def test_pinned_location_holds_accelerator_data_in_page_locked_memory() -> None:
    # GIVEN a tensor on the accelerator
    tensor = torch.randn(4, 8, device="cuda:0")

    # WHEN a pinned CPU location and a plain CPU location receive it
    pinned = DeviceLocation(torch.device("cpu"), pinned=True).receive(tensor)
    plain = DeviceLocation(torch.device("cpu")).receive(tensor)

    # THEN only the first is page-locked, and both hold the same values
    assert pinned.device == torch.device("cpu")
    assert pinned.is_pinned()
    assert not plain.is_pinned()
    assert torch.equal(pinned, plain)


@requires_cuda
def test_pinned_data_returns_to_the_accelerator_as_it_was() -> None:
    # GIVEN a pinned CPU copy of a non-contiguous tensor on the accelerator
    tensor = torch.randn(8, 4, device="cuda:0").t()
    pinned = DeviceLocation(torch.device("cpu"), pinned=True).receive(tensor)

    # WHEN it goes back to the accelerator
    returned = DeviceLocation(torch.device("cuda:0")).receive(pinned)

    # THEN the values and the shape are the same
    assert returned.shape == tensor.shape
    assert torch.equal(returned, tensor)


@pytest.mark.parametrize(
    "lazy_type", [torch.nn.parameter.UninitializedParameter, torch.nn.parameter.UninitializedBuffer]
)
@pytest.mark.parametrize("pinned", [False, True])
def test_device_location_receives_a_lazy_tensor_that_has_no_data(
    lazy_type: type[torch.Tensor], pinned: bool
) -> None:
    # GIVEN a lazy tensor, such as the scale of a quantizer that is not calibrated yet
    lazy = lazy_type()

    # WHEN a CPU location receives it
    received = DeviceLocation(torch.device("cpu"), pinned=pinned).receive(lazy)

    # THEN it arrives without torch raising on its missing storage
    assert received.device == torch.device("cpu")


def test_disk_location_receives_a_lazy_tensor_without_writing_a_file(tmp_path: Path) -> None:
    # GIVEN a lazy tensor
    lazy = torch.nn.parameter.UninitializedBuffer()

    # WHEN a disk location receives it
    received = DiskLocation(tmp_path).receive(lazy)

    # THEN it stays lazy on the CPU, and no file is written
    assert isinstance(received, torch.nn.parameter.UninitializedBuffer)
    assert received.device == torch.device("cpu")
    assert not any(tmp_path.iterdir())


@requires_cuda
def test_pinned_location_keeps_the_quantization_and_pins_the_raw_data() -> None:
    # GIVEN a quantized tensor on the accelerator
    tensor = random_quantized((4, 8), device="cuda:0")

    # WHEN a pinned CPU location receives it
    received = DeviceLocation(torch.device("cpu"), pinned=True).receive(tensor)

    # THEN the quantization is intact, and the raw data is page-locked
    assert isinstance(received, QuantizedTensor)
    assert received.device == torch.device("cpu")
    assert received.raw_data.is_pinned()
    assert torch.equal(received.dequantize(), tensor.dequantize().cpu())


@requires_cuda
def test_place_pins_every_accelerator_tensor_in_a_container() -> None:
    # GIVEN a structure that holds accelerator tensors and a leaf that is no tensor
    value = {"a": torch.randn(2, 2, device="cuda:0"), "b": [torch.randn(3, device="cuda:0"), 7]}

    # WHEN a pinned CPU location places it
    placed = DeviceLocation(torch.device("cpu"), pinned=True).place(value)

    # THEN each tensor is page-locked, and the other leaf is as it was
    assert placed["a"].is_pinned()
    assert placed["b"][0].is_pinned()
    assert placed["b"][1] == 7


@pytest.mark.parametrize("is_quantized", [False, True])
@pytest.mark.parametrize(
    "target_device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
        ),
    ],
)
def test_device_loading_releases_disk_storage(
    tmp_path: Path, target_device: str, is_quantized: bool
) -> None:
    # GIVEN a tensor whose data lives in a file
    original_tensor = random_quantized((4, 8)) if is_quantized else torch.randn(4, 8)
    disk_tensor = DiskLocation(tmp_path).receive(original_tensor)

    # WHEN it is loaded into device memory and the offloaded reference is released
    loaded_tensor = DeviceLocation(torch.device(target_device)).receive(disk_tensor)
    del disk_tensor

    # THEN the file is released while the loaded tensor remains usable
    assert list(tmp_path.iterdir()) == []
    assert loaded_tensor.device == torch.device(target_device)
    if is_quantized:
        assert isinstance(loaded_tensor, QuantizedTensor)
        assert torch.equal(loaded_tensor.dequantize().cpu(), original_tensor.dequantize())
    else:
        assert torch.equal(loaded_tensor.cpu(), original_tensor)


def test_device_loading_preserves_disk_storage_while_a_view_needs_it(tmp_path: Path) -> None:
    # GIVEN a view that still needs the offloaded data
    disk_tensor = DiskLocation(tmp_path).receive(torch.arange(8.0))
    disk_view = disk_tensor[::2]

    # WHEN the tensor is loaded and its offloaded reference is released
    loaded_tensor = DeviceLocation(torch.device("cpu")).receive(disk_tensor)
    del disk_tensor

    # THEN the view keeps the file alive and does not alias the loaded data
    loaded_tensor.add_(10)
    assert torch.equal(disk_view, torch.arange(0.0, 8.0, 2.0))
    assert len(list(tmp_path.iterdir())) == 1

    # WHEN the last view is released, as with register cleanup inserted by lifetime_management_pass
    del disk_view

    # THEN the file is removed even though the loaded tensor is still alive
    assert list(tmp_path.iterdir()) == []
    assert torch.equal(loaded_tensor, torch.arange(8.0) + 10)


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


def _budget(budget: int) -> BudgetLocation:
    """A budget on the CPU whose spill adds one to a tensor, so a test sees what spilled."""
    return BudgetLocation(torch.device("cpu"), budget, spill=_Tagging())


def _spilled(tensor: torch.Tensor) -> bool:
    """Whether `tensor` is the all-ones result of a tensor of zeros that spilled."""
    return bool(tensor.eq(1).all())


def test_budget_location_holds_data_that_fits() -> None:
    # GIVEN a budget of 32 bytes
    location = _budget(32)

    # WHEN it receives a tensor of 16 bytes
    received = location.receive(torch.zeros(4))

    # THEN the tensor stays, and its bytes count against the budget
    assert not _spilled(received)
    assert location.held_bytes == 16


def test_budget_location_spills_data_that_does_not_fit() -> None:
    # GIVEN a budget of 8 bytes
    location = _budget(8)

    # WHEN it receives a tensor of 16 bytes
    received = location.receive(torch.zeros(4))

    # THEN the tensor goes to the spill location, and no byte counts against the budget
    assert _spilled(received)
    assert location.held_bytes == 0


def test_budget_location_spills_everything_when_the_budget_is_zero() -> None:
    # GIVEN a budget of zero
    location = _budget(0)

    # WHEN it receives tensors
    # THEN all of them spill
    assert _spilled(location.receive(torch.zeros(4)))
    assert _spilled(location.receive(torch.zeros(1)))


def test_budget_location_gives_room_to_the_first_data_that_arrives() -> None:
    # GIVEN a budget for two tensors
    location = _budget(32)

    # WHEN it receives three
    received = [location.receive(torch.zeros(4)) for _ in range(3)]

    # THEN the first two stay and the third spills
    assert [_spilled(tensor) for tensor in received] == [False, False, True]
    assert location.held_bytes == 32


def test_budget_location_gets_room_back_when_held_data_is_freed() -> None:
    # GIVEN a budget for one tensor that is spent
    location = _budget(16)
    held = location.receive(torch.zeros(4))
    assert _spilled(location.receive(torch.zeros(4)))

    # WHEN the held data is freed
    del held
    gc.collect()

    # THEN the room is back, and the next tensor stays
    assert location.held_bytes == 0
    assert not _spilled(location.receive(torch.zeros(4)))


def test_budget_location_does_not_count_data_twice() -> None:
    # GIVEN a budget for one tensor that holds a tensor
    location = _budget(16)
    held = location.receive(torch.zeros(4))

    # WHEN it receives a view of that tensor
    view = held[:2]
    received = location.receive(view)

    # THEN the view costs nothing more, and is not spilled
    assert received is view
    assert location.held_bytes == 16


def test_budget_location_counts_the_whole_storage_of_a_view() -> None:
    # GIVEN a budget for 16 bytes
    location = _budget(16)

    # WHEN it receives a view of 8 bytes of a storage of 32 bytes
    received = location.receive(torch.zeros(8)[:2])

    # THEN the view spills, because the storage does not fit
    assert _spilled(received)


def test_budget_location_counts_the_raw_data_of_a_quantized_tensor() -> None:
    # GIVEN a quantized tensor
    tensor = random_quantized((4, 8))

    # WHEN a budget location receives it
    location = _budget(10**6)
    received = location.receive(tensor)

    # THEN the tensor stays, and the raw data counts
    assert isinstance(received, QuantizedTensor)
    assert location.held_bytes == tensor.raw_data.untyped_storage().nbytes()


@pytest.mark.parametrize(
    "lazy_type",
    [torch.nn.parameter.UninitializedParameter, torch.nn.parameter.UninitializedBuffer],
)
def test_budget_location_receives_a_lazy_tensor_that_has_no_data(
    lazy_type: type[torch.Tensor],
) -> None:
    # GIVEN a budget of zero and a lazy tensor, such as the scale of a quantizer that is not
    # calibrated yet
    location = _budget(0)
    lazy = lazy_type()

    # WHEN the budget location receives it
    received = location.receive(lazy)

    # THEN it stays lazy and does not spill, and no byte counts against the budget
    assert isinstance(received, lazy_type)
    assert location.held_bytes == 0


def test_budget_location_shares_its_budget_between_the_tensors_of_a_container() -> None:
    # GIVEN a budget for two tensors
    location = _budget(32)

    # WHEN it places a structure of three tensors
    placed = location.place({"a": torch.zeros(4), "b": [torch.zeros(4), torch.zeros(4)]})

    # THEN one of them spills
    assert sum(_spilled(tensor) for tensor in (placed["a"], *placed["b"])) == 1


@requires_cuda
def test_budget_location_keeps_data_on_the_device_and_spills_the_rest_to_pinned_memory() -> None:
    # GIVEN a budget on the accelerator for two tensors, and a pinned CPU spill location
    spill = DeviceLocation(torch.device("cpu"), pinned=True)
    location = BudgetLocation(torch.device("cuda:0"), 32, spill=spill)

    # WHEN it receives three tensors, of which one comes from the CPU
    first = location.receive(torch.zeros(4))
    second = location.receive(torch.zeros(4, device="cuda:0"))
    third = location.receive(torch.zeros(4, device="cuda:0"))

    # THEN two are on the accelerator, and the last one is in pinned memory
    assert first.device.type == second.device.type == "cuda"
    assert third.device == torch.device("cpu")
    assert third.is_pinned()

    # AND the room returns when the data on the accelerator is freed
    del first, second
    assert location.held_bytes == 0
