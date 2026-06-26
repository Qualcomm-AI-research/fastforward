# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import pytest
import torch

from fastforward import mpath
from fastforward._orchestration.registry import (
    CompositeSelector,
    ModuleInstanceSelector,
    ModuleTypeSelector,
    MPathSelector,
    NoTargetsFound,
    Selector,
    TargetType,
    _AlgorithmRegistry,
    normalize,
)

# mpath selectors compare by fragment identity, init them here once.
_ATTENTION_QUERY = mpath.query("layers/*/attention")
_ATTN_QUERY = mpath.query("layers/*/attn")


@pytest.mark.parametrize(
    "target, expected",
    [
        (torch.nn.Linear, ModuleTypeSelector(types=(torch.nn.Linear,))),
        (
            (torch.nn.Linear, torch.nn.Conv2d),
            ModuleTypeSelector(types=(torch.nn.Linear, torch.nn.Conv2d)),
        ),
        (
            [torch.nn.Linear, torch.nn.Conv2d],
            ModuleTypeSelector(types=(torch.nn.Linear, torch.nn.Conv2d)),
        ),
        (_ATTENTION_QUERY, MPathSelector(query=_ATTENTION_QUERY)),
    ],
    ids=["single_type", "tuple_of_types", "list_of_types", "mpath_query"],
)
def test_normalize_single_list_tuple(target: TargetType, expected: Selector) -> None:
    # GIVEN a target type
    # WHEN normalizing the target
    result = normalize(target)

    # THEN the correct selector type is expected
    assert result == expected


def test_normalize_list_of_instances() -> None:
    # GIVEN two module instances
    m1 = torch.nn.Linear(4, 4)
    m2 = torch.nn.Conv2d(3, 3, 1)

    # WHEN normalizing a list of instances
    result = normalize([m1, m2])

    # THEN the expected selector is a CompositeSelector of ModuleInstanceSelectors
    assert result == CompositeSelector(
        selectors=(
            ModuleInstanceSelector(modules=frozenset([m1])),
            ModuleInstanceSelector(modules=frozenset([m2])),
        )
    )


@pytest.mark.parametrize(
    "target, match",
    [
        (int, "Expected a torch.nn.Module subclass"),
        ([], "Empty target sequence"),
        ([torch.nn.Linear, 42], "Invalid target"),
        ([int, str], "Expected a torch.nn.Module subclass"),
        ("layers/*/attention", "Invalid target"),
    ],
    ids=[
        "non_module_type",
        "empty_sequence",
        "invalid_item_in_sequence",
        "non_module_types_in_sequence",
        "raw_string",
    ],
)
def test_normalize_raises(target: object, match: str) -> None:
    # WHEN normalizing an invalid target
    # THEN a TypeError is raised
    with pytest.raises(TypeError, match=match):
        normalize(target)  # type: ignore[arg-type]


def test_normalize_heterogeneous_sequence() -> None:
    # GIVEN a mix of a type, an instance, and a parsed mpath query
    m = torch.nn.Linear(4, 4)

    # WHEN normalizing a heterogeneous sequence
    result = normalize([torch.nn.Conv2d, m, _ATTN_QUERY])

    # THEN a CompositeSelector with one selector per element is returned
    assert result == CompositeSelector(
        selectors=(
            ModuleTypeSelector(types=(torch.nn.Conv2d,)),
            ModuleInstanceSelector(modules=frozenset([m])),
            MPathSelector(query=_ATTN_QUERY),
        )
    )


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(4, 4)
        self.linear2 = torch.nn.Linear(4, 4)
        self.conv = torch.nn.Conv2d(3, 3, 1)


def _dummy_algorithm() -> None:
    pass


@pytest.mark.parametrize(
    "target, expected_modules_attr",
    [
        pytest.param(torch.nn.Linear, ["linear1", "linear2"], id="type_target"),
        pytest.param(mpath.query("linear1"), ["linear1"], id="mpath_target"),
        pytest.param("explicit", ["conv"], id="module_instances"),
    ],
)
def test_register_and_resolve(target: TargetType | str, expected_modules_attr: list[str]) -> None:
    # GIVEN a registry and a model
    registry = _AlgorithmRegistry()
    model = _TinyModel()

    # Swap placeholder for actual module instances
    if target == "explicit":
        target = [model.conv]
    assert not isinstance(target, str)  # "explicit" is the only str and is swapped above

    registry.register(_dummy_algorithm, target)

    # WHEN resolving
    result = registry.resolve(model, _dummy_algorithm)

    # THEN the expected modules are returned
    expected = [getattr(model, attr) for attr in expected_modules_attr]
    assert {s.region for s in result} == set(expected)


def test_register_overwrites_previous() -> None:
    # GIVEN a registry with Linear registered, then overwritten with Conv2d
    registry = _AlgorithmRegistry()
    model = _TinyModel()
    registry.register(_dummy_algorithm, torch.nn.Linear)
    registry.register(_dummy_algorithm, torch.nn.Conv2d)

    # WHEN resolving
    result = registry.resolve(model, _dummy_algorithm)

    # THEN only Conv2d is returned (overwritten, not appended)
    assert [s.region for s in result] == [model.conv]


def test_resolve_unregistered_algorithm_raises() -> None:
    # GIVEN an empty registry
    registry = _AlgorithmRegistry()
    model = _TinyModel()

    # WHEN resolving an unregistered algorithm
    # THEN NoTargetsFound is raised
    with pytest.raises(NoTargetsFound, match="No target registered"):
        registry.resolve(model, _dummy_algorithm)


def test_resolve_empty_match_raises() -> None:
    # GIVEN a registry with BatchNorm2d registered (not present in model)
    registry = _AlgorithmRegistry()
    model = _TinyModel()
    registry.register(_dummy_algorithm, torch.nn.BatchNorm2d)

    # WHEN resolving
    # THEN NoTargetsFound is raised
    with pytest.raises(NoTargetsFound, match="matched no modules"):
        registry.resolve(model, _dummy_algorithm)


def test_register_and_resolve_heterogeneous_target() -> None:
    # GIVEN a registry and a model
    registry = _AlgorithmRegistry()
    model = _TinyModel()

    # WHEN registering a heterogeneous target (type + specific instance)
    registry.register(_dummy_algorithm, [torch.nn.Linear, model.conv])

    # THEN resolving returns all Linear layers and the conv instance
    result = registry.resolve(model, _dummy_algorithm)
    assert {s.region for s in result} == {model.linear1, model.linear2, model.conv}
