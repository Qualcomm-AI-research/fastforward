# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import functools

import pytest
import torch

from fastforward import mpath
from fastforward._orchestration import registry
from fastforward._orchestration.data_flow import InputActivations
from fastforward._orchestration.registry import (
    AlgorithmSpec,
    CompositeSelector,
    ModuleInstanceSelector,
    ModuleTypeSelector,
    MPathSelector,
    NoTargetsFound,
    Selector,
    TargetType,
    _AlgorithmRegistry,
    normalize,
    override,
)

from ._models import TinyModel
from .conftest import make_flows

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
def test_register_and_resolve(
    target: TargetType | str, expected_modules_attr: list[str], tiny_model: TinyModel
) -> None:
    # GIVEN a registry and a model
    registry = _AlgorithmRegistry()
    model = tiny_model

    # Swap placeholder for actual module instances
    if target == "explicit":
        target = [model.conv]
    assert not isinstance(target, str)  # "explicit" is the only str and is swapped above

    registry.register(_dummy_algorithm, target, flows=make_flows())

    # WHEN resolving
    result = registry.resolve(model, algorithm=_dummy_algorithm)

    # THEN the expected modules are returned
    expected = [getattr(model, attr) for attr in expected_modules_attr]
    assert {s.region for s in result} == set(expected)


def test_register_overwrites_previous(tiny_model: TinyModel) -> None:
    # GIVEN a registry with Linear registered, then overwritten with Conv2d
    registry = _AlgorithmRegistry()
    model = tiny_model
    registry.register(_dummy_algorithm, torch.nn.Linear, flows=make_flows())
    registry.register(_dummy_algorithm, torch.nn.Conv2d, flows=make_flows())

    # WHEN resolving
    result = registry.resolve(model, algorithm=_dummy_algorithm)

    # THEN only Conv2d is returned (overwritten, not appended)
    assert [s.region for s in result] == [model.conv]


def test_resolve_unregistered_algorithm_raises(tiny_model: TinyModel) -> None:
    # GIVEN an empty registry
    registry = _AlgorithmRegistry()
    model = tiny_model

    # WHEN resolving an unregistered algorithm
    # THEN NoTargetsFound is raised
    with pytest.raises(NoTargetsFound, match="No target registered"):
        registry.resolve(model, algorithm=_dummy_algorithm)


def test_resolve_empty_match_raises(tiny_model: TinyModel) -> None:
    # GIVEN a registry with BatchNorm2d registered (not present in model)
    registry = _AlgorithmRegistry()
    model = tiny_model
    registry.register(_dummy_algorithm, torch.nn.BatchNorm2d, flows=make_flows())

    # WHEN resolving
    # THEN NoTargetsFound is raised
    with pytest.raises(NoTargetsFound, match="matched no modules"):
        registry.resolve(model, algorithm=_dummy_algorithm)


def test_register_and_resolve_heterogeneous_target(tiny_model: TinyModel) -> None:
    # GIVEN a registry and a model
    registry = _AlgorithmRegistry()
    model = tiny_model

    # WHEN registering a heterogeneous target (type + specific instance)
    registry.register(_dummy_algorithm, [torch.nn.Linear, model.conv], flows=make_flows())

    # THEN resolving returns all Linear layers and the conv instance
    result = registry.resolve(model, algorithm=_dummy_algorithm)
    assert {s.region for s in result} == {model.linear1, model.linear2, model.conv}


def test_module_instance_selector_missing_module_raises(tiny_model: TinyModel) -> None:
    # GIVEN a selector requiring an instance that is absent from the model
    model = tiny_model
    absent = torch.nn.Linear(4, 4)
    selector = ModuleInstanceSelector(modules=frozenset([absent]))

    # WHEN we resolve it against the model
    # THEN it raises because the required instance is not part of the model
    with pytest.raises(ValueError, match="not found on model"):
        selector.resolve(model)


def test_algorithm_registry_mapping_protocol() -> None:
    # GIVEN a registry with one algorithm registered
    registry = _AlgorithmRegistry()
    registry.register(_dummy_algorithm, torch.nn.Linear, flows=make_flows())

    # WHEN we use the Mapping protocol (__len__, __iter__, __getitem__)
    # THEN it reflects the single registration
    assert len(registry) == 1
    assert list(registry) == [_dummy_algorithm]
    spec = registry[_dummy_algorithm]
    assert spec.fn is _dummy_algorithm
    assert spec.selector == ModuleTypeSelector(types=(torch.nn.Linear,))


def test_resolve_with_explicit_specs(tiny_model: TinyModel) -> None:
    # GIVEN an empty registry and an explicitly constructed spec
    registry = _AlgorithmRegistry()
    model = tiny_model
    flows = make_flows()
    spec = AlgorithmSpec(fn=_dummy_algorithm, selector=normalize(torch.nn.Conv2d), flows=flows)

    # WHEN resolving with explicit specs (bypassing registration)
    result = registry.resolve(model, specs=[spec])

    # THEN the spec's target is resolved against the model
    assert [s.region for s in result] == [model.conv]
    assert all(s.delegate.fn is _dummy_algorithm for s in result)

    # THEN the spec's flows are carried through to the resolved region
    assert [s.flows for s in result] == [flows]


def test_override_inherits_flows_from_registration(tiny_model: TinyModel) -> None:
    # GIVEN an algorithm registered with specific flows and a target
    flows = [InputActivations("original")]
    registry.register(_dummy_algorithm, torch.nn.Linear, flows=flows)
    try:
        model = tiny_model

        # WHEN overriding the algorithm with a different target
        with override(_dummy_algorithm, model.conv):
            spec = registry._registry[_dummy_algorithm]

            # THEN the spec has the new target but the original flows
            assert spec.selector == ModuleInstanceSelector(modules=frozenset([model.conv]))
            assert spec.flows == flows
    finally:
        registry._registry._specs.pop(_dummy_algorithm, None)


def test_override_unregistered_algorithm_raises() -> None:
    # GIVEN an algorithm that was never registered
    # WHEN overriding it
    # THEN a ValueError is raised mentioning data flow requirements
    with pytest.raises(ValueError, match="data flow"):
        with override(_dummy_algorithm, torch.nn.Linear):
            pass


def _dummy_algorithm_with_args(_arg: int = 0) -> None:
    pass


def test_register_by_partial_keys_on_base_function() -> None:
    # GIVEN an algorithm registered *by* a partial
    registry_ = _AlgorithmRegistry()
    bound = functools.partial(_dummy_algorithm_with_args, _arg=1)
    registry_.register(bound, torch.nn.Linear, flows=make_flows())

    # WHEN looking it up by the bare base function
    # THEN it is found -- the key is the base function, so partial and base share
    # one registration -- and the stored fn is the partial, with its args intact.
    spec = registry_[_dummy_algorithm_with_args]
    assert spec.fn is bound
    assert spec.fn.keywords == {"_arg": 1}
