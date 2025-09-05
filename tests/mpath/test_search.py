# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from collections.abc import Sequence
from typing import Any

import pytest
import torch

from fastforward import mpath


@pytest.fixture()
def model() -> torch.nn.ModuleDict:
    module = torch.nn.ModuleDict(
        dict(
            layer1=torch.nn.ModuleList([torch.nn.Linear(10, 10), torch.nn.Linear(20, 20)]),
            layer2=torch.nn.ModuleList([torch.nn.Linear(30, 30), torch.nn.Linear(40, 40)]),
            layer3=torch.nn.ModuleList([torch.nn.Conv2d(30, 30, 3), torch.nn.Conv2d(40, 40, 3)]),
        )
    )
    module.custom_aliases = {"alias": mpath.query("layer1/0")}  # type: ignore[assignment]
    return module


def _assert_search_result(
    module: torch.nn.Module,
    query: str | mpath.Selector,
    expected_modules: Sequence[tuple[str, torch.nn.Module]],
    *,
    aliases: str | dict[str, mpath.selector.BaseSelector] | None = None,
) -> None:
    alias_kwargs: dict[str, Any] = {}
    if isinstance(aliases, str):
        alias_kwargs["root_aliases_attribute"] = aliases
    elif isinstance(aliases, dict):
        alias_kwargs["aliases"] = aliases

    results = mpath.search(query, module, **alias_kwargs)
    module_results = [(result.full_name, result.module) for result in results]

    for expected_module in expected_modules:
        assert expected_module in module_results


def test_search_class_fragment(model: torch.nn.Module) -> None:
    _assert_search_result(
        model,
        "**/[class:torch.nn.Linear]",
        [
            ("layer1.0", model.layer1[0]),
            ("layer1.1", model.layer1[1]),
            ("layer2.0", model.layer2[0]),
            ("layer2.1", model.layer2[1]),
        ],
    )


def test_search_wildcard(model: torch.nn.Module) -> None:
    _assert_search_result(
        model,
        "*",
        [
            ("layer1", model.layer1),
            ("layer2", model.layer2),
            ("layer3", model.layer3),
        ],
    )


def test_search_double_wildcard(model: torch.nn.Module) -> None:
    _assert_search_result(
        model,
        "**",
        [
            ("layer1", model.layer1),
            ("layer2", model.layer2),
            ("layer3", model.layer3),
            ("layer1.0", model.layer1[0]),
            ("layer1.1", model.layer1[1]),
            ("layer2.0", model.layer2[0]),
            ("layer2.1", model.layer2[1]),
            ("layer3.0", model.layer3[0]),
            ("layer3.1", model.layer3[1]),
        ],
    )


def test_search_negation(model: torch.nn.Module) -> None:
    _assert_search_result(
        model,
        "*/~[cls:torch.nn.Linear]",
        [
            ("layer3.0", model.layer3[0]),
            ("layer3.1", model.layer3[1]),
        ],
    )


def test_search_module_list(model: torch.nn.Module) -> None:
    _assert_search_result(
        model,
        "**/1",
        [
            ("layer1.1", model.layer1[1]),
            ("layer2.1", model.layer2[1]),
            ("layer3.1", model.layer3[1]),
        ],
    )
    _assert_search_result(
        model,
        "**/layer1/0",
        [
            ("layer1.0", model.layer1[0]),
        ],
    )


def test_search_multi_selector(model: torch.nn.Module) -> None:
    _assert_search_result(
        model,
        "**/{layer1, layer2}/0",
        [
            ("layer1.0", model.layer1[0]),
            ("layer2.0", model.layer2[0]),
        ],
    )
    _assert_search_result(
        model,
        "**/{layer1/0, layer2/1}",
        [
            ("layer1.0", model.layer1[0]),
            ("layer2.1", model.layer2[1]),
        ],
    )


def test_regex_extension(model: torch.nn.Module) -> None:
    _assert_search_result(
        model,
        r"[re:layer[12\]]/1",
        [
            ("layer1.1", model.layer1[1]),
            ("layer2.1", model.layer2[1]),
        ],
    )


def test_root_aliases(model: torch.nn.Module) -> None:
    # GIVEN a torch module with an aliases attribute
    # WHEN searching for submodules using mpath using an aliases attribute
    # THEN the aliases dictionary stored on `model` must be used.
    _assert_search_result(
        model,
        r"&alias",
        [
            ("layer1.0", model.layer1[0]),
        ],
        aliases="custom_aliases",
    )

    # WHEN searching for submodules using mpath using an aliases dictionary
    # THEN the given aliases dictionary should be used over the dict on `model`
    _assert_search_result(
        model,
        r"&alias",
        [
            ("layer2.1", model.layer2[1]),
        ],
        aliases={"alias": mpath.query("layer2/1")},
    )


def test_mpath_collection_set_operations(model: torch.nn.Module) -> None:
    results1 = mpath.search("layer1/**", model)
    results2 = mpath.search((mpath.query("layer1") | mpath.query("layer2")) / "**", model)
    results3 = mpath.search("layer3/**", model)

    assert len(results1.union(results2)) == 6
    assert len(results1.union(results3)) == 6
    assert len(results1.intersection(results2)) == 3
    assert len(results1.intersection(results3)) == 0
    assert len(results2.difference(results1)) == 3
    assert len(results2.difference(results3)) == 6
    assert len(results1.symmetric_difference(results2)) == 3
    assert len(results1.symmetric_difference(results3)) == 6

    assert results1.union(results2) == results1 | results2
    assert results1.union(results3) == results1 | results3
    assert results1.intersection(results2) == results1 & results2
    assert results1.intersection(results3) == results1 & results3
    assert results2.difference(results1) == results2 - results1
    assert results2.difference(results3) == results2 - results3
    assert results1.symmetric_difference(results2) == results1 ^ results2
    assert results1.symmetric_difference(results3) == results1 ^ results3
