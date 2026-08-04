# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import pytest
import torch

from fastforward._orchestration.data_flow import (
    ActivationsFlow,
    FlowMode,
    InputActivations,
    OutputActivations,
)
from fastforward._orchestration.graph_module import Region


def test_invalid_mode_raises_listing_valid_spellings() -> None:
    # GIVEN a misspelled flow mode
    # WHEN building a flow with it
    # THEN a ValueError names both valid spellings
    with pytest.raises(ValueError, match="'original'") as excinfo:
        InputActivations("quantised")
    assert "'quantized'" in str(excinfo.value)


@pytest.mark.parametrize("flow_cls", [InputActivations, OutputActivations], ids=["input", "output"])
def test_mode_accepts_string_and_enum_equivalently(flow_cls: type[ActivationsFlow]) -> None:
    # GIVEN a flow mode expressed as a string and as the equivalent enum member
    # WHEN building a flow with each
    from_string = flow_cls("original")
    from_enum = flow_cls(FlowMode.ORIGINAL)

    # THEN both are equal and resolve to the same enum member
    assert from_string == from_enum
    assert from_string.mode is FlowMode.ORIGINAL
    assert from_enum.mode is FlowMode.ORIGINAL


@pytest.mark.parametrize(
    "source",
    [42, torch.nn.Linear(4, 4)],
    ids=["non_callable", "module_instance"],
)
def test_invalid_source_raises(source: object) -> None:
    # GIVEN an invalid flow source: a non-callable, or a Module (structurally
    # callable but not a valid resolver)
    # WHEN building a flow with it
    # THEN a TypeError is raised
    with pytest.raises(TypeError, match="Invalid flow source"):
        InputActivations("original", source=source)


def test_callable_source_is_accepted_and_kept() -> None:
    # GIVEN a resolver naming the module execution should start at
    def source(region: Region) -> torch.nn.Module:
        assert isinstance(region, torch.nn.Module)
        return region

    # WHEN building a flow with it
    flow = InputActivations("original", source=source)

    # THEN the resolver is kept on the flow unchanged, next to the normalized mode
    assert flow.source is source
    assert flow.mode is FlowMode.ORIGINAL


def test_source_defaults_to_none_meaning_unbounded() -> None:
    # GIVEN no source
    # WHEN building a flow
    flow = InputActivations("original")

    # THEN the source is None: execution is not bounded by a starting module
    assert flow.source is None


@pytest.mark.parametrize("cache", [True, False], ids=["cached", "uncached"])
def test_cache_is_carried_through_make(cache: bool) -> None:
    # GIVEN an explicit cache choice
    # WHEN building a flow with it
    flow = InputActivations("original", cache=cache)

    # THEN the flow reports that choice
    assert flow.cache is cache
