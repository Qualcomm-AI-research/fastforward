# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from contextlib import nullcontext

import pytest

from fastforward._orchestration.data_flow import (
    ANY,
    ORIGINAL,
    QUANTIZED,
    FlowGenerator,
    InputActivations,
    OutputActivations,
    register_generator,
)


def test_unknown_generator_string_raises_with_available() -> None:
    # GIVEN a misspelled generator key
    # WHEN building a flow with it
    # THEN a KeyError lists available keys
    with pytest.raises(KeyError, match="'original'"):
        InputActivations("quantised")


@pytest.mark.parametrize("flow_cls", [InputActivations, OutputActivations], ids=["input", "output"])
def test_generator_accepts_string_and_instance(flow_cls: type[InputActivations]) -> None:
    # GIVEN a generator expressed as a string and as the FlowGenerator instance
    from_string = flow_cls("original")
    from_instance = flow_cls(ORIGINAL)

    # THEN both resolve to the same generator
    assert from_string.generator is ORIGINAL
    assert from_instance.generator is ORIGINAL
    assert from_string == from_instance


def test_bare_context_manager_auto_registers() -> None:
    # GIVEN a bare context manager not previously registered
    cm = nullcontext()

    # WHEN building a flow with it
    flow = InputActivations(cm)

    # THEN a FlowGenerator is created with the CM type's qualname as key
    assert flow.generator.key == "nullcontext"
    assert flow.generator.priority == 0


def test_bare_context_manager_reuses_registered() -> None:
    # GIVEN two flows built with the same CM type
    flow_a = InputActivations(nullcontext())
    flow_b = InputActivations(nullcontext())

    # THEN they resolve to the same generator instance
    assert flow_a.generator is flow_b.generator


@pytest.mark.parametrize("cache", [True, False], ids=["cached", "uncached"])
def test_cache_is_carried_through(cache: bool) -> None:
    # GIVEN an explicit cache choice
    # WHEN building a flow with it
    flow = InputActivations("original", cache=cache)

    # THEN the flow reports that choice
    assert flow.cache is cache


def test_builtin_generator_priority_order() -> None:
    # ANY < ORIGINAL < QUANTIZED
    assert ANY.priority < ORIGINAL.priority < QUANTIZED.priority


def test_register_generator_makes_key_available() -> None:
    # GIVEN a custom generator
    custom = FlowGenerator("test_custom", lambda _: nullcontext(), priority=3)
    register_generator(custom)

    # WHEN building a flow with the key
    flow = InputActivations("test_custom")

    # THEN it resolves to the registered generator
    assert flow.generator is custom


def test_register_anonymous_generator() -> None:
    # GIVEN an anonymous context manager passed to a flow
    flow = InputActivations(nullcontext())

    # WHEN building another flow with the auto-registered key
    flow_by_key = InputActivations("nullcontext")

    # THEN both resolve to the same generator
    assert flow.generator is flow_by_key.generator

    # THEN the default priority set is lowest
    assert flow.generator.priority == 0
