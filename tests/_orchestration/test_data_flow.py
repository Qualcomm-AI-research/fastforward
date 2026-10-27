# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from contextlib import nullcontext
from typing import Any, Iterator

import pytest
import torch

from fastforward._orchestration.data_flow import (
    ANY,
    ORIGINAL,
    QUANTIZED,
    FlowGenerator,
    FlowOp,
    Gradient,
    InputActivations,
    ModelOutputs,
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
    assert flow.generator.order == 0


def test_bare_context_manager_reuses_registered() -> None:
    # GIVEN two flows built with the same CM type
    flow_a = InputActivations(nullcontext())
    flow_b = InputActivations(nullcontext())

    # THEN they resolve to the same generator instance
    assert flow_a.generator is flow_b.generator


@pytest.mark.parametrize("cache", [True, False], ids=["cached", "uncached"])
@pytest.mark.parametrize("context", ["original", "quantized"], ids=["original", "quantized"])
def test_cache_is_carried_through(context: str, cache: bool) -> None:
    if context == "original" and not cache:
        pytest.skip("guarded by test_original_with_cache_false_is_rejected")

    # GIVEN an explicit cache choice
    # WHEN building a flow with it
    flow = InputActivations(context, cache=cache)

    # THEN the flow reports that choice
    assert flow.cache is cache


@pytest.mark.parametrize("flow_cls", [InputActivations, OutputActivations], ids=["input", "output"])
def test_original_with_cache_false_is_rejected(flow_cls: type[InputActivations]) -> None:
    # GIVEN an ORIGINAL flow, which cannot be produced again after a mutation
    # WHEN declaring it uncached
    # THEN the declaration itself is rejected
    with pytest.raises(NotImplementedError, match="copy of the model"):
        flow_cls(ORIGINAL, cache=False)


def test_builtin_generator_order() -> None:
    # ORIGINAL < QUANTIZED < ANY
    assert ORIGINAL.order < QUANTIZED.order < ANY.order


def test_register_generator_makes_key_available() -> None:
    # GIVEN a custom generator
    custom = FlowGenerator("test_custom", lambda _: nullcontext(), order=3)
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

    # THEN the default order is lowest
    assert flow.generator.order == 0


def _loss(**_: object) -> Iterator[torch.Tensor]:
    yield torch.zeros(())


def test_gradient_runs_under_the_generator_of_its_model_outputs() -> None:
    # WHEN declaring a gradient that reads the original outputs and a label value
    flow = Gradient(_loss, logits=ModelOutputs("original"), labels=[torch.tensor([1])])

    # THEN the flow runs under the generator of the outputs
    assert flow.generator is ORIGINAL


def _other_loss(**_: object) -> Iterator[torch.Tensor]:
    yield torch.zeros(())


def test_gradients_with_the_same_loss_and_sources_are_equal() -> None:
    # GIVEN a gradient with a label value
    labels = [torch.tensor([1])]
    gradient = Gradient(_loss, outputs=ModelOutputs("original"), labels=labels)

    # THEN a gradient with the same function, equal outputs and the same labels
    # object is equal, and has the same hash
    same = Gradient(_loss, outputs=ModelOutputs(ORIGINAL), labels=labels)
    assert gradient == same
    assert hash(gradient) == hash(same)

    # THEN another function or an equal copy of the labels gives another gradient
    assert gradient != Gradient(_other_loss, outputs=ModelOutputs("original"), labels=labels)
    assert gradient != Gradient(_loss, outputs=ModelOutputs("original"), labels=list(labels))


def _first(output: Any) -> Any:
    return output[0]


def test_gradients_are_equal_only_if_their_outputs_use_the_same_select() -> None:
    # GIVEN a gradient whose outputs have a select function
    gradient = Gradient(_loss, outputs=ModelOutputs("original", select=_first))

    # THEN a gradient with the same select function is equal, and has the same hash
    same = Gradient(_loss, outputs=ModelOutputs(ORIGINAL, select=_first))
    assert gradient == same
    assert hash(gradient) == hash(same)

    # THEN no select, or another select function, gives another gradient
    assert gradient != Gradient(_loss, outputs=ModelOutputs("original"))
    assert gradient != Gradient(_loss, outputs=ModelOutputs("original", select=lambda x: x))


def test_select_must_be_callable() -> None:
    # WHEN declaring model outputs with a select that is not callable
    # THEN it is refused
    with pytest.raises(TypeError, match="callable"):
        ModelOutputs("original", select=object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("f", "sources", "error", "match"),
    [
        (object(), {"outputs": ModelOutputs("original")}, TypeError, "callable"),
        (_loss, {"outputs": ModelOutputs("quantized")}, NotImplementedError, "pinned"),
        (
            _loss,
            {"a": ModelOutputs("original"), "b": ModelOutputs("quantized")},
            ValueError,
            "one generator",
        ),
        (_loss, {"labels": [1, 2]}, ValueError, "one generator"),
    ],
    ids=["not-callable", "not-pinned", "mixed-generators", "no-model-outputs"],
)
def test_unsupported_gradients_are_rejected(
    f: Any, sources: dict[str, Any], error: type[Exception], match: str
) -> None:
    # WHEN declaring a gradient that the scheduler cannot produce
    # THEN the declaration itself is rejected
    with pytest.raises(error, match=match):
        Gradient(f, **sources)


def _total(activations: Any) -> Any:
    return sum(args[0].sum() for args, _ in activations)


# A pinned generator that comes after a live one, to show that `pinned` decides first.
LATE_PINNED = register_generator(
    FlowGenerator("test_flow_op_late_pinned", lambda _: nullcontext(), order=20, pinned=True)
)


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        ([InputActivations("original")], ORIGINAL),
        ([InputActivations("quantized")], QUANTIZED),
        ([InputActivations("original"), InputActivations(LATE_PINNED)], LATE_PINNED),
        ([InputActivations("original"), InputActivations("quantized")], QUANTIZED),
        ([InputActivations(LATE_PINNED), InputActivations("quantized")], QUANTIZED),
    ],
    ids=["pinned", "live", "all-pinned-highest-order", "mixed", "live-before-later-pinned"],
)
def test_flow_op_runs_under_the_live_input_with_the_highest_order(
    inputs: list[InputActivations], expected: FlowGenerator
) -> None:
    # GIVEN an op over pinned inputs, live inputs, or both
    named: dict[str, Any] = {f"x{i}": flow for i, flow in enumerate(inputs)}
    op = FlowOp(_total, **named)

    # THEN it runs under the live input with the highest order, or under the pinned
    # input with the highest order if no input is live, so it is pinned only if all
    # its inputs are pinned
    assert op.generator is expected
    assert op.generator.pinned == all(flow.generator.pinned for flow in inputs)


def test_flow_ops_are_equal_only_if_they_read_the_same_settings() -> None:
    # GIVEN two functions that read settings from a module
    def one(_module: torch.nn.Module) -> dict[str, int]:
        return {"setting": 1}

    def other(_module: torch.nn.Module) -> dict[str, int]:
        return {"setting": 2}

    flow = InputActivations("original")

    # THEN ops are equal if `fn`, the reader and the inputs are equal
    # (in which case we only need to run one, and not all)
    assert FlowOp(_total, from_module=one, x=flow) == FlowOp(_total, from_module=one, x=flow)
    assert FlowOp(_total, from_module=one, x=flow) != FlowOp(_total, from_module=other, x=flow)
    assert FlowOp(_total, from_module=one, x=flow) != FlowOp(_total, x=flow)


@pytest.mark.parametrize(
    ("fn", "inputs", "error", "match"),
    [
        (_total, {}, ValueError, "at least one input"),
        (object(), {"activations": InputActivations("original")}, TypeError, "callable"),
    ],
    ids=["no-input", "not-callable"],
)
def test_unsupported_flow_ops_are_rejected(
    fn: Any, inputs: dict[str, Any], error: type[Exception], match: str
) -> None:
    # WHEN declaring an op that cannot run
    # THEN the declaration itself is rejected
    with pytest.raises(error, match=match):
        FlowOp(fn, **inputs)
