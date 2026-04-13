# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Any

from fastforward.export.pipeline import StageReference


def dummy_stage_fn(*_args: Any, **_kwargs: Any) -> str:
    return "dummy_stage"


def another_dummy_stage_fn(*_args: Any, **_kwargs: Any) -> str:
    return "another_dummy_stage"


def test_stage_reference_creation() -> None:
    # GIVEN a function
    # WHEN registering said function in a StageReference instance
    stage = StageReference(dummy_stage_fn, "dummy_stage")

    # THEN the stage reference points to the correct function, has
    # the given name, and has no dependencies
    assert stage.stage_fn == dummy_stage_fn
    assert stage.name == "dummy_stage"
    assert stage._dependencies == {}


def test_stage_reference_dependencies() -> None:
    # GIVEN various stage references
    stage1 = StageReference(dummy_stage_fn, "stage1")
    stage2 = StageReference(another_dummy_stage_fn, "stage2")
    stage3 = StageReference(dummy_stage_fn, "stage3")

    # WHEN creating dependencies between stages
    result1 = stage3.depends_on(stage1)
    result2 = stage3.depends_on(stage2)

    # THEN the result of the `depends_on` method should be the StageReference
    # instance calling that method.
    assert result1 is result2 is stage3

    # THEN the correct dependencies are added to each stage
    assert len(stage1._dependencies) == 0
    assert len(stage2._dependencies) == 0
    assert len(stage3._dependencies) == 2

    assert stage1 in stage3._dependencies
    assert stage2 in stage3._dependencies

    assert stage3._dependencies[stage1] == "stage1"
    assert stage3._dependencies[stage2] == "stage2"


def test_duplicate_dependencies() -> None:
    # GIVEN various stage references
    stage1 = StageReference(dummy_stage_fn, "stage1")
    stage2 = StageReference(another_dummy_stage_fn, "stage2")
    stage3 = StageReference(dummy_stage_fn, "stage3")

    # WHEN registering the same dependency twice
    stage2.depends_on(stage1)
    stage2.depends_on(stage1)

    # THEN the dependency should be registered only one time
    assert len(stage2._dependencies) == 1
    assert stage1 in stage2._dependencies

    # WHEN registering a stage with the same function as an earlier stage
    stage2.depends_on(stage3)

    # THEN the dependency is not considered a duplicate and is added to the stage dependencies
    assert stage1.stage_fn == stage3.stage_fn
    assert len(stage2._dependencies) == 2
    assert stage3 in stage2._dependencies
