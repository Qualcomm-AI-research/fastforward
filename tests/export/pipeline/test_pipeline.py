# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Any, TypeAlias

import pytest
import torch

from fastforward.export.pipeline import Pipeline, StageReference
from fastforward.export.pipeline.core import _ExecutionContext

_SampleInputsT: TypeAlias = list[tuple[tuple[Any, ...], dict[str, Any]]]


def dummy_stage_fn(*_args: Any, **_kwargs: Any) -> str:
    return "dummy_stage"


def another_dummy_stage_fn(*_args: Any, **_kwargs: Any) -> str:
    return "another_dummy_stage"


@pytest.fixture
def mock_module() -> torch.nn.Module:
    return torch.nn.Linear(10, 5)


@pytest.fixture
def sample_inputs() -> _SampleInputsT:
    return [((torch.randn(1, 10),), {})]


def test_pipeline_stage_register_duplicate_name_raises() -> None:
    # GIVEN a pipeline with a stage already registered under the name "stage1"
    pipeline = Pipeline()
    pipeline.register_stage(dummy_stage_fn, "stage1")

    # WHEN registering another stage with the same name
    # THEN a ValueError is raised indicating the name is already taken
    with pytest.raises(ValueError, match="Stage name 'stage1' is already registered"):
        pipeline.register_stage(another_dummy_stage_fn, "stage1")


def test_pipeline_stage_register() -> None:
    # GIVEN a pipeline instance
    pipeline = Pipeline()

    # WHEN registering a stage with capture_output set to False
    stage1 = pipeline.register_stage(dummy_stage_fn, "stage1", capture_stage_output=False)

    # THEN the stage should be present in the `_stages` attribute but NOT the `_result_stages` attribute
    assert stage1 in pipeline._stages
    assert stage1 not in pipeline._result_stages

    # WHEN registering a stage with capture_output set to True
    stage2 = pipeline.register_stage(another_dummy_stage_fn, "stage2", capture_stage_output=True)

    # THEN the stage should be present in both the `_stages` and `_result_stages` attributes
    # of the pipeline instance.
    assert stage2 in pipeline._stages
    assert stage2 in pipeline._result_stages
    assert pipeline._result_stages[stage2] == "stage2"

    # WHEN registering a stage as an evaluation stage
    stage3 = pipeline.register_eval_stage(dummy_stage_fn, "stage3", capture_stage_output=True)

    # THEN the stage should be present in both the `_stages` and `_result_stages` attributes
    # of the pipeline instance.
    assert stage3 in pipeline._stages
    assert stage3 in pipeline._result_stages
    assert stage3 in pipeline._evaluation_stages


def test_build_pipeline() -> None:
    # GIVEN a pipeline and some reference stages
    pipeline = Pipeline()

    # WHEN registering dummy stages and creating dependencies
    stage1 = pipeline.register_stage(dummy_stage_fn, "stage1")
    stage2 = pipeline.register_stage(dummy_stage_fn, "stage2")
    stage3 = pipeline.register_stage(dummy_stage_fn, "stage3")
    stage4 = pipeline.register_stage(dummy_stage_fn, "stage4")

    stage2.depends_on(stage1)
    stage3.depends_on(stage1)
    stage4.depends_on(stage2, stage3)

    # THEN the ordering of the dependencies should be correct
    built_pipeline = pipeline._build_pipeline()

    assert len(built_pipeline) == 4
    assert built_pipeline.index(stage1) < built_pipeline.index(stage2)
    assert built_pipeline.index(stage1) < built_pipeline.index(stage3)
    assert built_pipeline.index(stage2) < built_pipeline.index(stage4)
    assert built_pipeline.index(stage3) < built_pipeline.index(stage4)


def test_build_pipeline_circular_dependency() -> None:
    # GIVEN a pipeline and some reference stages
    pipeline = Pipeline()

    # WHEN registering a circular dependency
    stage1 = pipeline.register_stage(dummy_stage_fn, "stage1")
    stage2 = pipeline.register_stage(dummy_stage_fn, "stage2")

    stage1.depends_on(stage2)
    stage2.depends_on(stage1)

    # THEN this pipeline built should raise an error
    with pytest.raises(ValueError, match="Circular dependency detected"):
        pipeline._build_pipeline()


def test_build_target_stage_raises_for_unknown_stage() -> None:
    # GIVEN a pipeline with one registered stage and an unregistered StageReference
    pipeline = Pipeline()
    pipeline.register_stage(dummy_stage_fn, "stage1")
    unknown_stage = StageReference(dummy_stage_fn, "unknown_stage")

    # WHEN building the pipeline with the unregistered stage as the target
    # THEN a ValueError is raised indicating the target is not a known stage
    with pytest.raises(ValueError, match="Target stage 'unknown_stage' is not a known stage"):
        pipeline._build_pipeline(target_stage=unknown_stage)


def test_build_target_stage() -> None:
    # GIVEN a pipeline with chained dependencies
    pipeline = Pipeline()
    stage1 = pipeline.register_stage(dummy_stage_fn, "stage1")
    stage2 = pipeline.register_stage(dummy_stage_fn, "stage2")
    stage3 = pipeline.register_stage(dummy_stage_fn, "stage3")

    stage2.depends_on(stage1)
    stage3.depends_on(stage2)

    # WHEN building pipeline up to stage2
    built_pipeline = pipeline._build_pipeline(target_stage=stage2)

    # THEN only stage1 and stage2 should be included
    assert len(built_pipeline) == 2
    assert stage1 in built_pipeline
    assert stage2 in built_pipeline
    assert stage3 not in built_pipeline


def test_execution_context_result_retrieval(sample_inputs: _SampleInputsT) -> None:
    # GIVEN an execution context instance, a StageReference instance
    ctx = _ExecutionContext(None, {}, sample_inputs)
    stage = StageReference(dummy_stage_fn, "dummy_stage")
    result = "dummy_stage"

    # WHEN saving the stage result in the execution context
    ctx.save_result(stage, result)

    # THEN the retrieved result should be the same as the stored result
    assert ctx.get_results(stage) == result

    # THEN requesting a result that was not saved should result in an error
    non_existing_stage = StageReference(another_dummy_stage_fn, "another_dummy_stage")
    with pytest.raises(KeyError, match="No results saved for stage"):
        ctx.get_results(non_existing_stage)


def test_pipeline_execution_call(
    mock_module: torch.nn.Module, sample_inputs: _SampleInputsT
) -> None:
    def stage1(
        modules: tuple[torch.nn.Module], sample_inputs: _SampleInputsT, context: dict[str, Any]
    ) -> str:
        del sample_inputs, context
        return f"stage1_output_{modules[0].__class__.__name__}"

    def stage2(
        modules: tuple[torch.nn.Module], sample_inputs: _SampleInputsT, context: dict[str, Any]
    ) -> str:
        del sample_inputs, context
        return f"stage2_output_{modules[0]}"

    # GIVEN a pipeline and some dummy stages
    pipeline = Pipeline()
    s1 = pipeline.register_stage(stage1, "stage1")
    s2 = pipeline.register_stage(stage2, "stage2")
    s2.depends_on(s1)

    # WHEN building and executing the pipeline
    built_pipeline = pipeline._build_pipeline()
    ctx = pipeline._execute_pipeline(built_pipeline, mock_module, sample_inputs)

    # THEN the correct results should be captured in the execution context results dictionary
    assert "stage1_output_Linear" in ctx.results[s1]
    assert "stage2_output_stage1_output_Linear" in ctx.results[s2]


def test_pipeline_full_no_eval_stage(
    mock_module: torch.nn.Module, sample_inputs: _SampleInputsT
) -> None:
    def simple_stage(
        modules: tuple[torch.nn.Module], sample_inputs: _SampleInputsT, context: dict[str, Any]
    ) -> torch.nn.Module:
        del sample_inputs, context
        return modules[0]

    # GIVEN a pipeline with a single dummy (pass through stage)
    pipeline = Pipeline()
    simple_stage_ref = pipeline.register_stage(
        simple_stage, "simple_stage", capture_stage_output=True
    )

    # WHEN calling the pipeline
    results, eval_results = pipeline(mock_module, sample_inputs)

    # THEN there should be no evaluation results, and the only returned entry should be the original module
    assert len(eval_results) == 0
    assert isinstance(results, dict)
    assert len(results) == 1
    assert results[simple_stage_ref.name] == mock_module


def test_pipeline_return_last_raises_on_empty_pipeline(sample_inputs: _SampleInputsT) -> None:
    # GIVEN an empty pipeline with no registered stages
    pipeline = Pipeline()

    # WHEN calling the pipeline with _return_last=True
    # THEN a ValueError is raised
    with pytest.raises(ValueError, match="Cannot return last stage result"):
        pipeline(torch.nn.Identity(), sample_inputs, _return_last=True)


def test_pipeline_stage_output_is_not_suppressed(
    mock_module: torch.nn.Module, sample_inputs: _SampleInputsT, capfd: pytest.CaptureFixture[str]
) -> None:
    def noisy_stage(
        modules: tuple[torch.nn.Module], sample_inputs: _SampleInputsT, context: dict[str, Any]
    ) -> torch.nn.Module:
        del sample_inputs, context
        print("stage-stdout")
        return modules[0]

    # GIVEN a pipeline with a stage that prints to stdout
    pipeline = Pipeline()
    pipeline.register_stage(noisy_stage, "noisy_stage")

    # WHEN building and executing the pipeline
    built_pipeline = pipeline._build_pipeline()
    pipeline._execute_pipeline(built_pipeline, mock_module, sample_inputs)

    # THEN the stage's stdout output is captured and not suppressed
    captured = capfd.readouterr()
    assert "stage-stdout" in captured.out


def test_compare_eval_stages_preserves_metric_dtype_and_device() -> None:
    # GIVEN a pipeline with two eval stages, sample tensors on the available device,
    # and a custom metric that returns float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def eval_stage(
        modules: tuple[torch.nn.Module], sample_inputs: _SampleInputsT, context: dict[str, Any]
    ) -> list[torch.Tensor]:
        del modules, sample_inputs, context
        return []

    def custom_metric(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a - b).abs().mean().to(dtype=torch.float64)

    pipeline = Pipeline()
    stage1 = pipeline.register_eval_stage(eval_stage, "eval_stage1")
    stage2 = pipeline.register_eval_stage(eval_stage, "eval_stage2")

    results: dict[StageReference, list[torch.Tensor]] = {
        stage1: [torch.randn(5, device=device), torch.randn(5, device=device)],
        stage2: [torch.randn(5, device=device), torch.randn(5, device=device)],
    }

    # WHEN comparing the eval stages using the custom metric
    eval_results = pipeline._compare_eval_stages(results, eval_metric=custom_metric)
    metric_result = eval_results[(stage1, stage2)]

    # THEN the metric result preserves the dtype and device returned by the metric
    assert metric_result.dtype == torch.float64
    assert metric_result.device.type == device.type
    if device.index is not None:
        assert metric_result.device.index == device.index


def test_pipeline_full_with_eval_stage(
    mock_module: torch.nn.Module, sample_inputs: _SampleInputsT
) -> None:
    def random_eval_stage(
        modules: tuple[torch.nn.Module], sample_inputs: _SampleInputsT, context: dict[str, Any]
    ) -> list[torch.Tensor]:
        del modules, context
        return [torch.randn(5) for _ in range(len(sample_inputs))]

    def custom_metric(_a: Any, _b: Any) -> torch.Tensor:
        return torch.tensor(0.5)

    # GIVEN a pipeline with two dummy evaluation stages
    pipeline = Pipeline()
    eval_stage1_ref = pipeline.register_eval_stage(random_eval_stage, "random_eval_stage1")
    eval_stage2_ref = pipeline.register_eval_stage(random_eval_stage, "random_eval_stage2")

    # WHEN calling the pipeline with a custom metric
    results, eval_results = pipeline(mock_module, sample_inputs, eval_metric=custom_metric)

    # THEN there should be no captured stage results, and only a single evaluation result.
    assert len(results) == 0
    assert isinstance(eval_results, dict)
    assert len(eval_results) == 1

    eval_key1 = (eval_stage1_ref, eval_stage2_ref)
    eval_key2 = (eval_stage2_ref, eval_stage1_ref)
    metric_result = eval_results.get(eval_key1) or eval_results.get(eval_key2)
    assert metric_result is not None and metric_result == 0.5


def test_get_stage_returns_registered_reference() -> None:
    # GIVEN a pipeline with one registered stage
    pipeline = Pipeline()
    stage = pipeline.register_stage(dummy_stage_fn, "stage1")

    # WHEN retrieving the stage by name
    # THEN the returned reference is the same object that was registered
    assert pipeline.get_stage("stage1") is stage


def test_get_stage_unknown_name_raises() -> None:
    # GIVEN an empty pipeline
    pipeline = Pipeline()

    # WHEN retrieving a stage by an unknown name
    # THEN a KeyError is raised
    with pytest.raises(KeyError, match="No stage named 'missing'"):
        pipeline.get_stage("missing")


def test_insert_stage_before_splices_into_chain(
    mock_module: torch.nn.Module, sample_inputs: _SampleInputsT
) -> None:
    def head_stage(
        modules: tuple[torch.nn.Module], sample_inputs: _SampleInputsT, context: dict[str, Any]
    ) -> str:
        del sample_inputs, context
        return f"head_{modules[0].__class__.__name__}"

    def make_passthrough(label: str) -> Any:
        def stage(
            args: tuple[Any, ...], sample_inputs: _SampleInputsT, context: dict[str, Any]
        ) -> str:
            del sample_inputs, context
            return f"{label}({args[0]})"

        return stage

    # GIVEN a chain head -> tail
    pipeline = Pipeline()
    head = pipeline.register_stage(head_stage, "head")
    tail = pipeline.register_stage(make_passthrough("tail"), "tail").depends_on(head)

    # WHEN inserting a stage before the tail (default: inherit head as dep)
    inserted = pipeline.insert_stage_before("tail", make_passthrough("inserted"), name="inserted")

    # THEN ordering is head -> inserted -> tail and tail reads inserted's output
    built = pipeline._build_pipeline()
    assert built.index(head) < built.index(inserted) < built.index(tail)

    results, _ = pipeline(mock_module, sample_inputs, _return_last=True)
    assert results == {"tail": "tail(inserted(head_Linear))"}


def test_insert_stage_before_with_explicit_depends_on_does_not_rewire_target() -> None:
    # GIVEN a pipeline with chain head -> tail
    pipeline = Pipeline()
    head = pipeline.register_stage(dummy_stage_fn, "head")
    tail = pipeline.register_stage(dummy_stage_fn, "tail").depends_on(head)

    # WHEN inserting a stage before "tail" with an explicit empty depends_on
    new = pipeline.insert_stage_before("tail", dummy_stage_fn, name="parallel", depends_on=())

    # THEN the new stage has no deps and tail still depends only on head
    assert new.dependencies == ()
    assert tail.dependencies == (head,)


def test_insert_stage_before_unknown_target_raises() -> None:
    # GIVEN an empty pipeline
    pipeline = Pipeline()

    # WHEN inserting a stage before an unknown target name
    # THEN a KeyError is raised
    with pytest.raises(KeyError, match="No stage named 'missing'"):
        pipeline.insert_stage_before("missing", dummy_stage_fn, name="new")


def test_insert_stage_before_duplicate_name_raises() -> None:
    # GIVEN a pipeline with a registered stage named "tail"
    pipeline = Pipeline()
    pipeline.register_stage(dummy_stage_fn, "tail")

    # WHEN inserting a stage before "tail" with a name that is already registered
    # THEN a ValueError is raised
    with pytest.raises(ValueError, match="Stage name 'tail' is already registered"):
        pipeline.insert_stage_before("tail", dummy_stage_fn, name="tail")


def test_insert_stage_after_rewires_all_dependents() -> None:
    # GIVEN a pipeline where both "b" and "c" depend on "a"
    pipeline = Pipeline()
    a = pipeline.register_stage(dummy_stage_fn, "a")
    b = pipeline.register_stage(dummy_stage_fn, "b").depends_on(a)
    c = pipeline.register_stage(dummy_stage_fn, "c").depends_on(a)

    # WHEN inserting a new stage after "a"
    new = pipeline.insert_stage_after("a", dummy_stage_fn, name="new")

    # THEN new depends on "a", and both "b" and "c" are rewired to depend on new instead
    assert new.dependencies == (a,)
    assert b.dependencies == (new,)
    assert c.dependencies == (new,)
    assert a not in b._dependencies and a not in c._dependencies


def test_insert_stage_after_unknown_target_raises() -> None:
    # GIVEN an empty pipeline
    pipeline = Pipeline()

    # WHEN inserting a stage after an unknown target name
    # THEN a KeyError is raised
    with pytest.raises(KeyError, match="No stage named 'missing'"):
        pipeline.insert_stage_after("missing", dummy_stage_fn, name="new")


def test_replace_stage_preserves_position_and_collections() -> None:
    # GIVEN a pipeline with chain head -> middle -> tail where middle captures output
    pipeline = Pipeline()
    head = pipeline.register_stage(dummy_stage_fn, "head")
    middle = pipeline.register_stage(
        dummy_stage_fn, "middle", capture_stage_output=True
    ).depends_on(head)
    tail = pipeline.register_stage(dummy_stage_fn, "tail").depends_on(middle)

    # WHEN replacing the middle stage with a different function
    new = pipeline.replace_stage("middle", another_dummy_stage_fn)

    # THEN the new stage keeps the target's name, inherits deps and dependents, and
    # the old reference is removed from all collections while the new one takes its place
    assert new.name == "middle"
    assert new is not middle
    # New stage takes over deps and dependents
    assert new.dependencies == (head,)
    assert tail.dependencies == (new,)
    # Old reference is gone, new one is in result-capture set (capture inherited)
    assert middle not in pipeline._stages
    assert middle not in pipeline._result_stages
    assert new in pipeline._stages
    assert new in pipeline._result_stages


def test_replace_stage_preserves_eval_membership() -> None:
    # GIVEN a pipeline with an eval stage that captures output
    pipeline = Pipeline()
    eval_stage = pipeline.register_eval_stage(dummy_stage_fn, "eval", capture_stage_output=True)

    # WHEN replacing the eval stage with a different function and a new name
    new = pipeline.replace_stage("eval", another_dummy_stage_fn, name="eval2")

    # THEN the new stage is registered as an eval stage and the old one is removed from all collections
    assert new.name == "eval2"
    assert new in pipeline._evaluation_stages
    assert eval_stage not in pipeline._evaluation_stages
    assert eval_stage not in pipeline._stages


def test_replace_stage_capture_override() -> None:
    # GIVEN a pipeline with a stage that captures output
    pipeline = Pipeline()
    pipeline.register_stage(dummy_stage_fn, "x", capture_stage_output=True)

    # WHEN replacing the stage with capture_stage_output=False
    new = pipeline.replace_stage("x", another_dummy_stage_fn, capture_stage_output=False)

    # THEN the new stage does not capture output
    assert new not in pipeline._result_stages


def test_replace_stage_with_conflicting_new_name_raises() -> None:
    # GIVEN a pipeline with two registered stages "a" and "b"
    pipeline = Pipeline()
    pipeline.register_stage(dummy_stage_fn, "a")
    pipeline.register_stage(dummy_stage_fn, "b")

    # WHEN replacing stage "a" with a new name that conflicts with "b"
    # THEN a ValueError is raised and the original stage "a" is left untouched
    with pytest.raises(ValueError, match="Stage name 'b' is already registered"):
        pipeline.replace_stage("a", another_dummy_stage_fn, name="b")

    # Original `a` is untouched after a failed replacement.
    assert pipeline.get_stage("a").stage_fn is dummy_stage_fn


def test_replace_stage_unknown_target_raises() -> None:
    # GIVEN an empty pipeline
    pipeline = Pipeline()

    # WHEN replacing a stage that does not exist
    # THEN a KeyError is raised
    with pytest.raises(KeyError, match="No stage named 'missing'"):
        pipeline.replace_stage("missing", dummy_stage_fn)


def test_add_dependency_creates_edge_and_is_idempotent() -> None:
    # GIVEN a pipeline with two registered stages "a" and "b"
    pipeline = Pipeline()
    a = pipeline.register_stage(dummy_stage_fn, "a")
    b = pipeline.register_stage(dummy_stage_fn, "b")

    # WHEN adding the same dependency edge twice
    pipeline.add_dependency("b", "a")
    pipeline.add_dependency("b", "a")  # second call is a no-op

    # THEN only one dependency edge is created
    assert b.dependencies == (a,)


def test_add_dependency_unknown_name_raises() -> None:
    # GIVEN a pipeline with one registered stage "a"
    pipeline = Pipeline()
    pipeline.register_stage(dummy_stage_fn, "a")

    # WHEN adding a dependency on an unknown stage name
    # THEN a KeyError is raised
    with pytest.raises(KeyError, match="No stage named 'missing'"):
        pipeline.add_dependency("a", "missing")


def test_add_dependency_rejects_cycle() -> None:
    # GIVEN a pipeline where "b" already depends on "a"
    pipeline = Pipeline()
    pipeline.register_stage(dummy_stage_fn, "a")
    pipeline.register_stage(dummy_stage_fn, "b")
    pipeline.add_dependency("b", "a")

    # WHEN adding the reverse dependency that would create a cycle
    # THEN a ValueError is raised
    with pytest.raises(ValueError, match="would introduce a cycle"):
        pipeline.add_dependency("a", "b")


def test_remove_dependency_removes_edge() -> None:
    # GIVEN a pipeline where "b" depends on "a"
    pipeline = Pipeline()
    pipeline.register_stage(dummy_stage_fn, "a")
    b = pipeline.register_stage(dummy_stage_fn, "b")
    pipeline.add_dependency("b", "a")

    # WHEN removing the dependency edge
    pipeline.remove_dependency("b", "a")

    # THEN "b" has no dependencies
    assert b.dependencies == ()


def test_remove_dependency_missing_edge_raises() -> None:
    # GIVEN a pipeline with two registered stages that have no dependency between them
    pipeline = Pipeline()
    pipeline.register_stage(dummy_stage_fn, "a")
    pipeline.register_stage(dummy_stage_fn, "b")

    # WHEN removing a dependency edge that does not exist
    # THEN a KeyError is raised
    with pytest.raises(KeyError, match="does not depend on 'a'"):
        pipeline.remove_dependency("b", "a")


def test_qnn_onnx_pipeline_supports_user_inserted_stage() -> None:
    """End-to-end: user can splice a new stage into the built-in pipeline."""
    from fastforward.export.pipeline.qnn_onnx_pipeline import qnn_onnx_pipeline

    # GIVEN the built-in qnn_onnx_pipeline
    pipeline = qnn_onnx_pipeline(pipeline_kwargs={})

    # WHEN inserting a custom stage before "fx_to_onnx_program"
    pipeline.insert_stage_before("fx_to_onnx_program", dummy_stage_fn, name="fx_rewrite")

    # THEN the inserted stage appears between "cleanup_ff_quantizer_artifacts" and
    # "fx_to_onnx_program", and "fx_to_onnx_program" sees the rewrite as its only dependency
    built = pipeline._build_pipeline()
    names = [stage.name for stage in built]
    rewrite_idx = names.index("fx_rewrite")
    fx_idx = names.index("fx_to_onnx_program")
    cleanup_idx = names.index("cleanup_ff_quantizer_artifacts")

    assert cleanup_idx < rewrite_idx < fx_idx
    # Ensure fx_to_onnx_program now sees the rewrite as its only dep.
    fx_stage = pipeline.get_stage("fx_to_onnx_program")
    assert tuple(s.name for s in fx_stage.dependencies) == ("fx_rewrite",)
