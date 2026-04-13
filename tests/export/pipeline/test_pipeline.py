# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Any, TypeAlias

import pytest
import torch

from fastforward.export.pipeline import Pipeline, StageReference, _ExecutionContext

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
    pipeline = Pipeline()
    pipeline.register_stage(dummy_stage_fn, "stage1")

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
    pipeline = Pipeline()

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

    pipeline = Pipeline()
    pipeline.register_stage(noisy_stage, "noisy_stage")

    built_pipeline = pipeline._build_pipeline()
    pipeline._execute_pipeline(built_pipeline, mock_module, sample_inputs)

    captured = capfd.readouterr()
    assert "stage-stdout" in captured.out


def test_compare_eval_stages_preserves_metric_dtype_and_device() -> None:
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

    eval_results = pipeline._compare_eval_stages(results, eval_metric=custom_metric)
    metric_result = eval_results[(stage1, stage2)]

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
