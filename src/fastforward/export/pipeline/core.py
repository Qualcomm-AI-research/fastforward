# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Internal stage-based pipeline primitives for export pipeline composition.

This module provides lightweight infrastructure used to compose and execute
ordered export stages with dependency resolution and optional stage-to-stage
evaluation.
"""

import dataclasses
import itertools

from typing import Any, Callable, TypeAlias

import torch

from fastforward.testing import metrics

_SampleInputsT: TypeAlias = list[tuple[tuple[Any, ...], dict[str, Any]]]


class StageReference:
    """Reference to a pipeline stage."""

    def __init__(self, stage_fn: Callable[..., Any], name: str):
        self.stage_fn = stage_fn
        self.name = name
        self._dependencies: dict["StageReference", str] = {}

    def depends_on(self, *dependencies: "StageReference") -> "StageReference":
        """Add dependencies to this stage."""
        for dep in dependencies:
            if dep not in self._dependencies:
                self._dependencies[dep] = dep.name
        return self

    def __repr__(self) -> str:
        return f"StageReference(stage_fn={self.stage_fn}, name={self.name})"

    def __str__(self) -> str:
        return self.name


@dataclasses.dataclass
class _ExecutionContext:
    pipeline_input: Any
    pipeline_kwargs: dict[str, Any]
    sample_inputs: list[tuple[tuple[Any, ...], dict[str, Any]]]
    results: dict[StageReference, Any] = dataclasses.field(default_factory=dict)

    def save_result(self, stage: StageReference, result: Any) -> None:
        self.results[stage] = result

    def get_results(self, stage: StageReference) -> Any:
        if stage not in self.results:
            msg = f"No results saved for stage with name '{stage}'"
            raise KeyError(msg)
        return self.results[stage]

    def resolve_stage_inputs(self, stage: StageReference) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if not stage._dependencies:
            args = ((self.pipeline_input,),)
        else:
            args = (tuple(self.results[stage] for stage in stage._dependencies),)
        kwargs = {"sample_inputs": self.sample_inputs, "context": self.pipeline_kwargs}
        return args, kwargs


@dataclasses.dataclass
class _EvaluationAggregator:
    metric_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    def __call__(
        self, stage_results: dict[StageReference, list[torch.Tensor]]
    ) -> dict[tuple[StageReference, StageReference], torch.Tensor]:
        cross_results: dict[tuple[StageReference, StageReference], torch.Tensor] = {}
        for (stage_ref1, result1), (stage_ref2, result2) in itertools.combinations(
            stage_results.items(), 2
        ):
            results = [
                self.metric_fn(a.reshape(-1), b.reshape(-1)) for a, b in zip(result1, result2)
            ]
            results_mean = torch.stack(results).mean()
            cross_results[(stage_ref1, stage_ref2)] = results_mean
        return cross_results


class Pipeline:
    """A pipeline class that supports registering new stages and building the pipeline."""

    def __init__(
        self,
        pipeline_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._pipeline_kwargs = pipeline_kwargs or {}
        self._stages: dict[StageReference, str] = {}
        self._result_stages: dict[StageReference, str] = {}
        self._evaluation_stages: dict[StageReference, str] = {}

    def register_stage(
        self, stage_fn: Callable[..., Any], name: str, capture_stage_output: bool = False
    ) -> StageReference:
        """Register a stage to be executed in the pipeline."""
        stage_reference = StageReference(stage_fn, name)
        self._stages[stage_reference] = stage_reference.name
        if capture_stage_output is True:
            self._result_stages[stage_reference] = stage_reference.name
        return stage_reference

    def register_eval_stage(
        self, stage_fn: Callable[..., Any], name: str, capture_stage_output: bool = False
    ) -> StageReference:
        """Register an evaluation stage to be executed in the pipeline."""
        stage_reference = self.register_stage(stage_fn, name, capture_stage_output)
        self._evaluation_stages[stage_reference] = stage_reference.name
        return stage_reference

    def _build_pipeline(self, target_stage: StageReference | None = None) -> list[StageReference]:
        """Build the pipeline by resolving dependencies and ordering the stages."""
        # Perform a topological sort on the graph
        pipeline: list[StageReference] = []
        visited: set[StageReference] = set()
        visiting: set[StageReference] = set()

        def depth_first_traversal(stage: StageReference) -> None:
            """Recursively visit each stage and its dependencies."""
            if stage in visiting:
                msg = f"Circular dependency detected: {stage}"
                raise ValueError(msg)
            if stage in visited:
                return

            visiting.add(stage)
            for dependency in stage._dependencies:
                if dependency not in self._stages:
                    msg = f"Stage '{dependency}' is not a known stage"
                    raise ValueError(msg)
                depth_first_traversal(dependency)
            visiting.remove(stage)
            visited.add(stage)
            pipeline.append(stage)

        # Visit each stage and its dependencies
        stages = [target_stage] if target_stage else self._stages
        for stage in stages:
            depth_first_traversal(stage)

        return pipeline

    def _execute_pipeline(
        self, pipeline: list[StageReference], module: torch.nn.Module, sample_inputs: _SampleInputsT
    ) -> _ExecutionContext:
        exec_ctx = _ExecutionContext(
            pipeline_input=module,
            sample_inputs=sample_inputs,
            pipeline_kwargs={**self._pipeline_kwargs},
        )

        for stage in pipeline:
            args, kwargs = exec_ctx.resolve_stage_inputs(stage)
            result = stage.stage_fn(*args, **kwargs)
            exec_ctx.save_result(stage, result)

        return exec_ctx

    def _compare_eval_stages(
        self,
        results: dict[StageReference, list[torch.Tensor]],
        eval_metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> dict[tuple[StageReference, StageReference], torch.Tensor]:
        evaluation_stages = self._evaluation_stages
        eval_results = {key: results[key] for key in evaluation_stages if key in results}
        eval_aggregator = _EvaluationAggregator(eval_metric)

        return eval_aggregator(eval_results)

    def __call__(
        self,
        module: torch.nn.Module,
        sample_inputs: list[tuple[tuple[Any, ...], dict[str, Any]]],
        *,
        _return_last: bool = False,
        eval_metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[dict[str, Any], dict[tuple[StageReference, StageReference], torch.Tensor]]:
        """Builds and executes the pipeline."""
        if eval_metric is None:
            eval_metric = metrics.sqnr

        pipeline = self._build_pipeline()
        exec_ctx = self._execute_pipeline(pipeline, module, sample_inputs)
        eval_results = self._compare_eval_stages(exec_ctx.results, eval_metric=eval_metric)

        if _return_last:
            if not pipeline:
                msg = "Cannot return last stage result: no stages are registered"
                raise ValueError(msg)
            return ({pipeline[-1].name: exec_ctx.get_results(pipeline[-1])}, eval_results)
        return (
            {stage.name: exec_ctx.get_results(stage) for stage in self._result_stages},
            eval_results,
        )
