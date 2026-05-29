# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Internal stage-based pipeline primitives for export pipeline composition.

This module provides lightweight infrastructure used to compose and execute
ordered export stages with dependency resolution and optional stage-to-stage
evaluation.
"""

import dataclasses
import itertools

from typing import Any, Callable, Iterable, TypeAlias

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

    @property
    def dependencies(self) -> tuple["StageReference", ...]:
        """Stages this stage depends on, in insertion order."""
        return tuple(self._dependencies)

    def _set_dependencies(self, deps: Iterable["StageReference"]) -> None:
        self._dependencies = {dep: dep.name for dep in deps}

    def _remove_dependency(self, dep: "StageReference") -> None:
        if dep not in self._dependencies:
            msg = f"Stage '{self.name}' does not depend on '{dep.name}'"
            raise KeyError(msg)
        del self._dependencies[dep]

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
        if name in self._stages.values():
            msg = f"Stage name '{name}' is already registered"
            raise ValueError(msg)

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

    def get_stage(self, name: str) -> StageReference:
        """Return the registered stage with the given name.

        Args:
            name: Name of the registered stage to look up.

        Returns:
            The :class:`StageReference` registered under ``name``.

        Raises:
            KeyError: if no stage with that name has been registered.
        """
        for stage in self._stages:
            if stage.name == name:
                return stage
        msg = f"No stage named '{name}'"
        raise KeyError(msg)

    def insert_stage_before(
        self,
        target: str,
        stage_fn: Callable[..., Any],
        name: str,
        *,
        depends_on: Iterable[str] | None = None,
        capture_stage_output: bool = False,
    ) -> StageReference:
        """Insert a new stage so that it executes immediately before ``target``.

        By default the new stage inherits ``target``'s current dependencies and
        ``target`` is rewired to depend on the new stage instead, splicing it into
        the chain. Pass ``depends_on`` to set the new stage's dependencies
        explicitly; in that case ``target``'s dependencies are left untouched and
        the new stage is wired in only via the names you provide.

        Args:
            target: Name of the stage the new stage should run immediately before.
            stage_fn: Callable implementing the new stage.
            name: Unique name to register the new stage under.
            depends_on: Optional explicit dependency names. When ``None`` (default),
                the new stage inherits ``target``'s dependencies and ``target`` is
                rewired to depend on the new stage.
            capture_stage_output: Whether the new stage's output should be captured
                in the pipeline results.

        Returns:
            The :class:`StageReference` for the newly registered stage.

        Raises:
            KeyError: if ``target`` (or any name in ``depends_on``) is unknown.
            ValueError: if ``name`` is already registered.
        """
        target_ref = self.get_stage(target)

        if depends_on is None:
            inherited = list(target_ref._dependencies)
            new_stage = self.register_stage(stage_fn, name, capture_stage_output)
            new_stage.depends_on(*inherited)
            target_ref._set_dependencies([new_stage])
        else:
            explicit = [self.get_stage(dep_name) for dep_name in depends_on]
            new_stage = self.register_stage(stage_fn, name, capture_stage_output)
            new_stage.depends_on(*explicit)
        return new_stage

    def insert_stage_after(
        self,
        target: str,
        stage_fn: Callable[..., Any],
        name: str,
        *,
        capture_stage_output: bool = False,
    ) -> StageReference:
        """Insert a new stage so that it executes immediately after ``target``.

        The new stage depends on ``target``. Every existing stage that previously
        depended on ``target`` is rewired to depend on the new stage instead, so
        downstream stages observe the new stage's output rather than ``target``'s.

        For a side-branch stage that depends on ``target`` without displacing
        ``target``'s existing dependents, use ``register_stage`` followed by
        ``add_dependency`` instead.

        Args:
            target: Name of the stage the new stage should run immediately after.
            stage_fn: Callable implementing the new stage.
            name: Unique name to register the new stage under.
            capture_stage_output: Whether the new stage's output should be captured
                in the pipeline results.

        Returns:
            The :class:`StageReference` for the newly registered stage.

        Raises:
            KeyError: if ``target`` is unknown.
            ValueError: if ``name`` is already registered.
        """
        target_ref = self.get_stage(target)
        existing_dependents = self._dependents_of(target_ref)
        new_stage = self.register_stage(stage_fn, name, capture_stage_output)
        new_stage.depends_on(target_ref)
        for dependent in existing_dependents:
            dependent._remove_dependency(target_ref)
            dependent.depends_on(new_stage)
        return new_stage

    def replace_stage(
        self,
        target: str,
        stage_fn: Callable[..., Any],
        name: str | None = None,
        *,
        capture_stage_output: bool | None = None,
    ) -> StageReference:
        """Replace an existing stage in place, preserving its position in the graph.

        The replacement inherits ``target``'s dependencies, and every existing
        stage that depended on ``target`` is rewired to depend on the replacement.
        If ``target`` was a result-capture stage or an evaluation stage, the
        replacement takes its place in those collections too.

        Args:
            target: Name of the stage to replace.
            stage_fn: New stage callable.
            name: Optional new name. Defaults to ``target``'s name (drop-in swap).
            capture_stage_output: Whether the replacement should capture its
                output. ``None`` (default) inherits the target's setting.

        Returns:
            The :class:`StageReference` for the replacement stage.

        Raises:
            KeyError: if ``target`` is unknown.
            ValueError: if ``name`` is provided and already in use by another stage.
        """
        target_ref = self.get_stage(target)
        new_name = name if name is not None else target_ref.name

        if new_name != target_ref.name and new_name in self._stages.values():
            msg = f"Stage name '{new_name}' is already registered"
            raise ValueError(msg)

        was_eval = target_ref in self._evaluation_stages
        was_captured = target_ref in self._result_stages
        capture = capture_stage_output if capture_stage_output is not None else was_captured

        inherited = list(target_ref._dependencies)
        dependents = self._dependents_of(target_ref)

        del self._stages[target_ref]
        self._result_stages.pop(target_ref, None)
        self._evaluation_stages.pop(target_ref, None)

        if was_eval:
            new_stage = self.register_eval_stage(stage_fn, new_name, capture)
        else:
            new_stage = self.register_stage(stage_fn, new_name, capture)
        new_stage.depends_on(*inherited)

        for dependent in dependents:
            dependent._remove_dependency(target_ref)
            dependent.depends_on(new_stage)

        return new_stage

    def add_dependency(self, stage: str, dependency: str) -> None:
        """Add ``dependency`` as a prerequisite of ``stage``.

        Idempotent: adding an existing edge is a no-op. Raises if the edge would
        introduce a cycle so the failure surfaces at the call site rather than at
        build time.

        Args:
            stage: Name of the stage that will depend on ``dependency``.
            dependency: Name of the stage to add as a prerequisite.

        Raises:
            KeyError: if either stage name is unknown.
            ValueError: if the new edge would introduce a cycle.
        """
        target = self.get_stage(stage)
        dep = self.get_stage(dependency)
        if self._reaches(dep, target):
            msg = f"Adding dependency '{dependency}' to stage '{stage}' would introduce a cycle"
            raise ValueError(msg)
        target.depends_on(dep)

    def remove_dependency(self, stage: str, dependency: str) -> None:
        """Remove the ``stage`` -> ``dependency`` edge.

        Args:
            stage: Name of the stage whose dependency should be removed.
            dependency: Name of the dependency to detach from ``stage``.

        Raises:
            KeyError: if either stage name is unknown, or if the edge does not exist.
        """
        target = self.get_stage(stage)
        dep = self.get_stage(dependency)
        target._remove_dependency(dep)

    def _dependents_of(self, target: StageReference) -> list[StageReference]:
        return [stage for stage in self._stages if target in stage._dependencies]

    def _reaches(self, src: StageReference, dst: StageReference) -> bool:
        """Whether ``dst`` is reachable from ``src`` along dependency edges."""
        stack: list[StageReference] = [src]
        seen: set[StageReference] = set()
        while stack:
            current = stack.pop()
            if current is dst:
                return True
            if current in seen:
                continue
            seen.add(current)
            stack.extend(current._dependencies)
        return False

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
        if target_stage is not None and target_stage not in self._stages:
            msg = f"Target stage '{target_stage}' is not a known stage"
            raise ValueError(msg)

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
