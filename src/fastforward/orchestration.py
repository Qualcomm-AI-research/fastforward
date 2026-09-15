# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
from __future__ import annotations

import logging

from collections.abc import Mapping
from typing import Any

import torch

from fastforward._orchestration import registry
from fastforward._orchestration.graph_module import (
    GraphModule,
    reduce_resolution,
)
from fastforward._orchestration.instruction_engine import ActivationBundle as ActivationBundle
from fastforward._orchestration.instruction_engine import (
    InstructionEngine,
    InstructionPass,
    InstructionPasses,
    InstructionProgram,
    OffloadingStrategy,
    lifetime_management_pass,
)
from fastforward._orchestration.registry import Algorithm as Algorithm
from fastforward._orchestration.registry import AlgorithmSpec as AlgorithmSpec
from fastforward._orchestration.registry import Selector as Selector
from fastforward._orchestration.registry import register as register
from fastforward._orchestration.registry import resolve as resolve
from fastforward._orchestration.scheduler import schedule
from fastforward._orchestration.trace import trace

__all__ = ["layerwise_optimize"]

logger = logging.getLogger(__name__)


class _GraphExecutionContext:
    """Context manager that temporarily swaps a graph's execution program and engine.

    On enter, the graph's `_program` and `_engine` are replaced with the provided
    values. On exit, the original state is restored.

    Args:
        graph: The GraphModule whose execution state will be temporarily replaced.
        program: The program to install for the duration of the context.
        engine: The engine to install for the duration of the context.
    """

    def __init__(
        self,
        graph: GraphModule,
        program: InstructionProgram,
        engine: InstructionEngine,
    ) -> None:
        self._graph = graph
        self._original_program = graph._program
        self._original_engine = graph._engine
        graph._program = program
        graph._engine = engine

    def __enter__(self) -> GraphModule:
        return self._graph

    def __exit__(self, *args: Any) -> None:
        self._graph._program = self._original_program
        self._graph._engine = self._original_engine


class _ExecutionContext(_GraphExecutionContext):
    """Execution context that applies passes and executes the resulting program."""

    def __init__(
        self,
        graph: GraphModule,
        program: InstructionProgram,
        passes: list[InstructionPass] | None = None,
        offloading: OffloadingStrategy | None = None,
    ) -> None:
        all_passes = list(passes or [])
        if offloading is not None:
            all_passes.append(offloading.create_instruction_pass(graph))

        program = InstructionPasses.apply(program, all_passes)

        super().__init__(graph, program, InstructionEngine())


def layerwise_optimize(
    model: torch.nn.Module,
    data: Any,
    algorithm: registry.Algorithm | registry.AlgorithmSpec | list[registry.AlgorithmSpec],
    *,
    targets: registry.TargetType | None = None,
    graph: GraphModule | None = None,
    sample_args: tuple[Any, ...] = (),
    sample_kwargs: Mapping[str, Any] | None = None,
    offloading: OffloadingStrategy | None = None,
) -> None:
    """Run layer-wise optimization on a model.

    Traces the model, resolves targets, reduces the graph to the optimization path,
    schedules an instruction program, applies passes, and executes with optional offloading.

    When `graph` is None the model is traced, which needs one example input: give it through
    `sample_args`, `sample_kwargs`, or both.

    Args:
        model: The model to optimize.
        data: Calibration data to run through the model.
        algorithm: The optimization algorithm. Accepts a callable (looked up in the
            registry), a single `AlgorithmSpec`, or a list of specs.
        targets: Override which modules to target (uses registry default if None).
            Cannot be combined with explicit AlgorithmSpec(s).
        graph: Pre-built GraphModule (traces model if None).
        sample_args: Positional example inputs for tracing.
        sample_kwargs: Keyword example inputs for tracing.
        offloading: Optional strategy for device offloading during execution.

    Raises:
        TypeError: If `graph` is None and no example input is given, or if `graph` and an
            example input are both given.
    """
    # (1) Trace if no static graph provided
    has_sample = bool(sample_args) or bool(sample_kwargs)
    if graph is None and not has_sample:
        msg = (
            "layerwise_optimize needs an example input to trace the model. Pass sample_args "
            "and/or sample_kwargs, or pass an already traced graph through graph=."
        )
        raise TypeError(msg)
    if graph is not None and has_sample:
        msg = "Cannot combine graph= with sample_args/sample_kwargs."
        raise TypeError(msg)

    sample_kwargs = sample_kwargs or {}

    if graph is None:
        graph = trace(model, *sample_args, **sample_kwargs)

    # (2) Resolve targets and reduce graph
    match algorithm:
        case registry.AlgorithmSpec() | [*_] if targets is not None:
            msg = "Cannot combine targets= with explicit AlgorithmSpec(s)."
            raise TypeError(msg)
        case [*specs]:
            optimization_specs = registry.resolve(model, specs=specs)
        case registry.AlgorithmSpec():
            optimization_specs = registry.resolve(model, specs=[algorithm])
        case algorithm if callable(algorithm):
            with registry.override(algorithm, targets):
                optimization_specs = registry.resolve(model, algorithm=algorithm)
        case _:
            msg = f"Expected an Algorithm, AlgorithmSpec, or list of AlgorithmSpec; got {type(algorithm).__name__}."  # type: ignore[unreachable]
            raise TypeError(msg)

    graph, optimization_specs = reduce_resolution(graph, optimization_specs)

    # (3) Schedule instruction program
    program = schedule(graph, optimization_specs)

    # (4) Execute
    passes: list[InstructionPass] = [lifetime_management_pass]

    with _ExecutionContext(graph, program, passes=passes, offloading=offloading):
        graph(data)
