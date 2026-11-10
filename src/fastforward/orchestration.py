# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
from __future__ import annotations

import dataclasses
import logging

from collections.abc import Iterable, Iterator, Mapping, Sequence
from itertools import islice
from typing import Any

import torch

from fastforward._orchestration import registry
from fastforward._orchestration.data_flow import FlowOp as FlowOp
from fastforward._orchestration.data_flow import Gradient as Gradient
from fastforward._orchestration.data_flow import InputActivations as InputActivations
from fastforward._orchestration.data_flow import ModelOutputs as ModelOutputs
from fastforward._orchestration.data_flow import OutputActivations as OutputActivations
from fastforward._orchestration.graph_module import (
    GraphModule,
    _sanitize_inputs,
    reduce_resolution,
)
from fastforward._orchestration.instruction_engine import ActivationBundle as ActivationBundle
from fastforward._orchestration.instruction_engine import (
    InstructionEngine,
    InstructionPass,
    InstructionPasses,
    InstructionProgram,
    OffloadingStrategy,
    ProgramInputs,
    lifetime_management_pass,
)
from fastforward._orchestration.instruction_engine import Offload as Offload
from fastforward._orchestration.registry import Algorithm as Algorithm
from fastforward._orchestration.registry import AlgorithmSpec as AlgorithmSpec
from fastforward._orchestration.registry import Selector as Selector
from fastforward._orchestration.registry import register as register
from fastforward._orchestration.registry import resolve as resolve
from fastforward._orchestration.scheduler import schedule
from fastforward._orchestration.trace import trace

__all__ = ["layerwise_optimize"]

logger = logging.getLogger(__name__)


def _batched(data: Iterable[Any], chunk_size: int) -> Iterator[tuple[Any, ...]]:
    """Lazily batch data, based on itertools.batched (Python 3.12+ only)."""
    chunk = iter(data)
    while batch := tuple(islice(chunk, chunk_size)):
        yield batch


def _chunked_inputs(
    data: Iterable[Any],
    chunk_size: int,
    input_names: Sequence[str],
) -> Iterator[ProgramInputs]:
    """Yield each group of calibration items as one program input.

    Each group goes through `_sanitize_inputs`, so items such as dicts bind to the inputs
    of the graph the same way as in a run without chunks.
    """
    for chunk in _batched(data, chunk_size):
        args, kwargs = _sanitize_inputs(input_names, (list(chunk),), {})
        yield ProgramInputs(args=args, kwargs=kwargs)


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


@dataclasses.dataclass(frozen=True, kw_only=True)
class OrchestrationConfig:
    """How `layerwise_optimize` configures any orchestration optimization.

    Args:
        offloading: Where weights and activations rest between uses. `None` keeps them
            where they are at runtime. See also `Offload`.

        grad_checkpoint: The modules for which gradient checkpointing is active. Accepts the
            same forms as the `targets` argument of `layerwise_optimize`, for example
            `ff.mpath.query("**/layers/*")`. A module that encloses no target is ignored.

        chunk_size: The number of calibration items grouped into each model input.
            When set, `data` is consumed lazily and the chunk-wise part of the
            instruction program runs once for each group. The same register is retained
            between groups, allowing operations to accumulate results across the full
            dataset. The remaining instructions, including the final optimization, run
            once after all groups have been processed.

    Raises:
        ValueError: If `chunk_size` is less than one.
    """

    offloading: OffloadingStrategy | None = None
    grad_checkpoint: registry.TargetType | registry.Selector | None = None
    chunk_size: int | None = None

    def __post_init__(self) -> None:
        if self.chunk_size is not None and self.chunk_size < 1:
            msg = f"chunk_size must be at least one, got {self.chunk_size}."
            raise ValueError(msg)


def layerwise_optimize(
    model: torch.nn.Module,
    data: Any,
    algorithm: registry.Algorithm | registry.AlgorithmSpec | list[registry.AlgorithmSpec],
    *,
    targets: registry.TargetType | None = None,
    config: OrchestrationConfig = OrchestrationConfig(),
    graph: GraphModule | None = None,
    sample_args: tuple[Any, ...] = (),
    sample_kwargs: Mapping[str, Any] | None = None,
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
            registry), a single `AlgorithmSpec`, or a list of specs. The specs on one region
            run one after the other, in list order. Two specs must target equal or
            disjoint regions.
        targets: Override which modules to target. If None, uses the default targets the
            algorithm was registered with. Cannot be combined with explicit AlgorithmSpec(s).
        config: Orchestration settings (offloading strategy, gradient checkpointing, chunk size).
        graph: Pre-built GraphModule (traces model if None).
        sample_args: Positional example inputs for tracing.
        sample_kwargs: Keyword example inputs for tracing.

    Raises:
        TypeError: If `graph` is None and no example input is given, if `graph` and an
            example input are both given, or if `config.chunk_size` is set and `data` is
            one tensor.
        ValueError: If `config.chunk_size` is set and an algorithm cannot run on chunks of
            the data, or no algorithm reads any data. Also if two specs share part of a
            region.
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

    if config.chunk_size is not None and isinstance(data, torch.Tensor):
        # Without chunks a tensor is one batch, so chunking it along its rows would differ.
        msg = "Cannot chunk one tensor. Pass an iterable of batches, or do not set chunk_size."
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
    grad_checkpoint: list[torch.nn.Module] = []
    if config.grad_checkpoint is not None:
        grad_checkpoint = [
            region
            for region in registry.normalize(config.grad_checkpoint).resolve(model)
            if isinstance(region, torch.nn.Module)
        ]
    program = schedule(
        graph, optimization_specs, checkpoint=grad_checkpoint, chunked=config.chunk_size is not None
    )

    # (4) Execute
    passes: list[InstructionPass] = [lifetime_management_pass]

    with _ExecutionContext(graph, program, passes=passes, offloading=config.offloading):
        if config.chunk_size is None:
            graph(data)
        else:
            inputs = _chunked_inputs(data, config.chunk_size, graph.input_names)
            InstructionEngine().run_incrementally(graph._program, inputs)
