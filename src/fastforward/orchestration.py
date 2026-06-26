# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
from __future__ import annotations

import logging

from typing import Any

import torch

from fastforward._orchestration import registry
from fastforward._orchestration.graph_module import (
    GraphModule,
    _GraphExecutionContext,
    reduce_resolution,
)
from fastforward._orchestration.instruction_engine import ActivationBundle as ActivationBundle
from fastforward._orchestration.instruction_engine import (
    InstructionEngine,
    InstructionPass,
    InstructionPasses,
    InstructionProgram,
    InstructionScheduler,
    OffloadingStrategy,
    lifetime_management_pass,
    optimization_only_pass,
)
from fastforward._orchestration.registry import Algorithm as Algorithm
from fastforward._orchestration.registry import Selector as Selector
from fastforward._orchestration.registry import register as register
from fastforward._orchestration.registry import resolve as resolve
from fastforward._orchestration.trace import trace

logger = logging.getLogger(__name__)


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
    algorithm: registry.Algorithm,
    *,
    targets: registry.TargetType | None = None,
    graph: GraphModule | None = None,
    offloading: OffloadingStrategy | None = None,
    **kwargs: Any,
) -> None:
    """Run layer-wise optimization on a model.

    Traces the model, resolves targets, reduces the graph to the optimization path,
    schedules an instruction program, applies passes, and executes with optional offloading.

    Args:
        model: The model to optimize.
        data: Calibration data to run through the model.
        algorithm: The optimization algorithm to apply. It is invoked once per
            target region as ``algorithm(module, dataset)``, where ``dataset`` is
            an `ActivationBundle` of that module's captured calibration inputs.
            Iterating ``dataset`` yields one `(args, kwargs)` pair per batch,
            ready to replay as ``module(*args, **kwargs)``. Note that ``args`` is
            often a singleton tuple (e.g. ``(x,)``) and ``kwargs`` is often empty
            (``{}``); see `ActivationBundle` for the full contract.
        targets: Override which modules to target (uses registry default if None).
        graph: Pre-built GraphModule (traces model if None).
        offloading: Optional strategy for device offloading during execution.
        **kwargs: Additional arguments forwarded to trace.
    """
    # (1) Trace if no static graph provided
    if graph is None:
        example_input = data[0] if isinstance(data, list) else data
        graph = trace(model, example_input, **kwargs)

    # (2) Resolve targets and reduce graph
    with registry.override(algorithm, targets):
        optimization_specs = registry.resolve(model, algorithm)
    graph = reduce_resolution(graph, optimization_specs)

    # (3) Schedule instruction program
    program = InstructionScheduler().schedule(graph)

    # (4) Execute
    passes: list[InstructionPass] = [optimization_only_pass, lifetime_management_pass]

    with _ExecutionContext(graph, program, passes=passes, offloading=offloading):
        graph(data, **kwargs)
