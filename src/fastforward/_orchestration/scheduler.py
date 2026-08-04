# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Scheduling: compile a GraphModule and its optimizations into an InstructionProgram.

Given a `GraphModule` and an optional list of specs (each an algorithm to run
at a region, together with the data it needs to see), the scheduler produces
a sequence of instructions that executes the graph and dispatches the
optimizations at the right points. When no specs are supplied, the program is
just a plain forward pass.

The design is generate-then-prune: expand each region's requirements
independently as if it were running alone, then dedup and drop redundant work.
Requirements are declared as `Flow`s -- a small vocabulary for saying "the
inputs of my target, taken from an original pass" or the like. Binding
resolves those declarations against the graph into `FlowPlan`s.

Currently: binding only. Generate/prune and the instruction program arrive later.
"""

from __future__ import annotations

import dataclasses

from collections.abc import Sequence
from typing import assert_never

import torch

from fastforward._orchestration.data_flow import (
    Flow,
    FlowSource,
    InputActivations,
    OutputActivations,
)
from fastforward._orchestration.graph_module import (
    GraphModule,
    NodeRef,
    ancestors,
    topological_sort,
)


@dataclasses.dataclass(frozen=True)
class FlowPlan:
    """A flow resolved against a concrete graph.

    Wraps the original `flow` alongside the `nodes` that produce, in dependency
    order, its data. `nodes[-1]` is where the flow's result is read: for
    `InputActivations` that is the last predecessor of the region; for
    `OutputActivations` it is the region itself.

    `nodes` is empty when nothing has to execute first. An `InputActivations` on
    a region with no predecessors reads the graph's own inputs, so there is no
    `nodes[-1]` to read at.

    Args:
        flow: The original declaration this plan resolves.
        nodes: Nodes to execute in dependency order to produce the data.
    """

    flow: Flow
    nodes: tuple[NodeRef, ...]

    @property
    def cache(self) -> bool:
        """Whether the work done to produce this data may be reused."""
        return self.flow.cache


def bind_flows(graph: GraphModule, region: NodeRef, flows: Sequence[Flow]) -> list[FlowPlan]:
    """Bind each declared flow to concrete nodes on `graph`.

    Dispatches on the concrete `Flow` subclass; the fall-through is
    `assert_never` so mypy points here when a new subclass is missed.
    """
    plans: list[FlowPlan] = []
    for flow in flows:
        match flow:
            case InputActivations():
                # Reads the region's inputs, so the region itself must not run.
                nodes = _run_upto(graph, region, flow.source, include_region=False)
            case OutputActivations():
                # Reads the region's output, so the region runs last.
                nodes = _run_upto(graph, region, flow.source, include_region=True)
            case _:
                assert_never(flow)
        plans.append(FlowPlan(flow=flow, nodes=nodes))
    return plans


def _run_upto(
    graph: GraphModule,
    region: NodeRef,
    source: FlowSource | None,
    *,
    include_region: bool,
) -> tuple[NodeRef, ...]:
    """Nodes that must run to produce the data arriving at (or leaving) `region`.

    When `source` is `None` the walk is unbounded: every ancestor of the region.
    When `source` names a predecessor, the walk stops there. `include_region`
    keeps or drops the region itself from the result.

    The result is empty when nothing has to run first -- dropping the region
    from a region that has no predecessors of its own leaves nothing behind.
    """
    stop: NodeRef | None = None
    if source is not None:
        region_module = graph.node(region).target
        if not isinstance(region_module, torch.nn.Module):
            msg = f"Region {region.name!r} is not a module; cannot apply resolver."
            raise TypeError(msg)
        stop = graph.node_ref(source(region_module))

    nodes, reached = ancestors(graph, region, stop=stop)
    if stop is not None and not reached:
        msg = (
            f"Flow source {stop.name!r} is not an ancestor of "
            f"{region.name!r}: no path from the source to the region."
        )
        raise ValueError(msg)

    if not include_region:
        nodes = nodes - {region}

    return tuple(topological_sort(graph, nodes))
