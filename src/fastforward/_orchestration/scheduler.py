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

from fastforward._orchestration.data_flow import (
    Flow,
    FlowGenerator,
    InputActivations,
    OutputActivations,
)
from fastforward._orchestration.graph_module import (
    GraphModule,
    NodeRef,
    ancestors,
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
    def generator(self) -> FlowGenerator:
        """The flow generator that defines execution context for this plan."""
        return self.flow.generator

    @property
    def cache(self) -> bool:
        """Whether the work done to produce this data may be reused."""
        return self.flow.cache


def bind_flows(graph: GraphModule, region: NodeRef, flows: Sequence[Flow]) -> list[FlowPlan]:
    """Bind each declared flow to concrete nodes on `graph`."""
    topo_index = {ref.id: i for i, ref in enumerate(graph.topo_order)}

    plans: list[FlowPlan] = []
    for flow in flows:
        match flow:
            case InputActivations():
                node_set = _ancestors_of(graph, region, include_region=False)
            case OutputActivations():
                node_set = _ancestors_of(graph, region, include_region=True)
        ordered = sorted(node_set, key=lambda ref: topo_index[ref.id])
        plans.append(FlowPlan(flow=flow, nodes=tuple(ordered)))
    return plans


def _ancestors_of(graph: GraphModule, region: NodeRef, *, include_region: bool) -> set[NodeRef]:
    """Collect the ancestor set of `region`.

    Returns the raw set; the caller is responsible for ordering.
    """
    nodes, _ = ancestors(graph, region)

    if not include_region:
        nodes -= {region}

    return nodes
