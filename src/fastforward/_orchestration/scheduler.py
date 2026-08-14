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

from collections.abc import Mapping, Sequence

from fastforward._orchestration.data_flow import (
    DataFlow,
    FlowGenerator,
    InputActivations,
    OutputActivations,
)
from fastforward._orchestration.graph_module import (
    GraphModule,
    NodeRef,
    _BaseRef,
    ancestors,
)


@dataclasses.dataclass(frozen=True)
class FlowPlan:
    """A single `DataFlow` resolved against a concrete graph.

    A `FlowPlan` pairs a declaration with the work needed to satisfy it: the
    nodes to execute, and the refs to read once those nodes have run.

    The read points are `reads` and `read_kwargs`. For `InputActivations` these
    are the region's own args and kwargs. They may name several refs on separate
    branches, and may include graph inputs, which appear in no node. For
    `OutputActivations`, `reads` names a single ref, the region itself, and
    `read_kwargs` is empty.

    `nodes` holds every node that must run first, in dependency order. It
    contains the region for `OutputActivations` and omits it for
    `InputActivations`. It is empty when no node has to run, such as an
    `InputActivations` on a region whose args are all graph inputs.

    Args:
        flow: The declaration this plan resolves.
        nodes: The nodes to execute, in dependency order, to produce the data.
        reads: The positional refs the algorithm receives.
        read_kwargs: The keyword refs the algorithm receives.
    """

    flow: DataFlow
    nodes: tuple[NodeRef, ...]
    reads: tuple[_BaseRef, ...] = ()
    read_kwargs: Mapping[str, _BaseRef] = dataclasses.field(default_factory=dict)

    @property
    def generator(self) -> FlowGenerator:
        """The flow generator that defines execution context for this plan."""
        return self.flow.generator

    @property
    def cache(self) -> bool:
        """Whether the work done to produce this data may be reused."""
        return self.flow.cache


def bind_flows(graph: GraphModule, region: NodeRef, flows: Sequence[DataFlow]) -> list[FlowPlan]:
    """Bind each declared flow to concrete nodes on `graph`."""
    topo_index = {ref.id: i for i, ref in enumerate(graph.topo_order)}
    node = graph.node(region)

    plans: list[FlowPlan] = []
    for flow in flows:
        read_kwargs: Mapping[str, _BaseRef] = {}
        match flow:
            case InputActivations():
                node_set = _ancestors_of(graph, region, include_region=False)
                # What arrives at the region: its own arguments.
                reads = tuple(node.args)
                read_kwargs = node.kwargs
            case OutputActivations():
                node_set = _ancestors_of(graph, region, include_region=True)
                # What leaves the region: the region's own output.
                reads = (region,)
            case _:
                msg = f"unsupported DataFlow type: {type(flow).__name__}"
                raise TypeError(msg)

        ordered = sorted(node_set, key=lambda ref: topo_index[ref.id])
        plans.append(
            FlowPlan(
                flow=flow,
                nodes=tuple(ordered),
                reads=reads,
                read_kwargs=read_kwargs,
            )
        )
    return plans


def _ancestors_of(graph: GraphModule, region: NodeRef, *, include_region: bool) -> set[NodeRef]:
    """Collect the ancestor set of `region`.

    Returns the raw set; the caller is responsible for ordering.
    """
    nodes, _ = ancestors(graph, region)

    if not include_region:
        nodes -= {region}

    return nodes
