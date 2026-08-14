# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import attrs
import pytest
import torch

from fastforward._orchestration.data_flow import DataFlow, InputActivations, OutputActivations
from fastforward._orchestration.graph_module import GraphModule, NodeRef
from fastforward._orchestration.scheduler import bind_flows
from fastforward._orchestration.trace import _MIN_TORCH_VERSION, trace
from packaging.version import Version

from ._models import TwoLinear

pytestmark = pytest.mark.skipif(
    Version(torch.__version__.split("+", 1)[0]) < _MIN_TORCH_VERSION,
    reason=f"requires PyTorch >= {_MIN_TORCH_VERSION}",
)


def _traced_two_linear(model: TwoLinear) -> GraphModule:
    """Trace `model` so its nodes can be referenced by `bind_flows`."""
    return trace(model.eval(), torch.randn(2, 8))


def _branching_kwarg_graph() -> tuple[GraphModule, NodeRef]:
    """A graph whose region reads two branches positionally plus one keyword.

    Tracing cannot produce this shape: a keyword-only forward argument becomes a
    graph input, not a node kwarg, so every traced node has empty kwargs. Build
    it by hand to exercise the multi-read and keyword paths.

    Returns:
        The graph and a ref to its region.
    """
    graph = GraphModule()
    inp = graph.add_input("x")
    left = graph.add_node("left", torch.nn.Linear(8, 8), [inp])
    right = graph.add_node("right", torch.nn.Linear(8, 8), [inp])
    region = graph.add_node("region", torch.nn.Linear(8, 8), [left, right], {"extra": inp})
    graph.add_output(region)
    return graph, region


def test_input_activations_nodes_exclude_region(two_linear: TwoLinear) -> None:
    # GIVEN a traced model and a region with predecessors (fc2)
    graph = _traced_two_linear(two_linear)
    region = graph.node_ref(two_linear.fc2)

    # WHEN binding an InputActivations flow on that region
    [plan] = bind_flows(graph, region, [InputActivations("original")])

    # THEN the bound nodes cover the predecessors but not the region itself
    assert region not in plan.nodes
    assert graph.node_ref(two_linear.fc1) in plan.nodes


def test_output_activations_nodes_end_at_region(two_linear: TwoLinear) -> None:
    # GIVEN a traced model and a region with predecessors (fc2)
    graph = _traced_two_linear(two_linear)
    region = graph.node_ref(two_linear.fc2)

    # WHEN binding an OutputActivations flow on that region
    [plan] = bind_flows(graph, region, [OutputActivations("original")])

    # THEN the last of the bound nodes is the region
    assert plan.nodes[-1] == region


def test_input_nodes_are_output_nodes_without_region(two_linear: TwoLinear) -> None:
    # GIVEN a traced model and a region with predecessors (fc2)
    graph = _traced_two_linear(two_linear)
    region = graph.node_ref(two_linear.fc2)

    # WHEN binding both an InputActivations and an OutputActivations flow on
    # the same region in a single call
    input_plan, output_plan = bind_flows(
        graph,
        region,
        [InputActivations("original"), OutputActivations("original")],
    )

    # THEN the input plan's nodes are the output plan's nodes minus the last one
    assert input_plan.nodes == output_plan.nodes[:-1]
    # THEN the last node in the output plan is the region
    assert output_plan.nodes[-1] == region


def test_input_activations_on_first_layer_runs_nothing(two_linear: TwoLinear) -> None:
    # GIVEN a traced model and the FIRST layer (fc1), which has no input nodes
    graph = _traced_two_linear(two_linear)
    region = graph.node_ref(two_linear.fc1)

    # WHEN binding an InputActivations flow on that region
    [plan] = bind_flows(graph, region, [InputActivations("original")])

    # THEN nothing has to run (the data is the graph's own input)
    assert plan.nodes == ()


def test_input_activations_reads_are_region_args_and_kwargs() -> None:
    # GIVEN a region reading two branches positionally plus one keyword
    graph, region = _branching_kwarg_graph()
    node = graph.node(region)
    assert node.kwargs  # guard: a traced region has none, which made this vacuous

    # WHEN generating a plan to optimize that region w.r.t. its inputs
    [plan] = bind_flows(graph, region, [InputActivations("original")])

    # THEN the plan gives us the args and the kwargs arriving at the region
    assert plan.reads == tuple(node.args)
    assert plan.read_kwargs == {"extra": graph._inputs["x"]}


def test_output_activations_reads_are_region_itself() -> None:
    # GIVEN a region that carries a keyword argument
    graph, region = _branching_kwarg_graph()
    assert graph.node(region).kwargs  # guard: kwargs must exist to be dropped

    # WHEN generating a plan to optimize that region w.r.t. its output
    [plan] = bind_flows(graph, region, [OutputActivations("original")])

    # THEN only the region's own output is read, and its kwargs are dropped
    assert plan.reads == (region,)
    assert plan.read_kwargs == {}


def test_input_activations_reads_every_branch_not_just_the_last() -> None:
    # GIVEN a region fed by two independent branches
    graph, region = _branching_kwarg_graph()
    left, right = tuple(graph.node(region).args)

    # WHEN generating a plan to optimize that region w.r.t. its inputs
    [plan] = bind_flows(graph, region, [InputActivations("original")])

    # THEN both branches are read, so the reads are not a single node
    assert plan.reads == (left, right)
    # THEN the last bound node is only one of them, so it is not the read point
    assert plan.nodes[-1] == right
    assert set(plan.reads) - {plan.nodes[-1]} == {left}


def test_bind_flows_rejects_unknown_dataflow_type(two_linear: TwoLinear) -> None:
    # GIVEN a custom DataFlow subclass not handled by bind_flows
    @attrs.define(frozen=True)
    class UnknownFlow(DataFlow):
        pass

    graph = _traced_two_linear(two_linear)
    region = graph.node_ref(two_linear.fc2)

    # WHEN / THEN planning for an unknown flow type raises TypeError
    with pytest.raises(TypeError, match="unsupported DataFlow type"):
        bind_flows(graph, region, [UnknownFlow("original")])
