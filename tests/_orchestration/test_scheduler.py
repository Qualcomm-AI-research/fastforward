# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import pytest
import torch

from fastforward._orchestration.data_flow import InputActivations, OutputActivations
from fastforward._orchestration.graph_module import GraphModule, Region
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


def test_input_activations_nodes_exclude_region(two_linear: TwoLinear) -> None:
    # GIVEN a traced model and a region with predecessors (fc2)
    graph = _traced_two_linear(two_linear)
    region = graph.node_ref(two_linear.fc2)

    # WHEN binding an InputActivations flow on that region
    [plan] = bind_flows(graph, region, [InputActivations.make("original")])

    # THEN the bound nodes cover the predecessors but not the region itself
    assert region not in plan.nodes
    assert graph.node_ref(two_linear.fc1) in plan.nodes


def test_output_activations_nodes_end_at_region(two_linear: TwoLinear) -> None:
    # GIVEN a traced model and a region with predecessors (fc2)
    graph = _traced_two_linear(two_linear)
    region = graph.node_ref(two_linear.fc2)

    # WHEN binding an OutputActivations flow on that region
    [plan] = bind_flows(graph, region, [OutputActivations.make("original")])

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
        [InputActivations.make("original"), OutputActivations.make("original")],
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
    [plan] = bind_flows(graph, region, [InputActivations.make("original")])

    # THEN nothing has to run (the data is the graph's own input)
    assert plan.nodes == ()


def test_non_ancestor_source_raises(two_linear: TwoLinear) -> None:
    # GIVEN a traced model and a source resolver naming a module that is NOT
    # an ancestor of the region being bound (fc2 comes after fc1, not before)
    graph = _traced_two_linear(two_linear)
    region = graph.node_ref(two_linear.fc1)

    def source(_module: Region) -> torch.nn.Module:
        return two_linear.fc2

    # WHEN binding a flow with that source
    # THEN a ValueError is raised mentioning it is not an ancestor
    with pytest.raises(ValueError, match="not an ancestor"):
        bind_flows(graph, region, [InputActivations.make("original", source=source)])
