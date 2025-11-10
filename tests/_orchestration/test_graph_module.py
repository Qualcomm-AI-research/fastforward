# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


import functools
import uuid

from typing import Any, Iterable

import pytest
import torch

from fastforward._orchestration.graph_module import (
    Const,
    Direction,
    GraphModule,
    InputRef,
    LocalOptimizer,
    NodeRef,
    SubgraphSpec,
    create_subgraph,
    find_nodes_on_path,
    find_reachable_nodes,
    remap_subgraph_reference,
)
from fastforward._orchestration.instruction_engine import (
    ActivationDataset,
    CallModule,
)


class Add(torch.nn.Module):
    """Placeholder due to lack of torch.nn.Add()."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Add two tensors together and return the result."""
        return x + y

    def to_graph_module(self) -> GraphModule:
        """Transform 'Add' to a GraphModule."""
        graph = GraphModule()
        x = graph.add_input("x")
        y = graph.add_input("y")
        output = graph.add_node("add", self, [x, y])
        graph.add_output(output)
        return graph


class Residual(torch.nn.Module):
    """Simple residual forward pass."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(5, 5)

    def forward(self, x: torch.Tensor) -> Any:
        """Forward pass of MyResidual."""
        h = self.linear(x)
        h = torch.nn.ReLU()(h)
        return x + h

    def to_graph_module(self) -> GraphModule:
        """Transform 'Residual' to a GraphModule."""
        graph = GraphModule()
        input = graph.add_input("input")
        linear = graph.add_node("linear", self.linear, [input])
        relu = graph.add_node("relu", torch.nn.ReLU(), [linear])
        (add,) = graph.add_subgraph("add", Add().to_graph_module(), [input, relu])
        graph.add_output(add)
        return graph


class Model(torch.nn.Module):
    """Simple Module class for demonstrative purposes."""

    def __init__(self) -> None:
        super().__init__()
        self.residual_1 = Residual()
        self.residual_2 = Residual()

    def forward(self, x: torch.Tensor) -> Any:
        """Forward pass of MyModel."""
        h = self.residual_1(x)
        h = self.residual_2(h)
        return torch.nn.Sigmoid()(h)

    def to_graph_module(self) -> GraphModule:
        """Transform 'MyModel' to a GraphModule."""
        graph = GraphModule()
        input = graph.add_input("input")
        (residual_1,) = graph.add_subgraph("residual_1", self.residual_1.to_graph_module(), [input])
        (residual_2,) = graph.add_subgraph(
            "residual_2", self.residual_2.to_graph_module(), [residual_1]
        )
        sigmoid = graph.add_node("sigmoid", torch.nn.Sigmoid(), [residual_2])
        graph.add_output(sigmoid)
        return graph


def test_graph_module_forward_pass() -> None:
    # GIVEN a PyTorch model and its equivalent GraphModule representation
    model = Model()
    graph = model.to_graph_module()
    x = torch.randn(1, 5)

    # WHEN we run both the original model and the graph module engine
    model_output = model(x)
    engine_output = graph(x)

    # THEN the outputs should be identical
    torch.testing.assert_close(model_output, engine_output)


class MultiOutputModel(torch.nn.Module):
    """Model with multiple outputs."""

    def __init__(self) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(5, 3)
        self.linear2 = torch.nn.Linear(5, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning two outputs."""
        out1 = torch.nn.ReLU()(self.linear1(x))
        out2 = torch.nn.Sigmoid()(self.linear2(x))
        return out1, out2

    def to_graph_module(self) -> GraphModule:
        """Convert to GraphModule with multiple outputs."""
        graph = GraphModule()
        input = graph.add_input("input")
        linear1_out = graph.add_node("linear1", self.linear1, args=[input])
        relu_out = graph.add_node("relu", torch.nn.ReLU(), args=[linear1_out])
        linear2_out = graph.add_node("linear2", self.linear2, args=[input])
        sigmoid_out = graph.add_node("sigmoid", torch.nn.Sigmoid(), args=[linear2_out])
        graph.add_output(relu_out, sigmoid_out)
        return graph


def test_multi_output_graph_module() -> None:
    # GIVEN a model with multiple outputs and its corresponding GraphModule
    model = MultiOutputModel()
    graph = model.to_graph_module()
    x = torch.randn(1, 5)

    # WHEN we run both the original model and the graph module
    model_out1, model_out2 = model(x)
    graph_out1, graph_out2 = graph(x)

    # THEN the outputs should be identical
    torch.testing.assert_close(model_out1, graph_out1)
    torch.testing.assert_close(model_out2, graph_out2)


class ConstReturn(torch.nn.Module):
    """A model that returns only a constant value."""

    def forward(self, _: Any, c: Any) -> Any:
        return c


def test_const_input_type() -> None:
    # GIVEN a GraphModule with a constant value node
    c = torch.randn(5)
    graph = GraphModule()
    input = graph.add_input("input")
    const_node = graph.add_node("const_node", ConstReturn(), args=[input, Const(c)])
    graph.add_output(const_node)

    # WHEN we execute the graph module
    graph_output = graph(None)

    # THEN the output should be the constant value
    assert graph_output is c


class ConstReturnKwargs(torch.nn.Module):
    """A model that returns only a keyword argument."""

    def forward(self, _: Any, const_kwarg: Any = None) -> Any:
        return const_kwarg


def test_const_type_kwargs() -> None:
    # GIVEN a GraphModule with a constant value passed as a keyword argument
    c = torch.randn(5)
    graph = GraphModule()
    input = graph.add_input("input")
    const_node = graph.add_node(
        "const_node",
        ConstReturnKwargs(),
        args=[input],
        kwargs={"const_kwarg": Const(c)},
    )
    graph.add_output(const_node)

    # WHEN we execute the graph module
    graph_output = graph(None)

    # THEN the output should be the constant value
    assert graph_output is c


def test_keyword_binding_for_subgraph() -> None:
    # GIVEN a GraphModule of Add() and sample inputs
    add_subgraph = Add().to_graph_module()
    input_x = torch.randn(5)
    input_y = torch.randn(5)

    # GIVEN a GraphModule with Add as a subgraph where we bind a keyword argument
    graph = GraphModule()
    input = graph.add_input("input")
    (add,) = graph.add_subgraph("add", add_subgraph, [input], {"y": Const(input_y)})
    graph.add_output(add)

    # WHEN we execute the graph module
    result = graph(input_x)

    # THEN the output should be the sum of input_x and input_y
    torch.testing.assert_close(result, input_x + input_y)


def test_create_subgraph_functional_equivalence() -> None:
    # GIVEN a Model and GraphModule equivalent
    model = Model()
    graph = model.to_graph_module()

    # GIVEN the minimal node set lying in the GraphModule
    path_nodes = find_nodes_on_path(
        graph,
        graph.node_ref(graph.residual_1.linear),
        graph.node_ref(graph.sigmoid),
    )

    # WHEN we materialise that path as a standalone GraphModule
    subgraph = create_subgraph(graph, path_nodes)

    # THEN the subgraph must only contain the nodes on the path
    assert set(subgraph._nodes) == set(node.id for node in path_nodes)

    # THEN executing the subgraph produces the same result as the parent graph
    x = torch.randn(1, 5)
    torch.testing.assert_close(graph(x), subgraph(x))


def test_find_reachable_nodes_happy_path() -> None:
    # GIVEN a GraphModule and its plan
    graph = Model().to_graph_module()

    # WHEN we collect nodes reachable forward from residual_1.linear
    fwd = find_reachable_nodes(
        graph,
        graph.node_ref(graph.residual_1.linear),
        direction=Direction.FORWARD,
    )

    # THEN some expected downstream nodes are present
    assert {
        graph.node_ref(graph.residual_1.relu),
        graph.node_ref(graph.residual_2.linear),
        graph.node_ref(graph.sigmoid),
    } <= fwd

    # WHEN we collect nodes reachable backward from sigmoid
    bwd = find_reachable_nodes(
        graph,
        graph.node_ref(graph.sigmoid),
        direction=Direction.BACKWARD,
    )

    # THEN an early upstream node is included
    assert graph.node_ref(graph.residual_1.linear) in bwd

    # GIVEN a allowlist that omits intermediate nodes
    allowlist = {graph.node_ref(graph.residual_2.linear), graph.node_ref(graph.sigmoid)}

    # WHEN we traverse with the allowlist
    restricted = find_reachable_nodes(
        graph,
        graph.node_ref(graph.residual_2.linear),
        direction=Direction.FORWARD,
        allowlist=allowlist,
    )

    # THEN traversal stops at the start node
    assert restricted == {graph.node_ref(graph.residual_2.linear)}


def test_node_ref_identity_across_graphs() -> None:
    # GIVEN a small reusable sub-graph with a single Linear op
    linear_mod = torch.nn.Linear(3, 3)
    sub = GraphModule()
    x = sub.add_input("x")
    lin = sub.add_node("linear", linear_mod, [x])
    sub.add_output(lin)

    # WHEN we inline the same sub-graph into two different parent graphs
    g1 = GraphModule()
    (lin_g1,) = g1.add_subgraph("g1", sub, [g1.add_input("x")])
    g1.add_output(lin_g1)

    g2 = GraphModule()
    (lin_g2,) = g2.add_subgraph("g2", sub, [g2.add_input("x")])
    g2.add_output(lin_g2)

    # THEN the NodeRefs share UUID identity but carry different qualified names
    assert lin_g1 == lin_g2
    assert hash(lin_g1) == hash(lin_g2)
    assert lin_g1.name != lin_g2.name


def test_attribute_ref_indexing() -> None:
    # GIVEN a model that returns a tuple and a GraphModule that uses AttributeRef to access elements
    graph = GraphModule()
    model = MultiOutputModel()
    input_ref = graph.add_input("input")

    # WHEN we add a node that returns a tuple and use indexing to access individual outputs
    tuple_output = graph.add_node("tuple_model", model.to_graph_module(), [input_ref])
    first_output = tuple_output[0]  # AttributeRef to first element
    second_output = tuple_output[1]  # AttributeRef to second element

    graph.add_output(first_output, second_output)

    # WHEN we execute the graph
    x = torch.randn(1, 5)
    graph_out1, graph_out2 = graph(x)

    # THEN the outputs should match the expected computation
    model_out1, model_out2 = model(x)

    torch.testing.assert_close(graph_out1, model_out1)
    torch.testing.assert_close(graph_out2, model_out2)


def test_remap_subgraph_reference_noderef() -> None:
    # GIVEN a Noderef that could be external or internal to the subgraph
    node_id = uuid.UUID(int=0)

    # WHEN the NodeRef is external to the subgraph (has input binding)
    external_ref = NodeRef(id=node_id, name="external_node")
    input_binding = {"external_node": NodeRef(id=node_id, name="other_name")}
    remapped_external = remap_subgraph_reference(external_ref, input_binding, {}, "")

    # THEN it should be remapped through the binding
    assert remapped_external == NodeRef(id=node_id, name="other_name")

    # WHEN the NodeRef is internal to the subgraph
    internal_ref = NodeRef(id=node_id, name="internal_node")
    subgraph_nodes = {"internal_node": node_id}
    remapped_no_scope = remap_subgraph_reference(internal_ref, {}, subgraph_nodes, "")

    # THEN it should keep its original name
    assert remapped_no_scope == NodeRef(id=node_id, name="internal_node")

    # WHEN the NodeRef is internal with a scope
    remapped_with_scope = remap_subgraph_reference(internal_ref, {}, subgraph_nodes, "scope")

    # THEN it should be prefixed with the scope
    assert remapped_with_scope == NodeRef(id=node_id, name="scope.internal_node")


def test_remap_subgraph_reference_input_ref() -> None:
    # GIVEN an InputRef
    input_id = uuid.UUID(int=0)
    input_ref = InputRef(id=input_id, name="input")

    # WHEN the InputRef has a binding
    input_binding = {"input": InputRef(id=input_id, name="other_input")}
    remapped = remap_subgraph_reference(input_ref, input_binding, {}, "")

    # THEN it should be remapped via the binding
    assert remapped == InputRef(id=input_id, name="other_input")


def test_remap_subgraph_reference_attribute_ref() -> None:
    # GIVEN an AttributeRef that referencs a specific transformer layers QKV weightd
    node_id = uuid.UUID(int=0)
    model_ref = NodeRef(id=node_id, name="transformer_layers")
    attribute_ref = model_ref[0].attention.qkv_proj

    # WHEN the underlying NodeRef is external (has input binding)
    input_binding = {"transformer_layers": NodeRef(id=node_id, name="bound_model")}
    remapped_external = remap_subgraph_reference(attribute_ref, input_binding, {}, "")

    # THEN the base reference should be remapped, but the attribute chain preserved
    expected_external = NodeRef(id=node_id, name="bound_model")[0].attention.qkv_proj
    assert remapped_external == expected_external

    # WHEN the underlying NodeRef is internal with a scope
    subgraph_nodes = {"transformer_layers": node_id}
    remapped_internal = remap_subgraph_reference(attribute_ref, {}, subgraph_nodes, "prefix")

    # THEN the base reference should be scoped, but the attribute chain preserved
    expected_internal = NodeRef(id=node_id, name="prefix.transformer_layers")[0].attention.qkv_proj
    assert remapped_internal == expected_internal

    # WHEN we have an AttributeRef that has a different underlying NodeRef but the same ref structure
    alternative_model_ref = NodeRef(id=uuid.UUID(int=2), name="transformer_layers")
    alternative_attribute_ref = alternative_model_ref[0].attention.qkv_proj

    # THEN the two AttributeRefs should be unequal
    assert attribute_ref != alternative_attribute_ref


def test_local_error_opt() -> None:
    """Integration test for Local Error Optimization."""
    # GIVEN a Model with two residual blocks and its GraphModule representation
    model = Model()
    graph = model.to_graph_module()

    # GIVEN we store initial weights for comparison
    initial_residual1_weight = model.residual_1.linear.weight.data.clone()
    initial_residual2_weight = model.residual_2.linear.weight.data.clone()

    # GIVEN a simple calibration dataset
    calibration_data = [torch.randn(1, 5) for _ in range(10)]

    # GIVEN a dummy optimization function
    def dummy(module: torch.nn.Module, dataset: Iterable[torch.Tensor], lr: float) -> None:
        optim = torch.optim.SGD(params=module.parameters(), lr=lr)

        for batch in dataset:
            optim.zero_grad()

            output = module(batch)
            loss = (output**2).mean()
            loss.backward()

            optim.step()

    # GIVEN a SubgraphSpec that targets only the first residual's linear layer
    specs = [
        SubgraphSpec(
            graph.node_ref(model.residual_1.linear),
            graph.node_ref(model.residual_1.linear),
            fn=functools.partial(dummy, lr=0.1),
        )
    ]

    # WHEN we run the LocalOptimizer on the calibration data
    optimizer = LocalOptimizer(graph, specs)
    optimizer.optimize(calibration_data)

    # THEN only residual_1's linear weights should have changed
    assert not torch.allclose(initial_residual1_weight, model.residual_1.linear.weight.data)
    assert torch.allclose(initial_residual2_weight, model.residual_2.linear.weight.data)


def test_local_optimization_overlapping_specs_raises() -> None:
    """Test that LocalOptimizer rejects overlapping specs."""
    # GIVEN two SubgraphSpecs that overlap
    model = Model()
    graph = model.to_graph_module()
    specs = [
        SubgraphSpec(
            input=graph.node_ref(graph.residual_1.linear),
            output=graph.node_ref(graph.residual_1.relu),
        ),
        SubgraphSpec(
            input=graph.node_ref(graph.residual_1.linear),
            output=graph.node_ref(graph.residual_1.relu),
        ),
    ]

    # WHEN we try to create an optimizer with overlapping specs
    # THEN it should raise a ValueError
    with pytest.raises(ValueError, match="Overlapping nodes"):
        LocalOptimizer(graph, specs)


def test_call_module_single_tensor_arg() -> None:
    """Test that CallModule correctly handles a single tensor argument without unpacking tensor elements."""
    # GIVEN a simple linear module
    module = torch.nn.Linear(5, 3)

    # GIVEN a register with a single tensor batch
    input_id = uuid.uuid4()
    target_id = uuid.uuid4()
    register = {input_id: ActivationDataset([torch.randn(2, 5)])}

    # GIVEN a CallModule instruction with single arg
    instruction = CallModule(module=module, args=[input_id], kwargs={}, target=target_id)

    # WHEN we execute the instruction
    instruction.execute(register)

    # THEN the output should be computed correctly
    assert target_id in register
    output_dataset = register[target_id]
    assert len(output_dataset) == 1
    assert output_dataset.batches[0].shape == (2, 3)


def test_local_optimization_with_attribute_refs() -> None:
    """Test LocalOptimizer with AttributeRef outputs in subgraphs."""
    # GIVEN a model that returns multiple outputs
    model = MultiOutputModel()
    graph = model.to_graph_module()

    # GIVEN we track initial weights
    initial_linear1_weight = model.linear1.weight.data.clone()
    initial_linear2_weight = model.linear2.weight.data.clone()

    # GIVEN calibration data
    calibration_data = [torch.randn(1, 5) for _ in range(5)]

    # GIVEN an optimization function
    def optimize_first_output(module: torch.nn.Module, dataset: Iterable[torch.Tensor]) -> None:
        optim = torch.optim.SGD(params=module.parameters(), lr=0.1)
        for batch in dataset:
            optim.zero_grad()
            output = module(batch)
            loss = (output**2).mean()
            loss.backward()
            optim.step()

    # GIVEN a spec targeting the first output path
    specs = [
        SubgraphSpec(
            input=graph.node_ref(graph.linear1),
            output=graph.node_ref(graph.relu),
            fn=optimize_first_output,
        )
    ]

    # WHEN we run the optimizer
    optimizer = LocalOptimizer(graph, specs)
    optimizer.optimize(calibration_data)

    # THEN linear1 should be optimized
    assert not torch.allclose(initial_linear1_weight, model.linear1.weight.data)
    # THEN linear2 should remain unchanged
    assert torch.allclose(initial_linear2_weight, model.linear2.weight.data)


def test_local_optimization_multiple_non_overlapping_specs() -> None:
    """Test LocalOptimizer with multiple non-overlapping specs."""
    # GIVEN a model with two residual blocks
    model = Model()
    graph = model.to_graph_module()

    # GIVEN we track initial weights
    initial_residual1_weight = model.residual_1.linear.weight.data.clone()
    initial_residual2_weight = model.residual_2.linear.weight.data.clone()

    # GIVEN calibration data
    calibration_data = [torch.randn(1, 5) for _ in range(5)]

    # GIVEN a simple optimization function
    def simple_opt(module: torch.nn.Module, dataset: Iterable[torch.Tensor]) -> None:
        optim = torch.optim.SGD(params=module.parameters(), lr=0.1)
        for batch in dataset:
            optim.zero_grad()
            output = module(batch)
            loss = (output**2).mean()
            loss.backward()
            optim.step()

    # GIVEN two non-overlapping specs
    specs = [
        SubgraphSpec(
            input=graph.node_ref(graph.residual_1.linear),
            output=graph.node_ref(graph.residual_1.linear),
            fn=simple_opt,
        ),
        SubgraphSpec(
            input=graph.node_ref(graph.residual_2.linear),
            output=graph.node_ref(graph.residual_2.linear),
            fn=simple_opt,
        ),
    ]

    # WHEN we run the optimizer
    optimizer = LocalOptimizer(graph, specs)
    optimizer.optimize(calibration_data)

    # THEN both residual blocks should be optimized
    assert not torch.allclose(initial_residual1_weight, model.residual_1.linear.weight.data)
    assert not torch.allclose(initial_residual2_weight, model.residual_2.linear.weight.data)


def test_local_optimization_entire_graph() -> None:
    """Test LocalOptimizer when spec covers entire graph."""
    # GIVEN a model and its graph
    model = Model()
    graph = model.to_graph_module()

    # GIVEN we track all weights
    initial_residual1_weight = model.residual_1.linear.weight.data.clone()
    initial_residual2_weight = model.residual_2.linear.weight.data.clone()

    # GIVEN calibration data
    calibration_data = [torch.randn(1, 5) for _ in range(5)]

    # GIVEN an optimization function
    def full_opt(module: torch.nn.Module, dataset: Iterable[torch.Tensor]) -> None:
        optim = torch.optim.SGD(params=module.parameters(), lr=0.1)
        for batch in dataset:
            optim.zero_grad()
            output = module(batch)
            loss = (output**2).mean()
            loss.backward()
            optim.step()

    # GIVEN a spec covering the entire graph
    specs = [
        SubgraphSpec(
            input=graph.node_ref(graph.residual_1.linear),
            output=graph.node_ref(graph.sigmoid),
            fn=full_opt,
        )
    ]

    # WHEN we run the optimizer
    optimizer = LocalOptimizer(graph, specs)
    optimizer.optimize(calibration_data)

    # THEN all weights should be optimized
    assert not torch.allclose(initial_residual1_weight, model.residual_1.linear.weight.data)
    assert not torch.allclose(initial_residual2_weight, model.residual_2.linear.weight.data)


def test_local_optimization_with_const_inputs() -> None:
    """Test LocalOptimizer with Const inputs in the graph."""
    # GIVEN a graph with a constant input
    const_value = torch.randn(5)
    graph = GraphModule()
    input_ref = graph.add_input("input")
    const_node = graph.add_node("const_node", ConstReturn(), args=[input_ref, Const(const_value)])
    linear = torch.nn.Linear(5, 3)
    linear_node = graph.add_node("linear", linear, args=[const_node])
    graph.add_output(linear_node)

    # GIVEN we track initial weights
    initial_weight = linear.weight.data.clone()

    # GIVEN calibration data
    calibration_data = [None for _ in range(5)]

    # GIVEN an optimization function
    def opt_with_const(module: torch.nn.Module, dataset: Iterable[torch.Tensor]) -> None:
        optim = torch.optim.SGD(params=module.parameters(), lr=0.1)
        for batch in dataset:
            optim.zero_grad()
            output = module(batch)
            loss = (output**2).mean()
            loss.backward()
            optim.step()

    # GIVEN a spec targeting the linear layer
    specs = [
        SubgraphSpec(
            input=linear_node,
            output=linear_node,
            fn=opt_with_const,
        )
    ]

    # WHEN we run the optimizer
    optimizer = LocalOptimizer(graph, specs)
    optimizer.optimize(calibration_data)

    # THEN the linear layer should be optimized
    assert not torch.allclose(initial_weight, linear.weight.data)


def test_local_optimization_no_specs() -> None:
    """Test LocalOptimizer with no optimization specs (only partitioning)."""
    # GIVEN a model and its graph
    model = Model()
    graph = model.to_graph_module()

    # GIVEN we track initial weights
    initial_residual1_weight = model.residual_1.linear.weight.data.clone()
    initial_residual2_weight = model.residual_2.linear.weight.data.clone()

    # GIVEN calibration data
    calibration_data = [torch.randn(1, 5) for _ in range(5)]

    # GIVEN no optimization specs (empty list)
    specs: list[SubgraphSpec] = []

    # WHEN we run the optimizer
    optimizer = LocalOptimizer(graph, specs)
    optimizer.optimize(calibration_data)

    # THEN no weights should change (only forward passes)
    assert torch.allclose(initial_residual1_weight, model.residual_1.linear.weight.data)
    assert torch.allclose(initial_residual2_weight, model.residual_2.linear.weight.data)


def test_local_optimization_with_kwargs() -> None:
    """Test LocalOptimizer with modules that use keyword arguments."""
    # GIVEN a graph with keyword arguments
    graph = GraphModule()
    input_ref = graph.add_input("input")
    const_value = torch.randn(5)
    const_node = graph.add_node(
        "const_node",
        ConstReturnKwargs(),
        args=[input_ref],
        kwargs={"const_kwarg": Const(const_value)},
    )
    linear = torch.nn.Linear(5, 3)
    linear_node = graph.add_node("linear", linear, args=[const_node])
    graph.add_output(linear_node)

    # GIVEN we track initial weights
    initial_weight = linear.weight.data.clone()

    # GIVEN calibration data
    calibration_data = [None for _ in range(5)]

    # GIVEN an optimization function
    def opt_kwargs(module: torch.nn.Module, dataset: Iterable[torch.Tensor]) -> None:
        optim = torch.optim.SGD(params=module.parameters(), lr=0.1)
        for batch in dataset:
            optim.zero_grad()
            output = module(batch)
            loss = (output**2).mean()
            loss.backward()
            optim.step()

    # GIVEN a spec targeting the linear layer
    specs = [
        SubgraphSpec(
            input=linear_node,
            output=linear_node,
            fn=opt_kwargs,
        )
    ]

    # WHEN we run the optimizer
    optimizer = LocalOptimizer(graph, specs)
    optimizer.optimize(calibration_data)

    # THEN the linear layer should be optimized
    assert not torch.allclose(initial_weight, linear.weight.data)


def test_local_optimization_with_multiple_inputs() -> None:
    """Test LocalOptimizer with graph that has multiple inputs."""
    # GIVEN a graph with multiple inputs
    graph = GraphModule()
    input1 = graph.add_input("input1")
    input2 = graph.add_input("input2")

    linear = torch.nn.Linear(5, 5)
    (merged,) = graph.add_subgraph("add", Add().to_graph_module(), [input1, input2])
    output = graph.add_node("linear", linear, args=[merged])
    graph.add_output(output)

    # GIVEN we track initial weights
    initial_weight = linear.weight.data.clone()

    # GIVEN calibration data with separate iterables for each input
    calibration_data_input1 = [torch.randn(1, 5) for _ in range(5)]
    calibration_data_input2 = [torch.randn(1, 5) for _ in range(5)]

    # GIVEN an optimization function
    def multi_input_opt(module: torch.nn.Module, dataset: Iterable[torch.Tensor]) -> None:
        optim = torch.optim.SGD(params=module.parameters(), lr=0.1)
        for batch in dataset:
            optim.zero_grad()
            output = module(batch)
            loss = (output**2).mean()
            loss.backward()
            optim.step()

    # GIVEN a spec targeting the linear layer
    specs = [
        SubgraphSpec(
            input=output,
            output=output,
            fn=multi_input_opt,
        )
    ]

    # WHEN we run the optimizer with multiple input datasets
    optimizer = LocalOptimizer(graph, specs)
    optimizer.optimize(calibration_data_input1, calibration_data_input2)

    # THEN the linear layer should be optimized
    assert not torch.allclose(initial_weight, linear.weight.data)
