# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


import functools
import uuid

import pytest
import torch

from fastforward._orchestration.graph_module import (
    DEFAULT_CONTEXT,
    Const,
    GraphModule,
    Group,
    InputRef,
    NodeRef,
    Span,
    SubgraphSpec,
    create_subgraph,
    find_nodes_on_path,
    inference_mode,
    local_optimize,
    reduce_resolution,
    remap_subgraph_reference,
)
from fastforward._orchestration.instruction_engine import (
    ActivationDataset,
    ActivationRegister,
    CallModule,
)

from .conftest import (
    Add,
    ConstReturn,
    ConstReturnKwargs,
    DualOutModel,
    Model,
    MultiOutputModel,
    TwoLayerModel,
    noop,
    sgd_step,
)


def test_graph_module_forward_pass(model: Model) -> None:
    # GIVEN a PyTorch model and its equivalent GraphModule representation
    graph = model.to_graph_module()
    x = torch.randn(1, 5)

    # WHEN we run both the original model and the graph module engine
    model_output = model(x)
    engine_output = graph(x)

    # THEN the outputs should be identical
    torch.testing.assert_close(model_output, engine_output)


def test_multi_output_graph_module(multi_output_model: MultiOutputModel) -> None:
    # GIVEN a model with multiple outputs and its corresponding GraphModule
    model = multi_output_model
    graph = model.to_graph_module()
    x = torch.randn(1, 5)

    # WHEN we run both the original model and the graph module
    model_out1, model_out2 = model(x)
    graph_out1, graph_out2 = graph(x)

    # THEN the outputs should be identical
    torch.testing.assert_close(model_out1, graph_out1)
    torch.testing.assert_close(model_out2, graph_out2)


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


def test_create_subgraph_functional_equivalence(model: Model) -> None:
    # GIVEN a Model and GraphModule equivalent
    graph = model.to_graph_module()

    # GIVEN the minimal node set lying in the GraphModule
    sigmoid = graph.get_submodule("sigmoid")
    path_nodes = find_nodes_on_path(
        graph,
        graph.node_ref(model.residual_1.linear),
        graph.node_ref(sigmoid),
    )

    # WHEN we materialise that path as a standalone GraphModule
    subgraph = create_subgraph(graph, path_nodes)

    # THEN the subgraph must only contain the nodes on the path
    assert set(subgraph._nodes) == set(node.id for node in path_nodes)

    # THEN executing the subgraph produces the same result as the parent graph
    x = torch.randn(1, 5)
    torch.testing.assert_close(graph(x), subgraph(x))


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


def test_attribute_ref_indexing(multi_output_model: MultiOutputModel) -> None:
    # GIVEN a model that returns a tuple and a GraphModule that uses AttributeRef to access elements
    graph = GraphModule()
    model = multi_output_model
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


def test_local_error_opt(model: Model) -> None:
    """Integration test for Local Error Optimization."""
    # GIVEN a Model with two residual blocks and its GraphModule representation
    graph = model.to_graph_module()

    # GIVEN we store initial weights for comparison
    initial_residual1_weight = model.residual_1.linear.weight.data.clone()
    initial_residual2_weight = model.residual_2.linear.weight.data.clone()

    # GIVEN a simple calibration dataset
    calibration_data = [torch.randn(1, 5) for _ in range(10)]

    # GIVEN a SubgraphSpec that targets only the first residual's linear layer
    specs = [
        SubgraphSpec(
            region=model.residual_1.linear,
            fn=functools.partial(sgd_step, lr=0.1),
        )
    ]

    # WHEN we run local_optimize on the calibration data
    with local_optimize(graph, specs):
        graph(calibration_data)

    # THEN only residual_1's linear weights should have changed
    assert not torch.allclose(initial_residual1_weight, model.residual_1.linear.weight.data)
    assert torch.allclose(initial_residual2_weight, model.residual_2.linear.weight.data)


def test_local_optimization_overlapping_specs_raises(model: Model) -> None:
    """Test that local_optimize rejects overlapping specs."""
    # GIVEN two SubgraphSpecs that overlap
    graph = model.to_graph_module()
    residual_1_linear = graph.get_submodule("residual_1.linear")
    residual_1_relu = graph.get_submodule("residual_1.relu")
    specs = [
        SubgraphSpec(
            region=Span(start=residual_1_linear, end=residual_1_relu),
            fn=noop,
        ),
        SubgraphSpec(
            region=Span(start=residual_1_linear, end=residual_1_relu),
            fn=noop,
        ),
    ]

    # WHEN we try to create a local_optimize context with overlapping specs
    # THEN it should raise a ValueError
    with pytest.raises(ValueError, match="Overlapping nodes"):
        local_optimize(graph, specs)


def test_call_module_single_tensor_arg() -> None:
    """Test that CallModule correctly handles a single tensor argument without unpacking tensor elements."""
    # GIVEN a simple linear module
    module = torch.nn.Linear(5, 3)

    # GIVEN a register with a single tensor batch in context-aware format
    input_ref = InputRef(uuid.uuid4(), "input")
    target_ref = NodeRef(uuid.uuid4(), "target")
    register: ActivationRegister = {
        input_ref: {DEFAULT_CONTEXT: ActivationDataset([torch.randn(2, 5)])}
    }

    # GIVEN a CallModule instruction with single arg and default context

    instruction = CallModule(
        module=module,
        args=[input_ref],
        kwargs={},
        target=target_ref,
        contexts=[DEFAULT_CONTEXT],
    )

    # WHEN we execute the instruction
    instruction.execute(register)

    # THEN the output should be computed correctly
    assert target_ref in register
    output_contexts = register[target_ref]
    assert len(output_contexts[DEFAULT_CONTEXT]) == 1
    assert output_contexts[DEFAULT_CONTEXT].batches[0].shape == (2, 3)


def test_local_optimization_with_attribute_refs(multi_output_model: MultiOutputModel) -> None:
    """Test local_optimize with AttributeRef outputs in subgraphs."""
    # GIVEN a model that returns multiple outputs
    model = multi_output_model
    graph = model.to_graph_module()

    # GIVEN we track initial weights
    initial_linear1_weight = model.linear1.weight.data.clone()
    initial_linear2_weight = model.linear2.weight.data.clone()

    # GIVEN calibration data
    calibration_data = [torch.randn(1, 5) for _ in range(5)]

    # GIVEN a spec targeting the first output path
    linear1 = graph.get_submodule("linear1")
    relu = graph.get_submodule("relu")
    specs = [
        SubgraphSpec(
            region=Span(start=linear1, end=relu),
            fn=sgd_step,
        )
    ]

    # WHEN we run the optimizer
    with local_optimize(graph, specs):
        graph(calibration_data)

    # THEN linear1 should be optimized
    assert not torch.allclose(initial_linear1_weight, model.linear1.weight.data)
    # THEN linear2 should remain unchanged
    assert torch.allclose(initial_linear2_weight, model.linear2.weight.data)


def test_local_optimization_multiple_non_overlapping_specs(model: Model) -> None:
    """Test local_optimize with multiple non-overlapping specs."""
    # GIVEN a model with two residual blocks
    graph = model.to_graph_module()

    # GIVEN we track initial weights
    initial_residual1_weight = model.residual_1.linear.weight.data.clone()
    initial_residual2_weight = model.residual_2.linear.weight.data.clone()

    # GIVEN calibration data
    calibration_data = [torch.randn(1, 5) for _ in range(5)]

    # GIVEN two non-overlapping specs
    residual_1_linear = graph.get_submodule("residual_1.linear")
    residual_2_linear = graph.get_submodule("residual_2.linear")
    specs = [
        SubgraphSpec(
            region=residual_1_linear,
            fn=sgd_step,
        ),
        SubgraphSpec(
            region=residual_2_linear,
            fn=sgd_step,
        ),
    ]

    # WHEN we run the optimizer
    with local_optimize(graph, specs):
        graph(calibration_data)

    # THEN both residual blocks should be optimized
    assert not torch.allclose(initial_residual1_weight, model.residual_1.linear.weight.data)
    assert not torch.allclose(initial_residual2_weight, model.residual_2.linear.weight.data)


def test_local_optimization_entire_graph(model: Model) -> None:
    """Test local_optimize when spec covers entire graph."""
    # GIVEN a model and its graph
    graph = model.to_graph_module()

    # GIVEN we track all weights
    initial_residual1_weight = model.residual_1.linear.weight.data.clone()
    initial_residual2_weight = model.residual_2.linear.weight.data.clone()

    # GIVEN calibration data
    calibration_data = [torch.randn(1, 5) for _ in range(5)]

    # GIVEN a spec covering the entire graph
    residual_1_linear = graph.get_submodule("residual_1.linear")
    sigmoid = graph.get_submodule("sigmoid")
    specs = [
        SubgraphSpec(
            region=Span(start=residual_1_linear, end=sigmoid),
            fn=sgd_step,
        )
    ]

    # WHEN we run the optimizer
    with local_optimize(graph, specs):
        graph(calibration_data)

    # THEN all weights should be optimized
    assert not torch.allclose(initial_residual1_weight, model.residual_1.linear.weight.data)
    assert not torch.allclose(initial_residual2_weight, model.residual_2.linear.weight.data)


def test_local_optimization_with_const_inputs() -> None:
    """Test local_optimize with Const inputs in the graph."""
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

    # GIVEN a spec targeting the linear layer
    specs = [
        SubgraphSpec(
            region=linear,
            fn=sgd_step,
        )
    ]

    # WHEN we run the optimizer
    with local_optimize(graph, specs):
        graph(calibration_data)

    # THEN the linear layer should be optimized
    assert not torch.allclose(initial_weight, linear.weight.data)


def test_local_optimization_no_specs(model: Model) -> None:
    """Test local_optimize with no optimization specs (only partitioning)."""
    # GIVEN a model and its graph
    graph = model.to_graph_module()

    # GIVEN we track initial weights
    initial_residual1_weight = model.residual_1.linear.weight.data.clone()
    initial_residual2_weight = model.residual_2.linear.weight.data.clone()

    # GIVEN calibration data
    calibration_data = [torch.randn(1, 5) for _ in range(5)]

    # GIVEN no optimization specs (empty list)
    specs: list[SubgraphSpec] = []

    # WHEN we run the optimizer
    with local_optimize(graph, specs):
        graph(calibration_data)

    # THEN no weights should change (only forward passes)
    assert torch.allclose(initial_residual1_weight, model.residual_1.linear.weight.data)
    assert torch.allclose(initial_residual2_weight, model.residual_2.linear.weight.data)


def _spec_cases(model: TwoLayerModel) -> list[tuple[str, list[SubgraphSpec], int]]:
    return [
        ("no specs", [], 2),
        (
            "fold target (layer_0)",
            [SubgraphSpec(region=model.layer_0, fn=noop)],
            2,
        ),
        (
            "both top-level folds",
            [
                SubgraphSpec(region=model.layer_0, fn=noop),
                SubgraphSpec(region=model.layer_1, fn=noop),
            ],
            2,
        ),
        (
            "leaf target (q_proj_0)",
            [SubgraphSpec(region=model.layer_0.attn.q_proj, fn=noop)],
            7,
        ),
        (
            "two leaves same fold (q_proj_0 + k_proj_0)",
            [
                SubgraphSpec(region=model.layer_0.attn.q_proj, fn=noop),
                SubgraphSpec(region=model.layer_0.attn.k_proj, fn=noop),
            ],
            7,
        ),
        (
            "symmetric leaves across layers",
            [
                SubgraphSpec(region=model.layer_0.attn.q_proj, fn=noop),
                SubgraphSpec(region=model.layer_1.attn.q_proj, fn=noop),
            ],
            12,
        ),
        (
            "mixed fold + leaf (attn_0 fold, q_proj_1 leaf)",
            [
                SubgraphSpec(region=model.layer_0.attn, fn=noop),
                SubgraphSpec(region=model.layer_1.attn.q_proj, fn=noop),
            ],
            8,
        ),
        (
            "path within single fold (q_proj_0 -> out_proj_0)",
            [
                SubgraphSpec(
                    region=Span(start=model.layer_0.attn.q_proj, end=model.layer_0.attn.out_proj),
                    fn=noop,
                )
            ],
            5,
        ),
        (
            "path across layers (mlp_0.down -> q_proj_1)",
            [
                SubgraphSpec(
                    region=Span(start=model.layer_0.mlp.down, end=model.layer_1.attn.q_proj),
                    fn=noop,
                )
            ],
            9,
        ),
        (
            "group of Q/K/V siblings in layer_0.attn",
            [
                SubgraphSpec(
                    region=Group((
                        model.layer_0.attn.q_proj,
                        model.layer_0.attn.k_proj,
                        model.layer_0.attn.v_proj,
                    )),
                    fn=noop,
                ),
            ],
            5,
        ),
        (
            "group of Q/K/V + adjacent out_proj singleton",
            [
                SubgraphSpec(
                    region=Group((
                        model.layer_0.attn.q_proj,
                        model.layer_0.attn.k_proj,
                        model.layer_0.attn.v_proj,
                    )),
                    fn=noop,
                ),
                SubgraphSpec(region=model.layer_0.attn.out_proj, fn=noop),
            ],
            5,
        ),
        (
            "group of Q/K/V in each of layer_0 and layer_1",
            [
                SubgraphSpec(
                    region=Group((
                        model.layer_0.attn.q_proj,
                        model.layer_0.attn.k_proj,
                        model.layer_0.attn.v_proj,
                    )),
                    fn=noop,
                ),
                SubgraphSpec(
                    region=Group((
                        model.layer_1.attn.q_proj,
                        model.layer_1.attn.k_proj,
                        model.layer_1.attn.v_proj,
                    )),
                    fn=noop,
                ),
            ],
            8,
        ),
    ]


def test_reduce_resolution_forward_pass_correctness(two_layer_model: TwoLayerModel) -> None:
    # GIVEN a 2-layer transformer-shaped model with nested folds
    model = two_layer_model
    graph = model.to_graph_module()
    x = torch.randn(1, 8)
    expected = model(x)

    # WHEN each spec configuration is reduced and executed
    for desc, specs, expected_node_count in _spec_cases(model):
        reduced = reduce_resolution(graph, specs)

        # THEN forward output matches the original model at every resolution
        torch.testing.assert_close(reduced(x), expected, msg=f"forward mismatch for case: {desc}")

        # THEN the reduced graph has the expected number of nodes
        assert len(reduced._nodes) == expected_node_count, (
            f"[{desc}] expected {expected_node_count} nodes, got {len(reduced._nodes)}"
        )


def test_reduce_resolution_no_specs_keeps_top_level_folds_coarse(
    two_layer_model: TwoLayerModel,
) -> None:
    # GIVEN a 2-layer model with no specs (fully coarse target)
    model = two_layer_model
    graph = model.to_graph_module()

    # WHEN we reduce with no specs
    reduced = reduce_resolution(graph, [])
    modules = [n.target for n in reduced._nodes.values()]

    # THEN only the top-level folds appear, not their internals
    assert len(modules) == 2
    assert model.layer_0 in modules
    assert model.layer_1 in modules
    assert model.layer_0.attn not in modules
    assert model.layer_0.mlp not in modules


def test_reduce_resolution_leaf_target_exposes_siblings_keeps_unrelated_coarse(
    two_layer_model: TwoLayerModel,
) -> None:
    # GIVEN a 2-layer model with a leaf target inside layer_0.attn
    model = two_layer_model
    graph = model.to_graph_module()
    specs = [SubgraphSpec(region=model.layer_0.attn.q_proj, fn=noop)]

    # WHEN we reduce with that spec
    reduced = reduce_resolution(graph, specs)
    modules = [n.target for n in reduced._nodes.values()]

    # THEN the target's siblings inside layer_0.attn are exposed as leaves
    assert model.layer_0.attn.q_proj in modules
    assert model.layer_0.attn.k_proj in modules
    assert model.layer_0.attn.v_proj in modules
    assert model.layer_0.attn.out_proj in modules

    # THEN layer_0.mlp and layer_1 stay coarse (no exposed descendants)
    assert model.layer_0.mlp in modules
    assert model.layer_0.mlp.up not in modules
    assert model.layer_0.mlp.down not in modules
    assert model.layer_1 in modules
    assert model.layer_1.attn not in modules
    assert model.layer_1.mlp not in modules

    # THEN opened folds (layer_0 and layer_0.attn) do NOT appear themselves
    assert model.layer_0 not in modules
    assert model.layer_0.attn not in modules


def test_reduce_resolution_path_spec_inserts_subgraph_node(two_layer_model: TwoLayerModel) -> None:
    # GIVEN a 2-layer model with a path spec spanning two layers
    model = two_layer_model
    graph = model.to_graph_module()
    specs = [
        SubgraphSpec(
            region=Span(start=model.layer_0.mlp.down, end=model.layer_1.attn.q_proj), fn=noop
        )
    ]

    # WHEN we reduce with that path spec
    reduced = reduce_resolution(graph, specs)
    modules = [n.target for n in reduced._nodes.values()]

    # THEN a synthesized GraphModule replaces the path leaves
    assert any(isinstance(m, GraphModule) for m in modules)

    # THEN the path's individual leaves do not appear (they're inside the subgraph)
    assert model.layer_0.mlp.down not in modules
    assert model.layer_1.attn.q_proj not in modules

    # THEN untouched siblings stay at the coarsest possible level
    assert model.layer_0.attn in modules
    assert model.layer_1.mlp in modules


def test_group_non_siblings_raises(two_layer_model: TwoLayerModel) -> None:
    # GIVEN modules from different layers (not siblings)
    model = two_layer_model
    graph = model.to_graph_module()
    specs = [
        SubgraphSpec(
            region=Group((model.layer_0.attn.q_proj, model.layer_1.attn.q_proj)),
            fn=noop,
        )
    ]

    # WHEN we try to reduce with a non-sibling group
    # THEN it raises a ValueError
    with pytest.raises(ValueError, match="siblings"):
        reduce_resolution(graph, specs)


def test_group_single_module(two_layer_model: TwoLayerModel) -> None:
    # GIVEN a group with a single module
    model = two_layer_model
    graph = model.to_graph_module()
    x = torch.randn(1, 8)
    expected = model(x)
    specs = [
        SubgraphSpec(
            region=Group((model.layer_0.attn.q_proj,)),
            fn=noop,
        )
    ]

    # WHEN we reduce with a single-member group
    reduced = reduce_resolution(graph, specs)

    # THEN forward output still matches
    torch.testing.assert_close(reduced(x), expected)


def test_reduce_resolution_multi_output_fold_unwraps_first_output(
    dual_out_model: DualOutModel,
) -> None:
    # GIVEN a model whose inner fold returns TWO outputs consumed independently
    model = dual_out_model
    graph = model.to_graph_module()
    x = torch.randn(1, 8)
    expected = model(x)

    # WHEN we reduce with no specs, leaving the multi-output fold coarse
    reduced = reduce_resolution(graph, [])

    # THEN forward execution must unwrap output 0 (not pass the whole tuple downstream)
    torch.testing.assert_close(reduced(x), expected)


def test_reduce_resolution_repeated_input_binding_to_fold() -> None:
    # GIVEN a 2-input fold whose positional inputs are BOTH bound to the same external ref
    add_module = Add()
    graph = GraphModule()
    inp = graph.add_input("x")
    (out,) = graph.add_subgraph(
        "add", add_module.to_graph_module(), [inp, inp], original_module=add_module
    )
    graph.add_output(out)
    x = torch.randn(1, 5)
    expected = add_module(x, x)

    # WHEN we reduce with no specs (fold stays coarse, must be called as add(x, x))
    reduced = reduce_resolution(graph, [])

    # THEN forward must preserve the duplicate binding rather than deduping to add(x)
    torch.testing.assert_close(reduced(x), expected)


def test_local_optimization_with_kwargs() -> None:
    """Test local_optimize with modules that use keyword arguments."""
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

    # GIVEN a spec targeting the linear layer
    specs = [
        SubgraphSpec(
            region=linear,
            fn=sgd_step,
        )
    ]

    # WHEN we run the optimizer
    with local_optimize(graph, specs):
        graph(calibration_data)

    # THEN the linear layer should be optimized
    assert not torch.allclose(initial_weight, linear.weight.data)


def test_local_optimization_with_multiple_inputs() -> None:
    """Test local_optimize with graph that has multiple inputs."""
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

    # GIVEN a spec targeting the linear layer
    specs = [
        SubgraphSpec(
            region=linear,
            fn=sgd_step,
        )
    ]

    # WHEN we run the optimizer with multiple input datasets
    with local_optimize(graph, specs):
        graph(calibration_data_input1, calibration_data_input2)

    # THEN the linear layer should be optimized
    assert not torch.allclose(initial_weight, linear.weight.data)


def test_node_with_no_inputs_executes_once() -> None:
    """Test that nodes with no inputs execute once."""

    class RNGTensor(torch.nn.Module):
        """Generate a tensor."""

        def forward(self) -> torch.Tensor:
            return torch.randn(5)

    # GIVEN a GraphModule with a tensor generator that has no inputs
    graph = GraphModule()
    x = graph.add_input("x")

    # Node with NO inputs - just generates a tensor
    y = graph.add_node("get_tensor", RNGTensor(), args=[])

    # Node that uses both x and the generated y
    (result,) = graph.add_subgraph("add", Add().to_graph_module(), [x, y])
    graph.add_output(result)

    # WHEN we execute the graph
    x_input = torch.randn(5)
    output = graph(x_input)

    # THEN the output should be a valid tensor (x + y)
    assert output.shape == (5,)
    assert isinstance(output, torch.Tensor)


def test_inference_mode_restores_state_on_exit(model: Model) -> None:
    # GIVEN a fresh GraphModule (program and engine are None)
    graph = model.to_graph_module()
    original_program = graph._program
    original_engine = graph._engine
    assert original_program is None
    assert original_engine is None

    # WHEN we enter and exit inference_mode
    with inference_mode(graph):
        # THEN program and engine should be set inside the context
        assert graph._program is not None
        assert graph._engine is not None

    # THEN program and engine should be restored to their original values
    assert graph._program is original_program
    assert graph._engine is original_engine


def test_inference_mode_restores_previously_compiled_state(model: Model) -> None:
    # GIVEN a GraphModule that has already been used (program/engine are set)
    graph = model.to_graph_module()
    x = torch.randn(1, 5)
    graph(x)  # triggers compilation
    original_program = graph._program
    original_engine = graph._engine
    assert original_program is not None
    assert original_engine is not None

    # WHEN we enter and exit inference_mode
    with inference_mode(graph):
        assert graph._program is not original_program
        assert graph._engine is not original_engine

    # THEN the original program and engine should be restored
    assert graph._program is original_program
    assert graph._engine is original_engine


def test_inference_mode_produces_same_output_as_default(model: Model) -> None:
    # GIVEN a GraphModule and an input tensor
    graph = model.to_graph_module()
    x = torch.randn(1, 5)

    # WHEN we run the graph normally and under inference_mode
    default_output = graph(x)
    with inference_mode(graph):
        inference_output = graph(x)

    # THEN both outputs should be identical
    torch.testing.assert_close(inference_output, default_output)


def test_inference_mode_enables_torch_inference_mode() -> None:
    # GIVEN a module that records whether torch.is_inference_mode_enabled during forward
    class ProbeModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.is_on_inference_mode = False

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.is_on_inference_mode = torch.is_inference_mode_enabled()
            return x

    probe = ProbeModule()
    graph = GraphModule()
    inp = graph.add_input("x")
    out = graph.add_node("probe", probe, [inp])
    graph.add_output(out)

    # WHEN we run the graph under inference_mode
    with inference_mode(graph):
        graph(torch.randn(1, 5))

    # THEN torch inference mode should have been active during forward
    assert probe.is_on_inference_mode


def _add_two_input_graph() -> GraphModule:
    """Two-input graph computing x + y, batch-by-batch."""
    graph = GraphModule()
    x = graph.add_input("x")
    y = graph.add_input("y")
    out = graph.add_node("add", Add(), [x, y])
    graph.add_output(out)
    return graph


def test_graph_accepts_named_kwargs() -> None:
    # GIVEN two-input graph and one batch of named tensors
    graph = _add_two_input_graph()
    x = torch.tensor([1.0])
    y = torch.tensor([2.0])

    # WHEN we call with kwargs matching input_names
    result = graph(x=x, y=y)

    # THEN it returns the expected sum
    assert torch.allclose(result, torch.tensor([3.0]))


def test_graph_accepts_positional_tensors() -> None:
    # GIVEN two-input graph and two positional tensors in declared order
    graph = _add_two_input_graph()

    # WHEN we call positionally
    result = graph(torch.tensor([1.0]), torch.tensor([2.0]))

    # THEN positional binding follows input_names order
    assert torch.allclose(result, torch.tensor([3.0]))


def test_graph_accepts_mixed_positional_and_keyword_tensors() -> None:
    # GIVEN a two-input graph with one tensor given positionally and one by name
    graph = _add_two_input_graph()

    # WHEN we call with a positional arg bound to the first input and a kwarg for the second
    result = graph(torch.tensor([1.0]), y=torch.tensor([2.0]))

    # THEN the positional arg binds to x (declared first) and the kwarg binds to y
    assert torch.allclose(result, torch.tensor([3.0]))


def test_graph_accepts_per_input_lists_of_batches() -> None:
    # GIVEN a two-input graph and N batches per input as parallel lists
    graph = _add_two_input_graph()
    xs = [torch.tensor([float(i)]) for i in range(3)]
    ys = [torch.tensor([float(10 * i)]) for i in range(3)]

    # WHEN we pass parallel lists per named input
    results = graph(x=xs, y=ys)

    # THEN each batch is summed independently
    expected = [x + y for x, y in zip(xs, ys)]
    flat = next(iter(results.values()))  # single context
    for got, exp in zip(flat, expected):
        assert torch.allclose(got, exp)


def test_graph_accepts_iterable_of_dict_batches() -> None:
    # GIVEN a two-input graph and an iterable yielding dicts whose keys match input_names
    #   (the natural shape produced by HF DataLoaders)
    graph = _add_two_input_graph()
    batches = [
        {"x": torch.tensor([1.0]), "y": torch.tensor([10.0])},
        {"x": torch.tensor([2.0]), "y": torch.tensor([20.0])},
        {"x": torch.tensor([3.0]), "y": torch.tensor([30.0])},
    ]

    # WHEN we hand the iterable over as a single positional argument
    results = graph(batches)

    # THEN each dict is splatted onto the named inputs
    expected = [b["x"] + b["y"] for b in batches]
    flat = next(iter(results.values()))
    for got, exp in zip(flat, expected):
        assert torch.allclose(got, exp)


def test_graph_accepts_iterable_of_tuple_batches() -> None:
    # GIVEN a two-input graph and an iterable yielding (x, y) tuples
    graph = _add_two_input_graph()
    batches = [
        (torch.tensor([1.0]), torch.tensor([10.0])),
        (torch.tensor([2.0]), torch.tensor([20.0])),
    ]

    # WHEN we pass the iterable as a single positional argument
    results = graph(batches)

    # THEN tuple positions bind to declared input order
    expected = [x + y for x, y in batches]
    flat = next(iter(results.values()))
    for got, exp in zip(flat, expected):
        assert torch.allclose(got, exp)


def test_graph_accepts_iterable_of_dataclass_batches() -> None:
    import dataclasses

    @dataclasses.dataclass
    class Batch:
        x: torch.Tensor
        y: torch.Tensor

    # GIVEN a two-input graph and an iterable of dataclass batches with matching field names
    graph = _add_two_input_graph()
    batches = [
        Batch(x=torch.tensor([1.0]), y=torch.tensor([10.0])),
        Batch(x=torch.tensor([2.0]), y=torch.tensor([20.0])),
    ]

    # WHEN we pass the iterable as a single positional argument
    results = graph(batches)

    # THEN dataclass fields bind to inputs by name
    expected = [b.x + b.y for b in batches]
    flat = next(iter(results.values()))
    for got, exp in zip(flat, expected):
        assert torch.allclose(got, exp)
