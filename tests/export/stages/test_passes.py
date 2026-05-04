# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import torch

from fastforward.export.stages.passes import (
    FF_QUANTIZATION_SPEC,
    AnnotateFFQuantSpecs,
    PropagateFFQuantSpecs,
    _ff_quantization_nodes,
)


class _QuantParamHost(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor([0.5], dtype=torch.float32))
        self.register_buffer("offset", torch.tensor([10.0], dtype=torch.float32))


def _build_module_with_quantize_and_dequantize() -> torch.fx.GraphModule:
    root = _QuantParamHost()
    graph = torch.fx.Graph()

    input_node = graph.placeholder("x")
    scale_node = graph.get_attr("scale")
    offset_node = graph.get_attr("offset")

    quantize_node = graph.call_function(
        torch.ops.fastforward.quantize_by_tile.default,
        args=(input_node, scale_node, (1,), 8.0, torch.int8, offset_node),
    )
    dequantize_node = graph.call_function(
        torch.ops.fastforward.dequantize_by_tile.default,
        args=(quantize_node, scale_node, (1,), offset_node, torch.float32),
    )
    graph.output(dequantize_node)

    module: torch.fx.GraphModule = torch.fx.GraphModule(root, graph)
    return module


def _build_module_with_two_quantize_and_dequantize_pairs() -> torch.fx.GraphModule:
    root = _QuantParamHost()
    graph = torch.fx.Graph()

    input_node = graph.placeholder("x")
    scale_node = graph.get_attr("scale")
    offset_node = graph.get_attr("offset")

    quantize_node_1 = graph.call_function(
        torch.ops.fastforward.quantize_by_tile.default,
        args=(input_node, scale_node, (1,), 8.0, torch.int8, offset_node),
    )
    dequantize_node_1 = graph.call_function(
        torch.ops.fastforward.dequantize_by_tile.default,
        args=(quantize_node_1, scale_node, (1,), offset_node, torch.float32),
    )
    quantize_node_2 = graph.call_function(
        torch.ops.fastforward.quantize_by_tile.default,
        args=(dequantize_node_1, scale_node, (1,), 8.0, torch.int8, offset_node),
    )
    dequantize_node_2 = graph.call_function(
        torch.ops.fastforward.dequantize_by_tile.default,
        args=(quantize_node_2, scale_node, (1,), offset_node, torch.float32),
    )
    graph.output(dequantize_node_2)

    module: torch.fx.GraphModule = torch.fx.GraphModule(root, graph)
    return module


def _build_module_with_mixed_ops_and_quantization() -> torch.fx.GraphModule:
    root = _QuantParamHost()
    graph = torch.fx.Graph()

    input_node = graph.placeholder("x")
    relu_node = graph.call_function(torch.ops.aten.relu.default, args=(input_node,))
    scale_node = graph.get_attr("scale")
    offset_node = graph.get_attr("offset")

    quantize_node = graph.call_function(
        torch.ops.fastforward.quantize_by_tile.default,
        args=(relu_node, scale_node, (1,), 8.0, torch.int8, offset_node),
    )
    dequantize_node = graph.call_function(
        torch.ops.fastforward.dequantize_by_tile.default,
        args=(quantize_node, scale_node, (1,), offset_node, torch.float32),
    )
    sigmoid_node = graph.call_function(torch.ops.aten.sigmoid.default, args=(dequantize_node,))
    graph.output(sigmoid_node)

    module: torch.fx.GraphModule = torch.fx.GraphModule(root, graph)
    return module


def _build_module_with_view() -> tuple[torch.fx.GraphModule, torch.fx.Node, torch.fx.Node]:
    graph = torch.fx.Graph()
    input_node = graph.placeholder("x")
    view_node = graph.call_function(
        torch.ops.aten.view.default,
        args=(input_node, (1, -1)),
    )
    graph.output(view_node)

    module: torch.fx.GraphModule = torch.fx.GraphModule(torch.nn.Module(), graph)
    return module, input_node, view_node


def _build_module_with_chained_views() -> tuple[
    torch.fx.GraphModule, torch.fx.Node, torch.fx.Node, torch.fx.Node
]:
    graph = torch.fx.Graph()
    input_node = graph.placeholder("x")
    view_node_1 = graph.call_function(
        torch.ops.aten.view.default,
        args=(input_node, (1, -1)),
    )
    view_node_2 = graph.call_function(
        torch.ops.aten.view.default,
        args=(view_node_1, (-1,)),
    )
    graph.output(view_node_2)

    module: torch.fx.GraphModule = torch.fx.GraphModule(torch.nn.Module(), graph)
    return module, input_node, view_node_1, view_node_2


def _build_module_with_non_view_op() -> tuple[torch.fx.GraphModule, torch.fx.Node, torch.fx.Node]:
    graph = torch.fx.Graph()
    input_node = graph.placeholder("x")
    non_view_node = graph.call_function(torch.ops.aten.relu.default, args=(input_node,))
    graph.output(non_view_node)

    module: torch.fx.GraphModule = torch.fx.GraphModule(torch.nn.Module(), graph)
    return module, input_node, non_view_node


def test_annotate_ff_quant_specs_removes_quantize_and_dequantize_nodes() -> None:
    # GIVEN: A graph containing FF quantize/dequantize operations.
    module = _build_module_with_quantize_and_dequantize()

    # WHEN: Running FF quantization annotation pass.
    output_module = AnnotateFFQuantSpecs()(module).graph_module

    # THEN: Quantize/dequantize nodes should be removed and quant spec attached to input.
    call_targets = [node.target for node in output_module.graph.nodes if node.op == "call_function"]
    assert torch.ops.fastforward.quantize_by_tile.default not in call_targets
    assert torch.ops.fastforward.dequantize_by_tile.default not in call_targets

    input_node = next(node for node in output_module.graph.nodes if node.op == "placeholder")
    assert FF_QUANTIZATION_SPEC in input_node.meta


def test_annotate_ff_quant_specs_removes_all_quantize_and_dequantize_nodes() -> None:
    # GIVEN: A graph containing multiple FF quantize/dequantize operations.
    module = _build_module_with_two_quantize_and_dequantize_pairs()

    # WHEN: Running FF quantization annotation pass.
    output_module = AnnotateFFQuantSpecs()(module).graph_module

    # THEN: All quantize/dequantize nodes should be removed.
    call_targets = [node.target for node in output_module.graph.nodes if node.op == "call_function"]
    assert torch.ops.fastforward.quantize_by_tile.default not in call_targets
    assert torch.ops.fastforward.dequantize_by_tile.default not in call_targets


def test_annotate_ff_quant_specs_removes_quant_nodes_and_retains_other_nodes() -> None:
    # GIVEN: A graph with FF quant/dequant nodes plus non-FF operators.
    module = _build_module_with_mixed_ops_and_quantization()

    # WHEN: Running FF quantization annotation pass.
    output_module = AnnotateFFQuantSpecs()(module).graph_module

    # THEN: FF quant/dequant nodes are removed, but other call_function nodes remain.
    call_targets = [node.target for node in output_module.graph.nodes if node.op == "call_function"]
    assert torch.ops.fastforward.quantize_by_tile.default not in call_targets
    assert torch.ops.fastforward.dequantize_by_tile.default not in call_targets
    assert torch.ops.aten.relu.default in call_targets
    assert torch.ops.aten.sigmoid.default in call_targets
    assert len(call_targets) == 2


def test_ff_quantization_nodes_only_returns_quantize_and_dequantize_nodes() -> None:
    # GIVEN: A mixed graph containing FF quant/dequant and non-FF call_function ops.
    module = _build_module_with_mixed_ops_and_quantization()

    # WHEN: Selecting only FF quantization nodes.
    quantization_nodes = list(_ff_quantization_nodes(module.graph))

    # THEN: Only quantize/dequantize nodes are returned.
    assert len(quantization_nodes) == 2
    assert quantization_nodes[0].target == torch.ops.fastforward.quantize_by_tile.default
    assert quantization_nodes[1].target == torch.ops.fastforward.dequantize_by_tile.default


def test_propagate_ff_quant_specs_propagates_through_view_ops() -> None:
    # GIVEN: A view op where only the input node has FF quantization spec.
    module, input_node, view_node = _build_module_with_view()
    input_node.meta[FF_QUANTIZATION_SPEC] = object()

    # WHEN: Running quantization spec propagation pass.
    output_module = PropagateFFQuantSpecs()(module).graph_module

    # THEN: The view node should receive the same quantization spec.
    propagated_view_node = next(
        node
        for node in output_module.graph.nodes
        if node.op == "call_function" and node.target == torch.ops.aten.view.default
    )
    assert FF_QUANTIZATION_SPEC in propagated_view_node.meta
    assert propagated_view_node.meta[FF_QUANTIZATION_SPEC] is input_node.meta[FF_QUANTIZATION_SPEC]
    assert propagated_view_node is view_node


def test_propagate_ff_quant_specs_propagates_downward_through_chained_view_ops() -> None:
    # GIVEN: Chained view ops where only the initial input has a quantization spec.
    module, input_node, view_node_1, view_node_2 = _build_module_with_chained_views()
    input_spec = object()
    input_node.meta[FF_QUANTIZATION_SPEC] = input_spec

    # WHEN: Running quantization spec propagation pass.
    output_module = PropagateFFQuantSpecs()(module).graph_module

    # THEN: Both downstream view nodes should receive the same quantization spec.
    propagated_view_nodes = [
        node
        for node in output_module.graph.nodes
        if node.op == "call_function" and node.target == torch.ops.aten.view.default
    ]
    assert len(propagated_view_nodes) == 2
    assert view_node_1.meta[FF_QUANTIZATION_SPEC] is input_spec
    assert view_node_2.meta[FF_QUANTIZATION_SPEC] is input_spec


def test_propagate_ff_quant_specs_propagates_upward_through_chained_view_ops() -> None:
    # GIVEN: Chained view ops where only the final output view has a quantization spec.
    module, input_node, view_node_1, view_node_2 = _build_module_with_chained_views()
    output_spec = object()
    view_node_2.meta[FF_QUANTIZATION_SPEC] = output_spec

    # WHEN: Running quantization spec propagation pass.
    output_module = PropagateFFQuantSpecs()(module).graph_module

    # THEN: The quantization spec should be propagated back to upstream view/input nodes.
    propagated_view_nodes = [
        node
        for node in output_module.graph.nodes
        if node.op == "call_function" and node.target == torch.ops.aten.view.default
    ]
    assert len(propagated_view_nodes) == 2
    assert view_node_1.meta[FF_QUANTIZATION_SPEC] is output_spec
    assert input_node.meta[FF_QUANTIZATION_SPEC] is output_spec


def test_propagate_ff_quant_specs_does_not_propagate_for_non_view_ops() -> None:
    # GIVEN: A non-view op where only input has FF quantization spec.
    module, input_node, non_view_node = _build_module_with_non_view_op()
    input_node.meta[FF_QUANTIZATION_SPEC] = object()

    # WHEN: Running quantization spec propagation pass.
    output_module = PropagateFFQuantSpecs()(module).graph_module

    # THEN: The non-view op should not receive quantization spec metadata.
    propagated_non_view_node = next(
        node
        for node in output_module.graph.nodes
        if node.op == "call_function" and node.target == torch.ops.aten.relu.default
    )
    assert FF_QUANTIZATION_SPEC not in propagated_non_view_node.meta
    assert propagated_non_view_node is non_view_node


def test_propagate_ff_quant_specs_preserves_existing_view_encoding() -> None:
    # GIVEN: A view op where both input and output already have quantization specs.
    module, input_node, view_node = _build_module_with_view()
    input_spec = object()
    output_spec = object()
    input_node.meta[FF_QUANTIZATION_SPEC] = input_spec
    view_node.meta[FF_QUANTIZATION_SPEC] = output_spec

    # WHEN: Running quantization spec propagation pass.
    output_module = PropagateFFQuantSpecs()(module).graph_module

    # THEN: Existing output encoding should be preserved (no overwrite).
    propagated_view_node = next(
        node
        for node in output_module.graph.nodes
        if node.op == "call_function" and node.target == torch.ops.aten.view.default
    )
    assert propagated_view_node.meta[FF_QUANTIZATION_SPEC] is output_spec
    assert propagated_view_node.meta[FF_QUANTIZATION_SPEC] is not input_spec
    assert propagated_view_node is view_node
