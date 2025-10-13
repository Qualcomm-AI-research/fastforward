# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Any

import onnxscript


def _fix_reshape_allowzero(model: onnxscript.ir.Model) -> None:
    """Fix Reshape nodes with allowzero=1 for QNN compatibility."""
    num_nodes = len(model.graph)
    for i in range(num_nodes):
        node = model.graph.node(i)
        if node.op_type == "Reshape" and "allowzero" in node.attributes:
            del node.attributes["allowzero"]


def _rename_nodes_and_update_encodings(
    torch_onnx_model: onnxscript.ir.Model, quantization_logs: dict[str, Any], name_prefix: str
) -> tuple[onnxscript.ir.Model, dict[str, Any]]:
    """Rename ONNX nodes and update the encodings dictionary."""
    torch_onnx_model, name_mapping = _fix_onnx_names(torch_onnx_model, name_prefix)
    quantization_logs = _fix_encoding_names(quantization_logs, name_mapping)
    return torch_onnx_model, quantization_logs


def _fix_onnx_names(
    torch_onnx_model: onnxscript.ir.Model, new_name_prefix: str
) -> tuple[onnxscript.ir.Model, dict[str, str]]:
    name_mapping: dict[str, str] = {}

    # Rename initializers (weights/parameters)
    for initializer in list(torch_onnx_model.graph.initializers.values()):
        if hasattr(initializer, "name") and initializer.name:
            old_name = initializer.name
            new_name = f"{new_name_prefix}_{old_name}_0"
            name_mapping[old_name] = new_name
            initializer.name = new_name

    # Rename nodes and their outputs
    for node in torch_onnx_model.graph._nodes:
        # Rename the node itself
        if hasattr(node, "name") and node.name:
            old_node_name = node.name
            new_node_name = f"{new_name_prefix}_{old_node_name}_0"
            node.name = new_node_name

        # Rename node outputs
        for output in node.outputs:
            if hasattr(output, "name") and output.name:
                old_output_name = output.name
                new_output_name = f"{new_name_prefix}_{old_output_name}_0"
                name_mapping[old_output_name] = new_output_name
                output.name = new_output_name

    # Update node input references
    for node in torch_onnx_model.graph._nodes:
        for input_value in node.inputs:
            if (
                input_value is not None
                and hasattr(input_value, "name")
                and input_value.name
                and input_value.name in name_mapping
            ):
                input_value.name = name_mapping[input_value.name]

    # Update graph inputs
    for graph_input in torch_onnx_model.graph.inputs:
        if hasattr(graph_input, "name") and graph_input.name and graph_input.name in name_mapping:
            graph_input.name = name_mapping[graph_input.name]

    # Update graph outputs
    for graph_output in torch_onnx_model.graph.outputs:
        if (
            hasattr(graph_output, "name")
            and graph_output.name
            and graph_output.name in name_mapping
        ):
            graph_output.name = name_mapping[graph_output.name]

    return torch_onnx_model, name_mapping


def _fix_encoding_names(
    encodings_dictionary: dict[str, Any], name_mapping: dict[str, str]
) -> dict[str, Any]:
    new_encodings_dictionary: dict[str, Any] = {}

    for old_name, value in encodings_dictionary.items():
        new_name = name_mapping.get(old_name)
        if new_name is not None:
            new_encodings_dictionary[new_name] = value
        else:
            new_encodings_dictionary[old_name] = value

    return new_encodings_dictionary


def _onnx_input_output_renaming(
    torch_onnx_model: onnxscript.ir.Model,
    input_names: list[str] | None,
    output_names: list[str] | None,
    quantization_logs: dict[str, Any],
    new_old_input_spec_mapping: dict[str, str],
) -> onnxscript.ir.Model:
    torch_onnx_inputs = torch_onnx_model.graph.inputs
    torch_onnx_outputs = torch_onnx_model.graph.outputs

    if input_names is None:
        input_names = []
        for entry in torch_onnx_inputs:
            # The input node should always have a name,
            # otherwise something is wrong with the graph.
            assert entry.name is not None
            input_names.append(entry.name)

    if output_names is None:
        output_names = []
        for entry in torch_onnx_outputs:
            # The output node should always have a name,
            # otherwise something is wrong with the graph.
            assert entry.name is not None
            output_names.append(entry.name)

    if len(torch_onnx_inputs) != len(input_names) or len(torch_onnx_outputs) != len(output_names):
        msg = (
            f"The number of user-defined inputs/outputs ({len(input_names)}, {len(output_names)}) "
            + "does not match the number of graph inputs/outputs "
            + f"({len(torch_onnx_inputs)}, {len(torch_onnx_outputs)})"
        )
        raise ValueError(msg)

    for old_input, new_input_name in zip(torch_onnx_inputs, input_names):
        old_input_name = old_input.name
        old_input.name = new_input_name
        if old_input_name in new_old_input_spec_mapping:
            new_old_input_spec_mapping[new_input_name] = new_old_input_spec_mapping.pop(
                old_input_name
            )

    for old_output, new_output_name in zip(torch_onnx_outputs, output_names):
        old_output_name = old_output.name
        old_output.name = new_output_name
        if old_output_name in quantization_logs:
            quantization_logs[new_output_name] = quantization_logs.pop(old_output_name)

    return torch_onnx_model
