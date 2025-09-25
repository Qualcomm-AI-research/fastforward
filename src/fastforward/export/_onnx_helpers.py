# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Any

import onnx
import onnxscript


def _fix_reshape_allowzero(model: onnxscript.ir.Model) -> None:
    """Fix Reshape nodes with allowzero=1 for QNN compatibility."""
    num_nodes = len(model.graph)
    for i in range(num_nodes):
        node = model.graph.node(i)
        if node.op_type == "Reshape" and "allowzero" in node.attributes:
            del node.attributes["allowzero"]


def _rename_nodes_and_update_encodings(
    model_proto: onnx.onnx_ml_pb2.ModelProto, quantization_logs: dict[str, Any], name_prefix: str
) -> tuple[onnx.onnx_ml_pb2.ModelProto, dict[str, Any]]:
    """Rename ONNX nodes and update the encodings dictionary.

    Due to a QNN issue where some nodes with the same name as existing
    ones are created we update the onnx node names as well as the encodings.
    """
    model_proto = _fix_onnx_names(model_proto, name_prefix)
    quantization_logs = _fix_encoding_names(quantization_logs, name_prefix)
    return model_proto, quantization_logs


def _fix_onnx_names(
    model_proto: onnx.onnx_ml_pb2.ModelProto, new_name_prefix: str
) -> onnx.onnx_ml_pb2.ModelProto:
    name_map = {}

    # Rename initializers (weights/parameters)
    for initializer in model_proto.graph.initializer:
        old_name = initializer.name
        new_name = f"{new_name_prefix}_{old_name}_0"
        name_map[old_name] = new_name
        initializer.name = new_name

    for node in model_proto.graph.node:
        if node.name:
            node.name = f"{new_name_prefix}_{node.name}_0"

        for i, output in enumerate(node.output):
            new_name = f"{new_name_prefix}_{output}_0"
            name_map[output] = new_name
            node.output[i] = new_name

    for node in model_proto.graph.node:
        for i, input_name in enumerate(node.input):
            if input_name in name_map:
                node.input[i] = name_map[input_name]

    # Update graph inputs
    for input_tensor in model_proto.graph.input:
        if input_tensor.name in name_map:
            input_tensor.name = name_map[input_tensor.name]

    # Update graph outputs
    for output_tensor in model_proto.graph.output:
        if output_tensor.name in name_map:
            output_tensor.name = name_map[output_tensor.name]

    # Update value_info (intermediate tensors)
    for value_info in model_proto.graph.value_info:
        if value_info.name in name_map:
            value_info.name = name_map[value_info.name]

    return model_proto


def _fix_encoding_names(
    encodings_dictionary: dict[str, Any], new_name_prefix: str
) -> dict[str, Any]:
    new_encodings_dictionary: dict[str, Any] = {}

    for key, value in encodings_dictionary.items():
        new_encodings_dictionary[f"{new_name_prefix}_{key}_0"] = value
    return new_encodings_dictionary
