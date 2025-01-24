# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import logging

from typing import Any, Sequence, TypedDict

import torch

from onnx.onnx_ml_pb2 import ModelProto
from onnxscript.ir import Model
from torch.export.graph_signature import InputSpec

from fastforward.quantization.affine import integer_minimum, quantization_range

logger = logging.getLogger(__name__)


class QNNEncodingEntry(TypedDict):
    bitwidth: int
    dtype: str
    is_symmetric: str
    max: float
    min: float
    offset: int
    scale: float


class QNNEncoding(TypedDict):
    activation_encodings: dict[str, list[QNNEncodingEntry]]
    param_encodings: dict[str, list[QNNEncodingEntry]]


def get_input_spec_new_old_mapping(
    old_input_specs: Sequence[InputSpec], new_input_specs: Sequence[InputSpec]
) -> dict[str, str]:
    if len(old_input_specs) != len(new_input_specs):
        raise IndexError(
            f"Detected different number of input specs before ({len(old_input_specs)}) "
            f"and after ({len(new_input_specs)}) applying graph operations. "
            "These need to be the same."
        )

    new_old_mapping = {}
    for old_input_spec, new_input_spec in zip(old_input_specs, new_input_specs):
        # Make sure we do not make some mistake and associate
        # arguments that have different targets.
        if old_input_spec.target != new_input_spec.target:
            raise RuntimeError(
                "The target for the same input spec before and after graph operations "
                f"has changed. InputSpec ({old_input_spec}) before had target: "
                f"{old_input_spec.target}, now it has target: {new_input_spec.target}."
            )
        old_name = getattr(old_input_spec.arg, "name", "")
        new_name = getattr(new_input_spec.arg, "name", "")

        new_old_mapping[new_name] = old_name

    return new_old_mapping


def get_inputs(
    onnxscript_model: Model,
    quantization_logs: dict[str, Any],
    new_old_mapping: dict[str, str],
) -> tuple[set[str], set[str]]:
    graph_inputs = onnxscript_model.graph.inputs
    used_input_nodes = set()
    unused_input_nodes = set()

    for graph_input in graph_inputs:
        new_arg_name = getattr(graph_input, "name", "")
        old_arg_name = new_old_mapping[new_arg_name]

        if old_arg_name not in quantization_logs:
            unused_input_nodes.add(new_arg_name)
        else:
            used_input_nodes.add(new_arg_name)
            update_arg_name_in_quantization_logs(old_arg_name, new_arg_name, quantization_logs)

    return used_input_nodes, unused_input_nodes


def update_arg_name_in_quantization_logs(
    old_arg_name: str, new_arg_name: str, quantization_logs: dict[str, Any]
) -> None:
    parameters = quantization_logs.pop(old_arg_name)
    quantization_logs[new_arg_name] = parameters


def get_activations(
    onnx_proto: ModelProto, quantization_logs: dict[str, Any]
) -> tuple[set[str], set[str]]:
    nodes = onnx_proto.graph.node
    used_activation_nodes = set()
    unused_activation_nodes = set()

    # For activation quantization, QNN is expecting the name of the node output in which
    # the quantization parameters will be applied. So, for each node in the ONNX graph we
    # grab its output (which name is already the same as in the dynamo graph, which is a
    # feature of the torch_onnx package). We also filter out, but keep the activations that do not
    # have quantization parameters as knowing these might be useful for bypassing in QNN.

    for node in nodes:
        for node_output in node.output:
            if node_output not in quantization_logs:
                unused_activation_nodes.add(node_output)
            else:
                used_activation_nodes.add(node_output)
    return used_activation_nodes, unused_activation_nodes


def get_parameters(
    onnxscript_model: Model, quantization_logs: dict[str, Any]
) -> tuple[set[str], set[str]]:
    # In ONNX the initializer entry of the graph contains the names of the model parameters.
    # Here we check which for which to which of these parameters quantization was applied, and
    # to which it was not (in case these are eventually needed to fill in some bypass instructions
    # in the QNN encodings file).

    initializers = onnxscript_model.graph.initializers
    used_parameters = set()
    unused_parameters = set()

    for initializer in initializers:
        if initializer not in quantization_logs:
            unused_parameters.add(initializer)
        else:
            used_parameters.add(initializer)

    return used_parameters, unused_parameters


def generate_qnn_encodings_dictionary(
    inputs: set[str],
    activations: set[str],
    parameters: set[str],
    quantization_logs: dict[str, Any],
) -> QNNEncoding:
    param_encodings: dict[str, list[QNNEncodingEntry]] = {}
    activation_encodings: dict[str, list[QNNEncodingEntry]] = {}

    # Inputs are also included in the activation encodings for QNN
    activations_and_inputs = activations | inputs

    # TODO: Test encodings generation format when using per channel quantization.
    for key, value in quantization_logs.items():
        scale = value["scale"]
        offset = value.get("offset")
        bitwidth = value["num_bits"]
        int_min = integer_minimum(bitwidth)
        if not isinstance(int_min, int):
            raise TypeError(
                "QNN requires the offset value to be an integer but the "
                "integer_mininum function returned a float."
            )
        if isinstance(offset, torch.Tensor):
            offset = torch.round(offset)

        qnn_offset = offset - 2 ** (bitwidth - 1)
        if not isinstance(qnn_offset, torch.Tensor):
            qnn_offset = torch.tensor(qnn_offset)

        min_range, max_range = quantization_range(scale, offset, bitwidth)

        encoding = []

        for s, o, oo, min_r, max_r in zip(scale, qnn_offset, offset, min_range, max_range):
            output_entry: QNNEncodingEntry = {
                "bitwidth": bitwidth,
                "dtype": "int",
                "is_symmetric": "True" if not oo else "False",
                "min": min_r.item(),
                "max": max_r.item(),
                "offset": int(o),
                "scale": s.item(),
            }
            encoding.append(output_entry)

        if key in activations_and_inputs:
            activation_encodings[key] = encoding
        elif key in parameters:
            param_encodings[key] = encoding
        else:
            logger.warning(
                f"Key: {key} not found in activations/inputs/parameters sets, "
                "and it will not be included in the encondigs file."
            )

    return {"param_encodings": param_encodings, "activation_encodings": activation_encodings}
