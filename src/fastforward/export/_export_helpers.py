# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import logging

from typing import Any, Sequence

from onnx.onnx_ml_pb2 import ModelProto
from onnxscript.ir import Model
from torch.export.graph_signature import InputSpec

from fastforward.exceptions import ExportError

logger = logging.getLogger(__name__)


def get_input_spec_new_old_mapping(
    old_input_specs: Sequence[InputSpec], new_input_specs: Sequence[InputSpec]
) -> dict[str, str]:
    if len(old_input_specs) != len(new_input_specs):
        msg = (
            f"Detected different number of input specs before ({len(old_input_specs)}) "
            f"and after ({len(new_input_specs)}) applying graph operations. "
            "These need to be the same."
        )
        raise IndexError(msg)

    new_old_mapping = {}
    for old_input_spec, new_input_spec in zip(old_input_specs, new_input_specs):
        # Make sure we do not make some mistake and associate
        # arguments that have different targets.
        if old_input_spec.target != new_input_spec.target:
            msg = (
                "The target for the same input spec before and after graph operations "
                f"has changed. InputSpec ({old_input_spec}) before had target: "
                f"{old_input_spec.target}, now it has target: {new_input_spec.target}."
            )
            raise RuntimeError(msg)
        old_name = getattr(old_input_spec.arg, "name")
        new_name = getattr(new_input_spec.arg, "name")

        new_old_mapping[new_name] = old_name

    return new_old_mapping


def get_inputs(
    onnxscript_model: Model,
    quantization_logs: dict[str, Any],
    new_old_mapping: dict[str, str],
) -> tuple[set[str], set[str]]:
    """Retrieve a model's input nodes.

    Given a model this function checks whether its inputs
    have been quantized (they exist as entries to a quantization
    log), and they are assigned the correct names. The function
    will return a tuple of all inputs separated to two groups:

    1) inputs that are associated with user defined quantization
        parameters, ie quantized inputs.
    2) inputs that are not associated with user defined quantization
        parameters, ie unquantized inputs.

    Args:
        onnxscript_model: An onnxscript model
        quantization_logs: Dictionary containing quantization
            settings for the various inputs/activations/parameters
            to the onnxscript_model
        new_old_mapping: Dictionary containing the translation of
            the onnxscript model inputs/activations/parameters
            names to the updated name. (NOTE: The change in name
            can occur due to manipulation of the dynamo graph,
            either through custom operations, or through the
            usage of the `run_decompositions` method).
    """
    graph_inputs = onnxscript_model.graph.inputs
    used_input_nodes = set()
    unused_input_nodes = set()

    for graph_input in graph_inputs:
        new_arg_name = getattr(graph_input, "name")
        old_arg_name = new_old_mapping[new_arg_name]

        if old_arg_name in quantization_logs:
            used_input_nodes.add(new_arg_name)
            update_arg_name_in_quantization_logs(old_arg_name, new_arg_name, quantization_logs)
        else:
            unused_input_nodes.add(new_arg_name)

    return used_input_nodes, unused_input_nodes


def update_arg_name_in_quantization_logs(
    old_arg_name: str, new_arg_name: str, quantization_logs: dict[str, Any]
) -> None:
    parameters = quantization_logs.pop(old_arg_name)
    quantization_logs[new_arg_name] = parameters


def get_activations(
    onnx_proto: ModelProto, quantization_logs: dict[str, Any]
) -> tuple[set[str], set[str]]:
    """Retrieve a model's activation nodes.

    Given a model this function checks whether its activations
    have been quantized (they exist as entries to a quantization
    log).

    For activation quantization, QNN is expecting the name of the node
    output in which the quantization parameters will be applied. So,
    for each node in the ONNX graph we grab its output (which name
    is already the same as in the dynamo graph, which is a feature of the
    torch_onnx package). We also filter out, but keep the activations that do not
    have quantization parameters as knowing these might be useful for bypassing in QNN.

    The function will return a tuple of all inputs separated
    to two groups:

    1) activations that are associated with user defined quantization
        parameters, ie quantized activations.
    2) activations that are not associated with user defined quantization
        parameters, ie unquantized activations.

    Args:
        onnx_proto: An onnx protobuf model
        quantization_logs: Dictionary containing quantization
            settings for the various inputs/activations/parameters
            to the onnxscript_model
    """
    nodes = onnx_proto.graph.node
    used_activation_nodes = set()
    unused_activation_nodes = set()

    for node in nodes:
        for node_output in node.output:
            if node_output in quantization_logs:
                used_activation_nodes.add(node_output)
            else:
                unused_activation_nodes.add(node_output)
    return used_activation_nodes, unused_activation_nodes


def get_parameters(
    onnx_proto: ModelProto, quantization_logs: dict[str, Any]
) -> tuple[set[str], set[str]]:
    """Retrieve a model's parameter nodes.

    Given a model this function checks whether its parameters (
    ie weights, biases etc) have been quantized (they exist as
    entries to a quantization log). Note that in In ONNX the initializer
    entry of the graph contains the names of the model parameters. We also
    filter out, but keep the parameters that do not have quantization
    parameters as knowing these might be useful for bypassing in QNN.

    The function will return a tuple of all parameters separated to two groups:

    1) parameters that are associated with user defined quantization
        parameters, ie quantized parameters.
    2) parameters that are not associated with user defined quantization
        parameters, ie unquantized parameters.

    Args:
        onnx_proto: An onnx protobuf model
        quantization_logs: Dictionary containing quantization
            settings for the various inputs/activations/parameters
            to the onnxscript_model
    """
    initializers = onnx_proto.graph.initializer
    used_parameters = set()
    unused_parameters = set()

    for initializer in initializers:
        initializer_name = initializer.name
        if initializer_name in quantization_logs:
            used_parameters.add(initializer_name)
        else:
            unused_parameters.add(initializer_name)

    return used_parameters, unused_parameters


def _strict_cast_to_int(value: float | int, value_name: str) -> int:
    if not isinstance(value, int) and not value.is_integer():
        msg = f"QNN requires the {value_name} value to be an integer (instead got {value})"
        raise ExportError(msg)

    return int(value)
