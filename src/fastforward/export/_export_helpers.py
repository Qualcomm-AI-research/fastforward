# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import logging

from typing import Any, Sequence, TypedDict

import torch

from onnx.onnx_ml_pb2 import ModelProto
from onnxscript.ir import Model
from torch.export.graph_signature import InputSpec
from typing_extensions import NotRequired

from fastforward.common import ensure_tensor
from fastforward.exceptions import ExportError
from fastforward.quantization._quantizer_impl import _infer_offset
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
    activation_encodings: dict[str, tuple[QNNEncodingEntry, ...]]
    param_encodings: dict[str, tuple[QNNEncodingEntry, ...]]


class QuantParametersDict(TypedDict):
    scale: torch.Tensor | float
    offset: torch.Tensor | float | int | None
    num_bits: float | int
    tile_size: NotRequired[tuple[int]]
    output_dtype: NotRequired[torch.dtype]


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
    onnxscript_model: Model, quantization_logs: dict[str, Any]
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
        onnxscript_model: An onnxscript model
        quantization_logs: Dictionary containing quantization
            settings for the various inputs/activations/parameters
            to the onnxscript_model
    """
    initializers = onnxscript_model.graph.initializers
    used_parameters = set()
    unused_parameters = set()

    for initializer in initializers:
        if initializer in quantization_logs:
            used_parameters.add(initializer)
        else:
            unused_parameters.add(initializer)

    return used_parameters, unused_parameters


def _strict_cast_to_int(value: float | int, value_name: str) -> int:
    if not isinstance(value, int) and not value.is_integer():
        raise ExportError(
            f"QNN requires the {value_name} value to be an integer (instead got {value})"
        )

    return int(value)


def create_qnn_encoding_entry(
    encoding_value: QuantParametersDict,
) -> tuple[QNNEncodingEntry, ...]:
    """Converts an encoding value dictionary to a QNNEncodingEntry.

    Args:
        encoding_value: dictionary containing quantization parameters.

    Returns:
        tuple containing QNNEncodingEntry dictionaries.
    """
    scale = encoding_value["scale"]
    offset = encoding_value["offset"]
    bitwidth = encoding_value["num_bits"]
    int_min = integer_minimum(bitwidth)

    scale = ensure_tensor(scale)
    if offset is None:
        offset = _infer_offset(offset, scale)
    offset = ensure_tensor(offset)
    offset = torch.round(offset)

    int_min = _strict_cast_to_int(int_min, "int_min")
    bitwidth = _strict_cast_to_int(bitwidth, "bitwidth")

    qnn_offset = offset - 2 ** (bitwidth - 1)
    if not isinstance(qnn_offset, torch.Tensor):
        qnn_offset = torch.tensor(qnn_offset)

    min_range, max_range = quantization_range(scale, offset, bitwidth)
    min_range = ensure_tensor(min_range)
    max_range = ensure_tensor(max_range)

    encoding = []

    for scale_entry, offset_entry, original_offset_entry, min_range_entry, max_range_entry in zip(
        scale, qnn_offset, offset, min_range, max_range
    ):
        output_entry: QNNEncodingEntry = {
            "bitwidth": int(bitwidth),
            "dtype": "int",
            "is_symmetric": "True" if original_offset_entry == 0 else "False",
            "min": min_range_entry.item(),
            "max": max_range_entry.item(),
            "offset": int(offset_entry),
            "scale": scale_entry.item(),
        }
        encoding.append(output_entry)

    return tuple(encoding)


def generate_qnn_encodings_dictionary(
    inputs: set[str],
    activations: set[str],
    parameters: set[str],
    quantization_logs: dict[str, Any],
) -> QNNEncoding:
    param_encodings: dict[str, tuple[QNNEncodingEntry, ...]] = {}
    activation_encodings: dict[str, tuple[QNNEncodingEntry, ...]] = {}

    # Inputs are also included in the activation encodings for QNN
    activations_and_inputs = activations | inputs

    for key, value in quantization_logs.items():
        encoding = create_qnn_encoding_entry(value)

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
