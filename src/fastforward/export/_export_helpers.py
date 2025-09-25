# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import logging

from abc import ABC, abstractmethod
from typing import Any, Generic, Literal, Sequence, TypedDict, TypeVar

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


class LegacyQNNEncodingEntry(TypedDict):
    bitwidth: int
    dtype: str
    is_symmetric: str
    max: float
    min: float
    offset: int
    scale: float


class LegacyQNNEncoding(TypedDict):
    version: Literal["0.6.1"]
    activation_encodings: dict[str, tuple[LegacyQNNEncodingEntry, ...]]
    param_encodings: dict[str, tuple[LegacyQNNEncodingEntry, ...]]


class V1QNNEncodingEntry(TypedDict):
    name: str
    enc_type: Literal["PER_TENSOR", "PER_CHANNEL", "PER_BLOCK", "LPBQ"]
    dtype: Literal["INT", "FLOAT"]
    bw: int
    is_sym: bool
    scale: list[float]
    offset: list[int]
    block_size: NotRequired[int]
    compressed_bw: NotRequired[int]
    per_block_int_scale: NotRequired[list[int]]


class V1QNNEncoding(TypedDict):
    version: Literal["1.0.0"]
    activation_encodings: list[V1QNNEncodingEntry]
    param_encodings: list[V1QNNEncodingEntry]


class V2QNNEncodingEntry(TypedDict):
    name: str
    output_dtype: str  # format should be "int4", "uint4" etc
    y_scale: float | list[float]
    y_zero_point: NotRequired[int | list[int]]
    axis: NotRequired[int]
    block_size: NotRequired[int]
    per_block_int_scale: NotRequired[list[int]]
    per_channel_float_scale: NotRequired[list[float]]


class V2QNNEncoding(TypedDict):
    version: Literal["2.0.0"]
    encodings: list[V2QNNEncodingEntry]


class QuantParametersDict(TypedDict):
    scale: torch.Tensor | float
    offset: torch.Tensor | float | int | None
    num_bits: float | int
    tile_size: NotRequired[tuple[int]]
    output_dtype: NotRequired[torch.dtype]


QNNEncodingEntry = LegacyQNNEncodingEntry | V1QNNEncodingEntry | V2QNNEncodingEntry
QNNEncoding = LegacyQNNEncoding | V1QNNEncoding | V2QNNEncoding

T = TypeVar("T", bound=QNNEncoding)
E = TypeVar("E", bound=QNNEncodingEntry)


class EncodingSchemaHandler(ABC, Generic[T, E]):
    @abstractmethod
    def create_encoding_entry(self, key: str, encoding_value: QuantParametersDict) -> E | list[E]:
        pass

    @abstractmethod
    def generate_encodings_dictionary(
        self,
        inputs: set[str],
        activations: set[str],
        parameters: set[str],
        quantization_logs: dict[str, Any],
    ) -> T:
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        pass

    @abstractmethod
    def add_encoding_to_dictionary(
        self, encodings_dictionary: T, encoding_name: str, encoding_value: QuantParametersDict
    ) -> T:
        pass


class LegacySchemaHandler(EncodingSchemaHandler[LegacyQNNEncoding, LegacyQNNEncodingEntry]):
    """Schema handler for legacy QNN encoding format (version 0.6.1).

    This handler generates encodings compatible with legacy QNN versions that expect
    a simple two-category structure separating parameters and activations.

    Schema Structure:
    ================

    Top Level:
    ----------------
    {
        "activation_encodings": {<activation_name>: (<encoding_entry>, ...)},
        "param_encodings": {<parameter_name>: (<encoding_entry>, ...)}
    }

    Encoding Entry Format:
    ---------------------
    {
        "bitwidth": int,           # Quantization bit width (e.g., 8, 16)
        "dtype": str,              # Data type, always "int"
        "is_symmetric": str,       # "True" or "False" as string
        "max": float,              # Maximum quantization range value
        "min": float,              # Minimum quantization range value
        "offset": int,             # Zero-point offset (QNN format: offset - 2^(bitwidth-1))
        "scale": float             # Quantization scale factor
    }

    Note:
        Supports only per tensor and per channel quantization. Per channel is only supported
        for the first axis.
    """

    @property
    def version(self) -> Literal["0.6.1"]:
        return "0.6.1"

    def create_encoding_entry(
        self, key: str, encoding_value: QuantParametersDict
    ) -> list[LegacyQNNEncodingEntry]:
        """Converts an encoding value dictionary to a QNNEncodingEntry.

        Args:
            key: the name of the node associated with the encodings.
            encoding_value: dictionary containing quantization parameters.

        Returns:
            QNNEncodingEntry dictionaries.
        """
        del key
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

        for (
            scale_entry,
            offset_entry,
            original_offset_entry,
            min_range_entry,
            max_range_entry,
        ) in zip(scale, qnn_offset, offset, min_range, max_range):
            output_entry: LegacyQNNEncodingEntry = {
                "bitwidth": int(bitwidth),
                "dtype": "int",
                "is_symmetric": "True" if original_offset_entry == 0 else "False",
                "min": min_range_entry.item(),
                "max": max_range_entry.item(),
                "offset": int(offset_entry),
                "scale": scale_entry.item(),
            }
            encoding.append(output_entry)

        return encoding

    def generate_encodings_dictionary(
        self,
        inputs: set[str],
        activations: set[str],
        parameters: set[str],
        quantization_logs: dict[str, Any],
    ) -> LegacyQNNEncoding:
        param_encodings: dict[str, tuple[LegacyQNNEncodingEntry, ...]] = {}
        activation_encodings: dict[str, tuple[LegacyQNNEncodingEntry, ...]] = {}

        # Inputs are also included in the activation encodings for QNN
        activations_and_inputs = activations | inputs

        for key, value in quantization_logs.items():
            encoding = self.create_encoding_entry(key, value)

            if key in activations_and_inputs:
                activation_encodings[key] = tuple(encoding)
            elif key in parameters:
                param_encodings[key] = tuple(encoding)
            else:
                logger.warning(
                    f"Key: {key} not found in activations/inputs/parameters sets, "
                    "and it will not be included in the encondigs file."
                )

        return {
            "version": self.version,
            "param_encodings": param_encodings,
            "activation_encodings": activation_encodings,
        }

    def add_encoding_to_dictionary(
        self,
        encodings_dictionary: LegacyQNNEncoding,
        encoding_name: str,
        encoding_value: QuantParametersDict,
    ) -> LegacyQNNEncoding:
        activation_encodings = encodings_dictionary["activation_encodings"]
        if encoding_name in activation_encodings:
            return encodings_dictionary

        new_activation_encoding = self.create_encoding_entry(encoding_name, encoding_value)
        activation_encodings[encoding_name] = tuple(new_activation_encoding)
        encodings_dictionary["activation_encodings"] = activation_encodings

        return encodings_dictionary


class V1SchemaHandler(EncodingSchemaHandler[V1QNNEncoding, V1QNNEncodingEntry]):
    """Schema handler for QNN encoding format version 1.0.0.

    This handler generates encodings using the first standardized QNN format.
    It is the first format to introduce versioning, and changes the activation
    and parameters structure to use lists instead of dictionaries.

    Schema Structure:
    ================

    Top Level:
    -----------
    {
        "version": "1.0.0", # Required! Otherwise QNN will revert to legacy version.
        "activation_encodings": [<encoding_entry>, ...],
        "param_encodings": [<encoding_entry>, ...]
    }

    Encoding Entry Format:
    ---------------------
    {
        "name": str,                          # Tensor/parameter name
        "enc_type": str,                      # "PER_TENSOR" | "PER_CHANNEL" | "PER_BLOCK" | "LPBQ"
        "dtype": str,                         # "INT" | "FLOAT"
        "bw": int,                            # Bit width (e.g., 8, 16)
        "is_sym": bool,                       # True for symmetric, False for asymmetric
        "scale": [float, ...],                # List of scale values (per-channel if multiple)
        "offset": [int, ...],                 # List of offset values (QNN format)
        "block_size"?: int,                   # Optional: for PER_BLOCK quantization
        "compressed_bw"?: int,                # Optional: for compressed quantization
        "per_block_int_scale"?: [int, ...]    # Optional: for block-wise quantization
    }
    """

    @property
    def version(self) -> Literal["1.0.0"]:
        return "1.0.0"

    def create_encoding_entry(
        self, key: str, encoding_value: QuantParametersDict
    ) -> V1QNNEncodingEntry:
        """Converts an encoding value dictionary to a QNNEncodingEntry.

        Args:
            key: the name of the node associated with the encodings.
            encoding_value: dictionary containing quantization parameters.

        Returns:
            tuple containing QNNEncodingEntry dictionaries.
        """
        scale = encoding_value["scale"]
        offset = encoding_value["offset"]
        bitwidth = encoding_value["num_bits"]
        # tile_size = encoding_value.get("tile_size")

        scale = ensure_tensor(scale)
        if offset is None:
            offset = _infer_offset(offset, scale)
        offset = ensure_tensor(offset)
        offset = torch.round(offset)

        is_symmetric = (torch.all(offset)).item() == 0

        bitwidth = _strict_cast_to_int(bitwidth, "bitwidth")

        qnn_offset = offset - 2 ** (bitwidth - 1)
        if not isinstance(qnn_offset, torch.Tensor):
            qnn_offset = torch.tensor(qnn_offset)

        encoding: V1QNNEncodingEntry = {
            "name": key,
            "dtype": "INT",
            "enc_type": "PER_TENSOR" if len(scale) == 1 else "PER_CHANNEL",
            "is_sym": is_symmetric,
            "bw": int(bitwidth),
            "scale": [s.item() for s in scale],
            "offset": [o.item() for o in qnn_offset],
        }

        return encoding

    def generate_encodings_dictionary(
        self,
        inputs: set[str],
        activations: set[str],
        parameters: set[str],
        quantization_logs: dict[str, Any],
    ) -> V1QNNEncoding:
        param_encodings: list[V1QNNEncodingEntry] = []
        activation_encodings: list[V1QNNEncodingEntry] = []

        # Inputs are also included in the activation encodings for QNN
        activations_and_inputs = activations | inputs

        for key, value in quantization_logs.items():
            encoding = self.create_encoding_entry(key, value)

            if key in activations_and_inputs:
                activation_encodings.append(encoding)
            elif key in parameters:
                param_encodings.append(encoding)
            else:
                logger.warning(
                    f"Key: {key} not found in activations/inputs/parameters sets, "
                    "and it will not be included in the encondigs file."
                )

        return {
            "version": self.version,
            "param_encodings": param_encodings,
            "activation_encodings": activation_encodings,
        }

    def add_encoding_to_dictionary(
        self,
        encodings_dictionary: V1QNNEncoding,
        encoding_name: str,
        encoding_value: QuantParametersDict,
    ) -> V1QNNEncoding:
        activation_encodings = encodings_dictionary["activation_encodings"]
        existing_names = {encoding["name"] for encoding in activation_encodings}
        if encoding_name in existing_names:
            return encodings_dictionary

        new_activation_encoding = self.create_encoding_entry(encoding_name, encoding_value)
        activation_encodings.append(new_activation_encoding)
        encodings_dictionary["activation_encodings"] = activation_encodings

        return encodings_dictionary


class V2SchemaHandler(EncodingSchemaHandler[V2QNNEncoding, V2QNNEncodingEntry]):
    """Schema handler for QNN encoding format version 2.0.0.

    This handler generates encodings using the second QNN format version that consolidates
    all encodings into a single list and uses more standardized field names aligned with
    ONNX quantization conventions.

    Schema Structure:
    ================

    Top Level:
    -----------
    {
        "version": "2.0.0", # Required! Otherwise QNN will fall back to the legacy version.
        "encodings": [<encoding_entry>, ...]
    }

    Encoding Entry Format:
    ---------------------
    {
        "name": str,                                # Tensor/parameter name
        "output_dtype": str,                        # "int8", "uint8", "int4", "uint4", etc.
        "y_scale": float | [float, ...],            # Scale value(s) - single or per-channel
        "y_zero_point"?: int | [int, ...],          # Optional: zero-point(s) for asymmetric
        "axis"?: int,                               # Optional: quantization axis for per-channel
        "block_size"?: int,                         # Optional: for block-wise quantization
        "per_block_int_scale"?: [int, ...],         # Optional: integer scales for blocks
        "per_channel_float_scale"?: [float, ...]    # Optional: float scales for channels
    }
    """

    @property
    def version(self) -> Literal["2.0.0"]:
        return "2.0.0"

    def create_encoding_entry(
        self, key: str, encoding_value: QuantParametersDict
    ) -> V2QNNEncodingEntry:
        """Converts an encoding value dictionary to a QNNEncodingEntry.

        Args:
            key: the name of the node associated with the encodings.
            encoding_value: dictionary containing quantization parameters.

        Returns:
            tuple containing QNNEncodingEntry dictionaries.
        """
        scale = encoding_value["scale"]
        offset = encoding_value["offset"]
        bitwidth = encoding_value["num_bits"]
        tile_size = encoding_value.get("tile_size")

        scale = ensure_tensor(scale)
        if offset is None:
            offset = _infer_offset(offset, scale)
        offset = ensure_tensor(offset)
        offset = torch.round(offset)

        is_symmetric = (torch.all(offset)).item() == 0

        bitwidth = _strict_cast_to_int(bitwidth, "bitwidth")

        qnn_offset = offset - 2 ** (bitwidth - 1)
        if not isinstance(qnn_offset, torch.Tensor):
            qnn_offset = torch.tensor(qnn_offset)

        encoding: V2QNNEncodingEntry = {
            "name": key,
            "output_dtype": ("" if is_symmetric else "u") + f"int{bitwidth}",
            "y_scale": [s.item() for s in scale] if len(scale) > 1 else scale.item(),
        }

        if not is_symmetric:
            encoding["y_zero_point"] = (
                [int(o.item()) for o in qnn_offset]
                if len(qnn_offset) > 1
                else int(qnn_offset.item())
            )

        if len(scale) > 1 and tile_size is not None:
            channel_axis = next((i for i, value in enumerate(tile_size) if value == 1), None)
            if channel_axis is not None:
                encoding["axis"] = channel_axis

        return encoding

    def generate_encodings_dictionary(
        self,
        inputs: set[str],
        activations: set[str],
        parameters: set[str],
        quantization_logs: dict[str, Any],
    ) -> V2QNNEncoding:
        encodings: list[V2QNNEncodingEntry] = []

        # Inputs are also included in the activation encodings for QNN
        activations_and_inputs = activations | inputs

        for key, value in quantization_logs.items():
            encoding = self.create_encoding_entry(key, value)

            if key in activations_and_inputs:
                encodings.append(encoding)
            elif key in parameters:
                encodings.append(encoding)
            else:
                logger.warning(
                    f"Key: {key} not found in activations/inputs/parameters sets, "
                    "and it will not be included in the encondigs file."
                )

        return {"version": self.version, "encodings": encodings}

    def add_encoding_to_dictionary(
        self,
        encodings_dictionary: V2QNNEncoding,
        encoding_name: str,
        encoding_value: QuantParametersDict,
    ) -> V2QNNEncoding:
        encodings = encodings_dictionary["encodings"]
        existing_names = {encoding["name"] for encoding in encodings}
        if encoding_name in existing_names:
            return encodings_dictionary

        new_encoding = self.create_encoding_entry(encoding_name, encoding_value)
        encodings.append(new_encoding)
        encodings_dictionary["encodings"] = encodings

        return encodings_dictionary


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
