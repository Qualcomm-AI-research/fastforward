# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import json
import logging
import pathlib

from collections.abc import Mapping
from typing import Any, TypeAlias, cast

import onnx
import torch

from onnxscript import ir
from packaging import version

from fastforward.common import ensure_tensor
from fastforward.exceptions import ExportError
from fastforward.export._export_schemas import (
    EncodingSchemaHandler,
    LegacySchemaHandler,
    V1SchemaHandler,
    V2SchemaHandler,
)
from fastforward.export._export_types import QuantParametersDict
from fastforward.export._onnx_helpers import _fix_onnx_names, _fix_reshape_allowzero

logger = logging.getLogger(__name__)

_SampleInputsT: TypeAlias = list[tuple[tuple[Any, ...], dict[str, Any]]]

FF_QUANTIZATION_SPEC = "__FF_QUANTIZATION_SPEC"
_FF_SCALE_KEY = "pkg.fastforward.quantization.scale"
_FF_OFFSET_KEY = "pkg.fastforward.quantization.offset"
_FF_NUM_BITS_KEY = "pkg.fastforward.quantization.num_bits"
_FF_SYMMETRIC_KEY = "pkg.fastforward.quantization.symmetric"
_FF_DATA_SHAPE_KEY = "pkg.fastforward.quantization.data_shape"
_FF_TILE_SIZE_KEY = "pkg.fastforward.quantization.tile_size"
_FF_FORMAT_KEY = "pkg.fastforward.quantization.format"
_FF_FORMAT_VERSION_KEY = "pkg.fastforward.quantization.format_version"
_FF_FORMAT_VALUE_JSON = "json"
_FF_FORMAT_VERSION_VALUE_V1 = "1"


def _resolve_output_path_from_context(
    context: dict[str, Any],
    *,
    suffix: str,
) -> pathlib.Path:
    output_dir = context["output_dir"]
    model_name = context["model_name"]
    return pathlib.Path(output_dir) / f"{model_name}.{suffix}"


def _rename_onnx_input_output_names(
    model: ir.Model,
    input_names: list[str] | None,
    output_names: list[str] | None,
) -> None:
    graph_inputs = model.graph.inputs
    graph_outputs = model.graph.outputs

    effective_input_names = input_names
    if effective_input_names is None:
        effective_input_names = []
        for entry in graph_inputs:
            assert entry.name is not None
            effective_input_names.append(entry.name)

    effective_output_names = output_names
    if effective_output_names is None:
        effective_output_names = []
        for entry in graph_outputs:
            assert entry.name is not None
            effective_output_names.append(entry.name)

    if len(graph_inputs) != len(effective_input_names) or len(graph_outputs) != len(
        effective_output_names
    ):
        msg = (
            f"The number of user-defined inputs/outputs ({len(effective_input_names)}, {len(effective_output_names)}) "
            + "does not match the number of graph inputs/outputs "
            + f"({len(graph_inputs)}, {len(graph_outputs)})"
        )
        raise ValueError(msg)

    for old_input, new_input_name in zip(graph_inputs, effective_input_names):
        old_input.name = new_input_name

    for old_output, new_output_name in zip(graph_outputs, effective_output_names):
        old_output.name = new_output_name


def _metadata_props_to_dict(item: Any) -> dict[str, str]:
    """Extract metadata properties from an ONNX proto/IR object."""
    if not hasattr(item, "metadata_props"):
        return {}

    metadata_props = item.metadata_props
    if isinstance(metadata_props, Mapping):
        return {str(key): str(value) for key, value in metadata_props.items()}

    output: dict[str, str] = {}
    for entry in metadata_props:
        if hasattr(entry, "key") and hasattr(entry, "value"):
            output[str(entry.key)] = str(entry.value)

    return output


def _parse_metadata_literal(value: str) -> Any:
    """Parse metadata values from ONNX metadata strings as JSON."""
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON metadata value: {value!r}"
        raise ValueError(msg) from exc


def _quant_params_from_metadata_props(
    metadata_props: Mapping[str, str],
) -> QuantParametersDict | None:
    """Convert FF quantization metadata into generic quantization parameters."""
    required_keys = (_FF_SCALE_KEY, _FF_OFFSET_KEY, _FF_NUM_BITS_KEY, _FF_SYMMETRIC_KEY)
    if any(key not in metadata_props for key in required_keys):
        return None

    scale = ensure_tensor(_parse_metadata_literal(metadata_props[_FF_SCALE_KEY]))
    offset = ensure_tensor(_parse_metadata_literal(metadata_props[_FF_OFFSET_KEY]))
    num_bits = _parse_metadata_literal(metadata_props[_FF_NUM_BITS_KEY])

    symmetric = _parse_metadata_literal(metadata_props[_FF_SYMMETRIC_KEY])
    if isinstance(symmetric, str):
        symmetric = symmetric.lower() == "true"
    else:
        symmetric = bool(symmetric)

    scale = scale.reshape(-1)
    offset = offset.reshape(-1)

    if scale.numel() != offset.numel():
        msg = f"Expected scale and offset to have same num of elements but received, {scale.numel()} and {offset.numel()}"
        raise RuntimeError(msg)

    # The metadata stores the annotation-space offset. For asymmetric quantization,
    # convert back to the raw FF quantizer offset expected by schema handlers.
    if not symmetric:
        offset = 2 ** (int(num_bits) - 1) - offset

    data_shape: tuple[int, ...] | None = None
    tile_size: tuple[int, ...] | None = None

    raw_data_shape = metadata_props.get(_FF_DATA_SHAPE_KEY)
    raw_tile_size = metadata_props.get(_FF_TILE_SIZE_KEY)
    if raw_data_shape is not None and raw_tile_size is not None:
        parsed_data_shape = _parse_metadata_literal(raw_data_shape)
        parsed_tile_size = _parse_metadata_literal(raw_tile_size)
        if isinstance(parsed_data_shape, (tuple, list)) and isinstance(
            parsed_tile_size, (tuple, list)
        ):
            data_shape = tuple(int(dim) for dim in parsed_data_shape)
            tile_size = tuple(int(dim) for dim in parsed_tile_size)

    quant_params: dict[str, Any] = {
        "scale": scale,
        "offset": offset,
        "num_bits": num_bits,
        "tile_size": tile_size,
        "data_shape": data_shape,
    }
    return cast(QuantParametersDict, quant_params)


def extract_qnn_encodings_from_onnx_proto(
    onnx_proto: onnx.onnx_ml_pb2.ModelProto,
    encoding_schema_handler: EncodingSchemaHandler = V1SchemaHandler(),
) -> dict[str, Any]:
    """Extract FastForward quantization encodings from ONNX metadata annotations.

    Encodings are produced by the provided schema handler, mirroring the main
    `export()` flow schema behavior.
    """
    # Ensure deterministic output even when a reusable schema handler instance
    # is passed from pipeline context.
    encoding_schema_handler.clear()

    initializer_names = {initializer.name for initializer in onnx_proto.graph.initializer}

    for initializer in onnx_proto.graph.initializer:
        metadata_props = _metadata_props_to_dict(initializer)
        quant_params = _quant_params_from_metadata_props(metadata_props)
        if quant_params is not None:
            encoding_schema_handler.add_encoding(initializer.name, quant_params, is_param=True)

    for graph_input in onnx_proto.graph.input:
        if graph_input.name in initializer_names:
            continue
        metadata_props = _metadata_props_to_dict(graph_input)
        quant_params = _quant_params_from_metadata_props(metadata_props)
        if quant_params is not None:
            encoding_schema_handler.add_encoding(graph_input.name, quant_params, is_param=False)

    for node in onnx_proto.graph.node:
        metadata_props = _metadata_props_to_dict(node)
        quant_params = _quant_params_from_metadata_props(metadata_props)
        if quant_params is None:
            continue

        for output_name in node.output:
            encoding_schema_handler.add_encoding(output_name, quant_params, is_param=False)

    for graph_output in onnx_proto.graph.output:
        metadata_props = _metadata_props_to_dict(graph_output)
        quant_params = _quant_params_from_metadata_props(metadata_props)
        if quant_params is not None:
            encoding_schema_handler.add_encoding(graph_output.name, quant_params, is_param=False)

    return dict(encoding_schema_handler.build_encodings_dictionary())


class OnnxQuantMetadataAnnotator:
    """Annotate ONNX IR items with quantization metadata derived from FX graph metadata."""

    def __init__(
        self, onnx_program: torch.onnx.ONNXProgram, fx_module: torch.fx.GraphModule
    ) -> None:
        self._model = cast(ir.Model, getattr(onnx_program, "model"))
        self._fx_quant_specs: dict[str, Any] = {}
        self._fx_node_by_target: dict[str, torch.fx.Node] = {}
        self._onnx_nodes_by_name: dict[str, Any] = {}

        for fx_node in fx_module.graph.nodes:
            if FF_QUANTIZATION_SPEC in fx_node.meta:
                self._fx_quant_specs[fx_node.name] = fx_node.meta[FF_QUANTIZATION_SPEC]
            if fx_node.op == "get_attr":
                self._fx_node_by_target[str(fx_node.target)] = fx_node

        for onnx_node in self._model.graph:
            if hasattr(onnx_node, "name") and onnx_node.name is not None:
                self._onnx_nodes_by_name[str(onnx_node.name)] = onnx_node

    def annotate(self) -> None:
        """Annotate ONNX graph items with FastForward quantization metadata."""
        for onnx_node in self._model.graph:
            self._annotate_item(onnx_node)

        for input_value in self._model.graph.inputs:
            self._annotate_item(input_value)

        for output_value in self._model.graph.outputs:
            self._annotate_item(output_value)
            self._annotate_output_producer(output_value)

        for initializer in self._model.graph.initializers.values():
            self._annotate_item(initializer)

    def _annotate_output_producer(self, output_value: Any) -> None:
        if not hasattr(output_value, "producer") or not output_value.producer():
            return
        producer = output_value.producer()
        producer_name = getattr(producer, "name", None)
        if producer_name is None:
            return
        onnx_node = self._onnx_nodes_by_name.get(str(producer_name))
        if onnx_node is not None:
            self._annotate_item(onnx_node)

    def _annotate_item(self, onnx_item: Any) -> None:
        quant_spec = self._quant_spec_for_onnx_item(onnx_item)
        if quant_spec is not None:
            self._apply_quant_spec_metadata(onnx_item, quant_spec)

    def _quant_spec_for_onnx_item(self, onnx_item: Any) -> Any | None:
        if hasattr(onnx_item, "meta"):
            fx_node = onnx_item.meta.get("node")

            if fx_node is not None and fx_node.name not in self._fx_quant_specs:
                if hasattr(onnx_item, "name") and onnx_item.name in self._fx_node_by_target:
                    matched_node = self._fx_node_by_target[onnx_item.name]
                    if matched_node.name in self._fx_quant_specs:
                        fx_node = matched_node

            if fx_node is not None and fx_node.name in self._fx_quant_specs:
                return self._fx_quant_specs[fx_node.name]

            from_node_chain = onnx_item.meta.get("from_node", [])
            for entry in reversed(from_node_chain):
                if entry.name in self._fx_quant_specs:
                    return self._fx_quant_specs[entry.name]

        if hasattr(onnx_item, "name") and onnx_item.name in self._fx_quant_specs:
            return self._fx_quant_specs[onnx_item.name]

        if hasattr(onnx_item, "outputs"):
            for output_value in onnx_item.outputs:
                if output_value.name in self._fx_quant_specs:
                    return self._fx_quant_specs[output_value.name]

        return None

    def _apply_quant_spec_metadata(self, onnx_item: Any, quant_spec: Any) -> None:
        if not hasattr(onnx_item, "metadata_props"):
            return

        onnx_item.metadata_props[_FF_FORMAT_KEY] = _FF_FORMAT_VALUE_JSON
        onnx_item.metadata_props[_FF_FORMAT_VERSION_KEY] = _FF_FORMAT_VERSION_VALUE_V1
        onnx_item.metadata_props[_FF_SCALE_KEY] = json.dumps(
            quant_spec.scale.tolist() if hasattr(quant_spec.scale, "tolist") else quant_spec.scale
        )
        onnx_item.metadata_props[_FF_OFFSET_KEY] = json.dumps(
            quant_spec.offset.tolist()
            if hasattr(quant_spec.offset, "tolist")
            else quant_spec.offset
        )
        onnx_item.metadata_props[_FF_NUM_BITS_KEY] = json.dumps(quant_spec.num_bits)
        onnx_item.metadata_props[_FF_SYMMETRIC_KEY] = json.dumps(bool(quant_spec.symmetric))
        if quant_spec.data_shape is not None and quant_spec.tile_size is not None:
            onnx_item.metadata_props[_FF_DATA_SHAPE_KEY] = json.dumps([
                int(dim) for dim in quant_spec.data_shape
            ])
            onnx_item.metadata_props[_FF_TILE_SIZE_KEY] = json.dumps([
                int(dim) for dim in quant_spec.tile_size
            ])


def stage_fx_to_onnx_program(
    modules: tuple[torch.fx.GraphModule, ...],
    sample_inputs: _SampleInputsT,
    context: dict[str, Any],
) -> torch.onnx.ONNXProgram:
    """Export a captured FX graph module to an ONNX program.

    Args:
        modules: Tuple containing a single calibrated torch.fx.GraphModule
        sample_inputs: List of sample input tuples for tracing
        context: Pipeline context dictionary

    Returns:
        ONNX program ready for post-export processing
    """
    if version.parse(torch.__version__) < version.parse("2.5"):
        msg = (
            "Export functionality is only supported for PyTorch version 2.5 and above. "
            "Please upgrade your PyTorch installation to use this feature."
        )
        raise ExportError(msg)

    module = modules[0]

    # Get the first sample input for ONNX export
    if len(sample_inputs) == 0:
        raise ValueError("sample_inputs cannot be empty for ONNX export")

    sample_args, sample_kwargs = sample_inputs[0]

    onnx_export_options = context.get("onnx_export_options", {})
    if not isinstance(onnx_export_options, dict):
        msg = "`onnx_export_options` must be a dictionary if provided in context."
        raise TypeError(msg)

    verbose = context.get("verbose")

    export_kwargs = {
        "dynamo": True,
        "verbose": verbose,
        **onnx_export_options,
    }

    # Use the new torch.onnx.export API with dynamo=True.
    # NOTE: callers can pass opset/version workarounds through
    # `context["onnx_export_options"]` when needed.
    onnx_program = torch.onnx.export(
        module,
        args=sample_args,
        kwargs=sample_kwargs,  # type: ignore[call-arg,unused-ignore]
        **export_kwargs,
    )
    if onnx_program is None:
        raise ExportError("torch.onnx.export returned None")

    return onnx_program


def stage_add_ff_quantization_metadata(
    modules: tuple[torch.onnx.ONNXProgram, torch.fx.GraphModule],
    sample_inputs: _SampleInputsT,
    context: dict[str, Any],
) -> torch.onnx.ONNXProgram:
    """Add FastForward quantization metadata to an ONNX program."""
    del sample_inputs, context
    onnx_program, fx_module = modules
    onnx_annotator = OnnxQuantMetadataAnnotator(onnx_program, fx_module)
    onnx_annotator.annotate()
    return onnx_program


def stage_rename_onnx_input_output_names(
    modules: tuple[torch.onnx.ONNXProgram, ...],
    sample_inputs: _SampleInputsT,
    context: dict[str, Any],
) -> torch.onnx.ONNXProgram:
    """Rename ONNX graph inputs and outputs using pipeline context."""
    del sample_inputs
    (onnx_program,) = modules

    input_names = context.get("input_names")
    if input_names is not None and (
        not isinstance(input_names, list) or any(not isinstance(name, str) for name in input_names)
    ):
        msg = "`input_names` must be a list of strings if provided in context."
        raise TypeError(msg)

    output_names = context.get("output_names")
    if output_names is not None and (
        not isinstance(output_names, list)
        or any(not isinstance(name, str) for name in output_names)
    ):
        msg = "`output_names` must be a list of strings if provided in context."
        raise TypeError(msg)

    model = cast(ir.Model, getattr(onnx_program, "model"))
    _rename_onnx_input_output_names(model, input_names, output_names)
    return onnx_program


def stage_fix_onnx_reshape_allowzero(
    modules: tuple[torch.onnx.ONNXProgram, ...],
    sample_inputs: _SampleInputsT,
    context: dict[str, Any],
) -> torch.onnx.ONNXProgram:
    """Fix ONNX Reshape nodes with allowzero for QNN compatibility."""
    del sample_inputs, context
    (onnx_program,) = modules
    model = cast(ir.Model, getattr(onnx_program, "model"))
    _fix_reshape_allowzero(model)
    return onnx_program


def stage_alter_onnx_node_names(
    modules: tuple[torch.onnx.ONNXProgram, ...],
    sample_inputs: _SampleInputsT,
    context: dict[str, Any],
) -> torch.onnx.ONNXProgram:
    """Optionally alter ONNX node names in-place using context configuration."""
    del sample_inputs
    (onnx_program,) = modules

    alter_node_names = context.get("alter_node_names", False)
    if not alter_node_names:
        return onnx_program

    alter_node_names_prefix = context.get("alter_node_names_prefix", "ff")
    if not isinstance(alter_node_names_prefix, str):
        msg = "`alter_node_names_prefix` must be a string if provided in context."
        raise TypeError(msg)

    model = cast(ir.Model, getattr(onnx_program, "model"))
    _fix_onnx_names(model, alter_node_names_prefix)
    return onnx_program


def stage_onnx_program_to_proto(
    modules: tuple[torch.onnx.ONNXProgram, ...],
    sample_inputs: _SampleInputsT,
    context: dict[str, Any],
) -> onnx.onnx_ml_pb2.ModelProto:
    """Convert an ONNXProgram IR model into a serialized ONNX ModelProto."""
    del sample_inputs, context
    module = modules[0]
    model = cast(ir.Model, getattr(module, "model"))
    proto = ir.to_proto(model)

    return proto


def stage_copy_metadata_from_ir_to_proto(
    modules: tuple[torch.onnx.ONNXProgram, onnx.onnx_ml_pb2.ModelProto],
    sample_inputs: _SampleInputsT,
    context: dict[str, Any],
) -> onnx.onnx_ml_pb2.ModelProto:
    """Copy metadata props from ONNX IR objects to matching proto graph objects."""
    del sample_inputs, context
    onnx_program, proto = modules
    model = cast(ir.Model, getattr(onnx_program, "model"))
    OnnxIrToProtoMetadataCopier(model, proto).copy()

    return proto


def stage_save_onnx_proto(
    modules: tuple[onnx.onnx_ml_pb2.ModelProto],
    sample_inputs: _SampleInputsT,
    context: dict[str, Any],
) -> onnx.onnx_ml_pb2.ModelProto:
    """Persist an ONNX ModelProto to disk using context-controlled save options."""
    del sample_inputs
    proto = modules[0]

    onnx_output_path = _resolve_output_path_from_context(context, suffix="onnx")
    onnx_save_kwargs = context.get("onnx_save_kwargs", {})
    if not isinstance(onnx_save_kwargs, dict):
        msg = "Expected context['onnx_save_kwargs'] to be type dict, but got type %s" % (
            type(onnx_save_kwargs).__name__
        )
        raise TypeError(msg)

    save_kwargs: dict[str, Any] = {
        "save_as_external_data": context.get("save_as_external_data", True),
        "all_tensors_to_one_file": context.get("all_tensors_to_one_file", True),
        "location": str(context.get("onnx_external_data_location", onnx_output_path.stem)),
    }
    save_kwargs.update(onnx_save_kwargs)
    onnx_output_path.parent.mkdir(exist_ok=True, parents=True)

    onnx.save(
        proto,
        str(onnx_output_path),
        **save_kwargs,
    )
    return proto


class OnnxIrToProtoMetadataCopier:
    """Copy metadata properties from ONNX IR items to ONNX ModelProto items."""

    def __init__(self, ir_model: ir.Model, proto: onnx.onnx_ml_pb2.ModelProto) -> None:
        self._ir_model = ir_model
        self._proto = proto

    def copy(self) -> None:
        """Copy metadata from IR initializers/nodes/inputs/outputs into proto counterparts."""
        # The ir.to_proto() function does not preserve metadata_props for all items.
        self._copy_initializers()
        self._copy_nodes()
        self._copy_inputs()
        self._copy_outputs()

    @staticmethod
    def _copy_metadata(src: Any, dst: Any) -> None:
        if hasattr(src, "metadata_props") and src.metadata_props:
            for key, value in src.metadata_props.items():
                metadata_prop = dst.metadata_props.add()
                metadata_prop.key = key
                metadata_prop.value = value

    def _copy_initializers(self) -> None:
        ir_initializers_by_name = {
            init.name: init for init in self._ir_model.graph.initializers.values()
        }
        for proto_init in self._proto.graph.initializer:
            if proto_init.name in ir_initializers_by_name:
                self._copy_metadata(ir_initializers_by_name[proto_init.name], proto_init)

    def _copy_nodes(self) -> None:
        ir_nodes_by_name = {
            node.name: node for node in self._ir_model.graph if hasattr(node, "name")
        }
        for proto_node in self._proto.graph.node:
            if proto_node.name in ir_nodes_by_name:
                self._copy_metadata(ir_nodes_by_name[proto_node.name], proto_node)

    def _copy_inputs(self) -> None:
        ir_inputs_by_name = {
            value.name: value for value in self._ir_model.graph.inputs if value.name is not None
        }
        for proto_input in self._proto.graph.input:
            if proto_input.name in ir_inputs_by_name:
                self._copy_metadata(ir_inputs_by_name[proto_input.name], proto_input)

    def _copy_outputs(self) -> None:
        ir_outputs_by_name = {
            value.name: value for value in self._ir_model.graph.outputs if value.name is not None
        }
        for proto_output in self._proto.graph.output:
            if proto_output.name in ir_outputs_by_name:
                self._copy_metadata(ir_outputs_by_name[proto_output.name], proto_output)


def stage_onnx_proto_to_encodings(
    modules: tuple[onnx.onnx_ml_pb2.ModelProto, ...],
    sample_inputs: _SampleInputsT,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Extract quantization encodings from ONNX metadata in pipeline form.

    This stage consumes an ONNX `ModelProto` and produces an encodings dictionary,
    using a schema handler provided through `context["encoding_schema_handler"]`
    when available, or `V1SchemaHandler` by default.
    """
    del sample_inputs

    (onnx_proto,) = modules
    schema_handler = context.get("encoding_schema_handler")
    if schema_handler is None:
        schema_handler = V1SchemaHandler()

    if not isinstance(schema_handler, (LegacySchemaHandler, V1SchemaHandler, V2SchemaHandler)):
        msg = "`encoding_schema_handler` must be a supported schema handler instance."
        raise TypeError(msg)

    encodings_dictionary = extract_qnn_encodings_from_onnx_proto(
        onnx_proto,
        encoding_schema_handler=schema_handler,
    )

    output_path = _resolve_output_path_from_context(context, suffix="encodings")
    with open(output_path, "w") as fp:
        json.dump(encodings_dictionary, fp, indent=4)

    return encodings_dictionary
