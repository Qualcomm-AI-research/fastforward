# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from types import SimpleNamespace
from typing import Any, cast

import onnx
import pytest
import torch

from fastforward.export.stages.onnx.onnx_export_stages import (
    _FF_DATA_SHAPE_KEY,
    _FF_FORMAT_KEY,
    _FF_FORMAT_VALUE_JSON,
    _FF_FORMAT_VERSION_KEY,
    _FF_FORMAT_VERSION_VALUE_V1,
    _FF_NUM_BITS_KEY,
    _FF_OFFSET_KEY,
    _FF_SCALE_KEY,
    _FF_SYMMETRIC_KEY,
    _FF_TILE_SIZE_KEY,
    OnnxIrToProtoMetadataCopier,
    OnnxQuantMetadataAnnotator,
    _metadata_props_to_dict,
    _quant_params_from_metadata_props,
    extract_qnn_encodings_from_onnx_proto,
    stage_add_ff_quantization_metadata,
    stage_alter_onnx_node_names,
    stage_fix_onnx_reshape_allowzero,
    stage_onnx_proto_to_encodings,
    stage_rename_onnx_input_output_names,
)
from onnx import TensorProto, helper
from onnxscript import ir


class _OnnxProgramStub:
    def __init__(self, model: ir.Model) -> None:
        self.model = model


def _model_proto_to_ir_model(model_proto: onnx.ModelProto) -> ir.Model:
    if hasattr(ir, "from_proto"):
        return ir.from_proto(model_proto)
    if hasattr(ir, "serde") and hasattr(ir.serde, "deserialize_model"):
        return ir.serde.deserialize_model(model_proto)
    msg = "Unsupported onnxscript.ir API: missing from_proto/serde.deserialize_model"
    raise RuntimeError(msg)


def _build_real_onnx_program_with_fx_module(
    *,
    attach_quant_spec: bool,
    include_shape_and_tile: bool,
) -> tuple[torch.onnx.ONNXProgram, torch.fx.GraphModule]:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    relu_node = helper.make_node("Relu", inputs=["x"], outputs=["y"], name="relu_node")
    graph = helper.make_graph([relu_node], "g", [x], [y], initializer=[])
    proto = helper.make_model(graph)
    ir_model = _model_proto_to_ir_model(proto)
    onnx_program = cast(torch.onnx.ONNXProgram, _OnnxProgramStub(ir_model))

    fx_graph = torch.fx.Graph()
    fx_input = fx_graph.placeholder("x")
    fx_node = fx_graph.call_function(torch.ops.aten.relu.default, args=(fx_input,))
    fx_graph.output(fx_node)
    fx_module = torch.fx.GraphModule(torch.nn.Module(), fx_graph)

    if attach_quant_spec:
        fx_input.meta["__FF_QUANTIZATION_SPEC"] = SimpleNamespace(
            scale=torch.tensor([0.5], dtype=torch.float32),
            offset=torch.tensor([10.0], dtype=torch.float32),
            num_bits=8,
            symmetric=False,
            data_shape=(1, 4) if include_shape_and_tile else None,
            tile_size=(1, 4) if include_shape_and_tile else None,
        )
        # Map ONNX graph input to FX placeholder quant spec.
        ir_model.graph.inputs[0].meta["node"] = fx_input

    return onnx_program, fx_module


def _build_real_onnx_program_with_initializer() -> torch.onnx.ONNXProgram:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    weight = helper.make_tensor(
        name="w",
        data_type=TensorProto.FLOAT,
        dims=[1, 4],
        vals=[1.0, 1.0, 1.0, 1.0],
    )
    add_node = helper.make_node("Add", inputs=["x", "w"], outputs=["y"], name="add_node")
    graph = helper.make_graph([add_node], "g_with_init", [x], [y], initializer=[weight])
    proto = helper.make_model(graph)
    return cast(torch.onnx.ONNXProgram, _OnnxProgramStub(_model_proto_to_ir_model(proto)))


def _build_real_onnx_program_with_reshape_allowzero() -> torch.onnx.ONNXProgram:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    shape = helper.make_tensor(name="shape", data_type=TensorProto.INT64, dims=[2], vals=[1, 4])
    reshape_node = helper.make_node(
        "Reshape", inputs=["x", "shape"], outputs=["y"], name="reshape_node", allowzero=1
    )
    graph = helper.make_graph([reshape_node], "g_reshape", [x], [y], initializer=[shape])
    proto = helper.make_model(graph)
    return cast(torch.onnx.ONNXProgram, _OnnxProgramStub(_model_proto_to_ir_model(proto)))


def _onnx_model(onnx_program: torch.onnx.ONNXProgram) -> ir.Model:
    return cast(ir.Model, getattr(onnx_program, "model"))


def _test_metadata_props_to_dict(item: object) -> dict[str, str]:
    metadata_props = getattr(item, "metadata_props", {})
    if isinstance(metadata_props, dict):
        return {str(k): str(v) for k, v in metadata_props.items()}
    output: dict[str, str] = {}
    for entry in metadata_props:
        if hasattr(entry, "key") and hasattr(entry, "value"):
            output[str(entry.key)] = str(entry.value)
    return output


def _set_metadata_props(item: Any, metadata: dict[str, str]) -> None:
    for key, value in metadata.items():
        metadata_prop = item.metadata_props.add()
        metadata_prop.key = key
        metadata_prop.value = value


def _ff_quant_metadata(
    *,
    scale: list[float] | list[list[float]],
    offset: list[float] | list[list[float]],
    num_bits: int,
    symmetric: bool,
    data_shape: list[int] | None = None,
    tile_size: list[int] | None = None,
) -> dict[str, str]:
    metadata = {
        _FF_SCALE_KEY: str(scale).replace("'", '"'),
        _FF_OFFSET_KEY: str(offset).replace("'", '"'),
        _FF_NUM_BITS_KEY: str(num_bits),
        _FF_SYMMETRIC_KEY: "true" if symmetric else "false",
    }
    if data_shape is not None:
        metadata[_FF_DATA_SHAPE_KEY] = str(data_shape)
    if tile_size is not None:
        metadata[_FF_TILE_SIZE_KEY] = str(tile_size)
    return metadata


def _collect_annotated_items(onnx_program: torch.onnx.ONNXProgram) -> list[dict[str, str]]:
    model = _onnx_model(onnx_program)
    metadata_dicts: list[dict[str, str]] = []

    for node in model.graph:
        metadata = _test_metadata_props_to_dict(node)
        if _FF_FORMAT_KEY in metadata:
            metadata_dicts.append(metadata)

    for value in model.graph.inputs:
        metadata = _test_metadata_props_to_dict(value)
        if _FF_FORMAT_KEY in metadata:
            metadata_dicts.append(metadata)

    for value in model.graph.outputs:
        metadata = _test_metadata_props_to_dict(value)
        if _FF_FORMAT_KEY in metadata:
            metadata_dicts.append(metadata)

    for value in model.graph.initializers.values():
        metadata = _test_metadata_props_to_dict(value)
        if _FF_FORMAT_KEY in metadata:
            metadata_dicts.append(metadata)

    return metadata_dicts


def test_metadata_props_to_dict_returns_empty_when_item_has_no_metadata_props() -> None:
    graph = torch.fx.Graph()
    item = graph.placeholder("x")
    assert _metadata_props_to_dict(item) == {}


def test_metadata_props_to_dict_handles_mapping_and_sequence_forms() -> None:
    mapping_item = SimpleNamespace(metadata_props={"k1": 1, "k2": True})
    assert _metadata_props_to_dict(mapping_item) == {"k1": "1", "k2": "True"}

    sequence_item = SimpleNamespace(
        metadata_props=[
            SimpleNamespace(key="a", value=10),
            SimpleNamespace(key="b", value=False),
        ]
    )
    assert _metadata_props_to_dict(sequence_item) == {"a": "10", "b": "False"}


def test_quant_params_from_metadata_props_returns_none_when_required_keys_missing() -> None:
    metadata = {
        _FF_SCALE_KEY: "[0.5]",
        _FF_OFFSET_KEY: "[10.0]",
    }
    assert _quant_params_from_metadata_props(metadata) is None


def test_quant_params_from_metadata_props_parses_and_converts_asymmetric_offsets() -> None:
    metadata = _ff_quant_metadata(
        scale=[0.5, 0.25],
        offset=[10.0, 11.0],
        num_bits=8,
        symmetric=False,
        data_shape=[1, 2],
        tile_size=[1, 1],
    )

    quant_params = _quant_params_from_metadata_props(metadata)

    assert quant_params is not None
    assert isinstance(quant_params["scale"], torch.Tensor)
    assert isinstance(quant_params["offset"], torch.Tensor)
    assert torch.equal(quant_params["scale"], torch.tensor([0.5, 0.25]))
    # asymmetric offset is converted back as: 2^(num_bits-1) - offset
    assert torch.equal(quant_params["offset"], torch.tensor([118.0, 117.0]))
    assert quant_params["num_bits"] == 8
    assert quant_params["data_shape"] == (1, 2)
    assert quant_params["tile_size"] == (1, 1)


def test_quant_params_from_metadata_props_keeps_shapes_none_when_not_provided() -> None:
    per_tensor = _quant_params_from_metadata_props(
        _ff_quant_metadata(scale=[0.5], offset=[10.0], num_bits=8, symmetric=True)
    )
    assert per_tensor is not None
    assert per_tensor.get("data_shape") is None
    assert per_tensor.get("tile_size") is None

    per_channel = _quant_params_from_metadata_props(
        _ff_quant_metadata(scale=[0.5, 0.25], offset=[10.0, 11.0], num_bits=8, symmetric=True)
    )
    assert per_channel is not None
    assert per_channel.get("data_shape") is None
    assert per_channel.get("tile_size") is None


def test_quant_params_from_metadata_props_raises_for_mismatched_scale_offset_sizes() -> None:
    metadata = _ff_quant_metadata(scale=[0.5, 0.25], offset=[10.0], num_bits=8, symmetric=True)
    with pytest.raises(
        RuntimeError, match="Expected scale and offset to have same num of elements"
    ):
        _quant_params_from_metadata_props(metadata)


def test_extract_qnn_encodings_from_onnx_proto_extracts_initializer_input_node_and_output_metadata() -> (
    None
):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    w_input = helper.make_tensor_value_info("w", TensorProto.FLOAT, [1, 4])

    weight = helper.make_tensor(
        name="w",
        data_type=TensorProto.FLOAT,
        dims=[1, 4],
        vals=[1.0, 1.0, 1.0, 1.0],
    )

    relu_node = helper.make_node("Relu", inputs=["x"], outputs=["relu_out"], name="relu_node")
    graph = helper.make_graph([relu_node], "g", [x, w_input], [y], initializer=[weight])
    proto = helper.make_model(graph)

    _set_metadata_props(
        proto.graph.initializer[0],
        _ff_quant_metadata(
            scale=[0.5], offset=[10.0], num_bits=8, symmetric=True, data_shape=[1], tile_size=[1]
        ),
    )
    _set_metadata_props(
        proto.graph.input[0],
        _ff_quant_metadata(
            scale=[0.25], offset=[11.0], num_bits=8, symmetric=True, data_shape=[1], tile_size=[1]
        ),
    )
    _set_metadata_props(
        proto.graph.node[0],
        _ff_quant_metadata(
            scale=[0.125], offset=[12.0], num_bits=8, symmetric=True, data_shape=[1], tile_size=[1]
        ),
    )
    _set_metadata_props(
        proto.graph.output[0],
        _ff_quant_metadata(
            scale=[0.0625], offset=[13.0], num_bits=8, symmetric=True, data_shape=[1], tile_size=[1]
        ),
    )

    encodings = extract_qnn_encodings_from_onnx_proto(proto)

    param_names = {entry["name"] for entry in encodings["param_encodings"]}
    activation_names = {entry["name"] for entry in encodings["activation_encodings"]}

    # initializer metadata becomes parameter encoding
    assert "w" in param_names
    # graph input metadata becomes activation encoding
    assert "x" in activation_names
    # node metadata applies to all outputs of that node
    assert "relu_out" in activation_names
    # graph output metadata is included
    assert "y" in activation_names
    # input name that is also an initializer is skipped in graph-input pass
    assert sum(1 for entry in encodings["activation_encodings"] if entry["name"] == "w") == 0


def test_extract_qnn_encodings_from_onnx_proto_raises_on_invalid_json_metadata() -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    relu_node = helper.make_node("Relu", inputs=["x"], outputs=["y"], name="relu_node")
    graph = helper.make_graph([relu_node], "g", [x], [y], initializer=[])
    proto = helper.make_model(graph)

    _set_metadata_props(
        proto.graph.input[0],
        {
            _FF_SCALE_KEY: "not-json",
            _FF_OFFSET_KEY: "[10.0]",
            _FF_NUM_BITS_KEY: "8",
            _FF_SYMMETRIC_KEY: "true",
        },
    )

    with pytest.raises(ValueError, match="Invalid JSON metadata value"):
        extract_qnn_encodings_from_onnx_proto(proto)


def test_stage_onnx_proto_to_encodings_returns_encodings_dictionary() -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    relu_node = helper.make_node("Relu", inputs=["x"], outputs=["y"], name="relu_node")
    graph = helper.make_graph([relu_node], "g", [x], [y], initializer=[])
    proto = helper.make_model(graph)

    _set_metadata_props(
        proto.graph.input[0],
        _ff_quant_metadata(
            scale=[0.5], offset=[10.0], num_bits=8, symmetric=True, data_shape=[1], tile_size=[1]
        ),
    )

    encodings = stage_onnx_proto_to_encodings((proto,), sample_inputs=[], context={})

    assert isinstance(encodings, dict)
    assert "activation_encodings" in encodings
    assert "param_encodings" in encodings


def _assert_common_metadata_present(metadata_props: dict[str, str]) -> None:
    assert metadata_props[_FF_FORMAT_KEY] == _FF_FORMAT_VALUE_JSON
    assert metadata_props[_FF_FORMAT_VERSION_KEY] == _FF_FORMAT_VERSION_VALUE_V1
    assert _FF_SCALE_KEY in metadata_props
    assert _FF_OFFSET_KEY in metadata_props
    assert _FF_NUM_BITS_KEY in metadata_props
    assert _FF_SYMMETRIC_KEY in metadata_props


def test_onnx_quant_metadata_annotator_adds_full_metadata_keys() -> None:
    # GIVEN: A real ONNX program whose FX input node has FF quantization metadata.
    onnx_program, fx_module = _build_real_onnx_program_with_fx_module(
        attach_quant_spec=True,
        include_shape_and_tile=True,
    )

    # WHEN: Annotating ONNX items with FF quantization metadata.
    OnnxQuantMetadataAnnotator(onnx_program, fx_module).annotate()
    annotated_metadata = _collect_annotated_items(onnx_program)

    # THEN: At least one item is annotated and includes full metadata keys.
    assert len(annotated_metadata) > 0
    for metadata in annotated_metadata:
        _assert_common_metadata_present(metadata)
        assert _FF_DATA_SHAPE_KEY in metadata
        assert _FF_TILE_SIZE_KEY in metadata


def test_onnx_quant_metadata_annotator_skips_items_without_quant_spec() -> None:
    # GIVEN: A real ONNX program with no FF quantization metadata on FX nodes.
    onnx_program, fx_module = _build_real_onnx_program_with_fx_module(
        attach_quant_spec=False,
        include_shape_and_tile=False,
    )

    # WHEN: Annotating ONNX items.
    OnnxQuantMetadataAnnotator(onnx_program, fx_module).annotate()
    annotated_metadata = _collect_annotated_items(onnx_program)

    # THEN: No item should be annotated.
    assert annotated_metadata == []


def test_stage_add_ff_quantization_metadata_returns_same_program_and_mutates_in_place() -> None:
    # GIVEN: A real ONNX program and matching FX module with quant metadata.
    onnx_program, fx_module = _build_real_onnx_program_with_fx_module(
        attach_quant_spec=True,
        include_shape_and_tile=True,
    )

    # WHEN: Running the stage-level metadata annotation function.
    output_program = stage_add_ff_quantization_metadata(
        (onnx_program, fx_module),
        sample_inputs=[],
        context={},
    )

    # THEN: The same ONNXProgram object is returned and metadata is present.
    assert output_program is onnx_program
    assert len(_collect_annotated_items(onnx_program)) > 0


def test_stage_rename_onnx_input_output_names_uses_existing_names_when_none_provided() -> None:
    # GIVEN: A real ONNX program with one input and one output.
    onnx_program, _ = _build_real_onnx_program_with_fx_module(
        attach_quant_spec=False,
        include_shape_and_tile=False,
    )
    model = _onnx_model(onnx_program)
    original_input_name = model.graph.inputs[0].name
    original_output_name = model.graph.outputs[0].name

    # WHEN: Running rename stage without input/output names in context.
    output_program = stage_rename_onnx_input_output_names(
        (onnx_program,),
        sample_inputs=[],
        context={},
    )

    # THEN: Names should remain unchanged and program should be returned in-place.
    assert output_program is onnx_program
    model = _onnx_model(onnx_program)
    assert model.graph.inputs[0].name == original_input_name
    assert model.graph.outputs[0].name == original_output_name


def test_stage_rename_onnx_input_output_names_renames_inputs_and_outputs() -> None:
    # GIVEN: A real ONNX program with one input and one output.
    onnx_program, _ = _build_real_onnx_program_with_fx_module(
        attach_quant_spec=False,
        include_shape_and_tile=False,
    )

    # WHEN: Running rename stage with explicit names.
    output_program = stage_rename_onnx_input_output_names(
        (onnx_program,),
        sample_inputs=[],
        context={"input_names": ["renamed_input"], "output_names": ["renamed_output"]},
    )

    # THEN: Input/output names should be updated.
    assert output_program is onnx_program
    model = _onnx_model(onnx_program)
    assert model.graph.inputs[0].name == "renamed_input"
    assert model.graph.outputs[0].name == "renamed_output"


def test_stage_rename_onnx_input_output_names_raises_on_count_mismatch() -> None:
    # GIVEN: A real ONNX program with one input and one output.
    onnx_program, _ = _build_real_onnx_program_with_fx_module(
        attach_quant_spec=False,
        include_shape_and_tile=False,
    )

    # WHEN/THEN: Providing mismatched numbers of names should raise.
    with pytest.raises(ValueError, match="does not match the number of graph inputs/outputs"):
        stage_rename_onnx_input_output_names(
            (onnx_program,),
            sample_inputs=[],
            context={"input_names": ["in0", "in1"], "output_names": ["out0"]},
        )


def test_stage_alter_onnx_node_names_alters_node_names_when_enabled() -> None:
    onnx_program = _build_real_onnx_program_with_initializer()

    stage_alter_onnx_node_names(
        (onnx_program,),
        sample_inputs=[],
        context={"alter_node_names": True},
    )

    node = next(iter(_onnx_model(onnx_program).graph))
    assert node.name is not None
    assert node.name.startswith("ff_")


def test_stage_fix_onnx_reshape_allowzero_removes_reshape_allowzero_attribute() -> None:
    onnx_program = _build_real_onnx_program_with_reshape_allowzero()
    node = next(iter(_onnx_model(onnx_program).graph))
    assert "allowzero" in node.attributes

    stage_fix_onnx_reshape_allowzero((onnx_program,), sample_inputs=[], context={})

    node = next(iter(_onnx_model(onnx_program).graph))
    assert "allowzero" not in node.attributes


def test_onnx_ir_to_proto_metadata_copier_copies_node_input_output_and_initializer_metadata() -> (
    None
):
    # GIVEN: A real ONNX IR model with metadata on node, input, output, and initializer.
    onnx_program = _build_real_onnx_program_with_initializer()
    ir_model = _onnx_model(onnx_program)
    ir_node = next(iter(ir_model.graph))
    ir_input = ir_model.graph.inputs[0]
    ir_output = ir_model.graph.outputs[0]
    ir_initializer = ir_model.graph.initializers["w"]

    ir_node.metadata_props["node_key"] = "node_value"
    ir_input.metadata_props["input_key"] = "input_value"
    ir_output.metadata_props["output_key"] = "output_value"
    ir_initializer.metadata_props["init_key"] = "init_value"

    proto = ir.to_proto(ir_model)

    # WHEN: Copying metadata from ONNX IR to ONNX proto.
    OnnxIrToProtoMetadataCopier(ir_model, proto).copy()

    # THEN: Metadata should be present on corresponding proto objects.
    proto_node_metadata = {entry.key: entry.value for entry in proto.graph.node[0].metadata_props}
    proto_input_metadata = {entry.key: entry.value for entry in proto.graph.input[0].metadata_props}
    proto_output_metadata = {
        entry.key: entry.value for entry in proto.graph.output[0].metadata_props
    }
    proto_initializer_metadata = {
        entry.key: entry.value for entry in proto.graph.initializer[0].metadata_props
    }

    assert proto_node_metadata["node_key"] == "node_value"
    assert proto_input_metadata["input_key"] == "input_value"
    assert proto_output_metadata["output_key"] == "output_value"
    assert proto_initializer_metadata["init_key"] == "init_value"


def test_onnx_ir_to_proto_metadata_copier_does_not_add_empty_metadata() -> None:
    # GIVEN: A real ONNX IR model with no metadata on graph entities.
    onnx_program = _build_real_onnx_program_with_initializer()
    ir_model = _onnx_model(onnx_program)
    proto = ir.to_proto(ir_model)

    # WHEN: Copying metadata from ONNX IR to ONNX proto.
    OnnxIrToProtoMetadataCopier(ir_model, proto).copy()

    # THEN: No metadata should be added to proto entities.
    assert len(proto.graph.node[0].metadata_props) == 0
    assert len(proto.graph.input[0].metadata_props) == 0
    assert len(proto.graph.output[0].metadata_props) == 0
    assert len(proto.graph.initializer[0].metadata_props) == 0
