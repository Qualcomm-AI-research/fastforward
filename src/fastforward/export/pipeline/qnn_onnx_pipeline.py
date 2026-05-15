# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Any

from fastforward.export.pipeline.core import Pipeline
from fastforward.export.stages.base_pipeline_stages import (
    stage_capture_impl_ff,
    stage_cleanup_ff_quantizer_artifacts,
    stage_convert_captured_impl_ff,
    stage_passthrough_ff_module,
)
from fastforward.export.stages.onnx.onnx_export_stages import (
    stage_add_ff_quantization_metadata,
    stage_alter_onnx_node_names,
    stage_copy_metadata_from_ir_to_proto,
    stage_fix_onnx_reshape_allowzero,
    stage_fx_to_onnx_program,
    stage_onnx_program_to_proto,
    stage_onnx_proto_to_encodings,
    stage_rename_onnx_input_output_names,
    stage_save_onnx_proto,
)


def qnn_onnx_pipeline(pipeline_kwargs: dict[str, Any]) -> Pipeline:
    """Create ONNX/QNN pipeline starting from FastForward modules.

    Args:
        pipeline_kwargs: Dictionary of pipeline configuration parameters

    Returns:
        ONNX/QNN pipeline for FF modules
    """
    onnx_pipeline = Pipeline(pipeline_kwargs)

    source_ff_module_stage = onnx_pipeline.register_stage(
        stage_passthrough_ff_module, "source_ff_module"
    )
    capture_ff_stage = onnx_pipeline.register_stage(stage_capture_impl_ff, "capture_ff")
    convert_captured_ff_stage = onnx_pipeline.register_stage(
        stage_convert_captured_impl_ff, "convert_captured_ff"
    ).depends_on(capture_ff_stage)
    cleanup_ff_quantizer_artifacts_stage = onnx_pipeline.register_stage(
        stage_cleanup_ff_quantizer_artifacts, "cleanup_ff_quantizer_artifacts"
    ).depends_on(convert_captured_ff_stage, source_ff_module_stage)
    fx_to_onnx_program = onnx_pipeline.register_stage(
        stage_fx_to_onnx_program, "fx_to_onnx_program"
    ).depends_on(cleanup_ff_quantizer_artifacts_stage)
    add_ff_quantization_metadata = onnx_pipeline.register_stage(
        stage_add_ff_quantization_metadata, "add_ff_quantization_metadata"
    ).depends_on(fx_to_onnx_program, cleanup_ff_quantizer_artifacts_stage)
    alter_onnx_node_names = onnx_pipeline.register_stage(
        stage_alter_onnx_node_names, "alter_onnx_node_names"
    ).depends_on(add_ff_quantization_metadata)
    fix_onnx_reshape_allowzero = onnx_pipeline.register_stage(
        stage_fix_onnx_reshape_allowzero, "fix_onnx_reshape_allowzero"
    ).depends_on(alter_onnx_node_names)
    rename_onnx_input_output_names = onnx_pipeline.register_stage(
        stage_rename_onnx_input_output_names, "rename_onnx_input_output_names"
    ).depends_on(fix_onnx_reshape_allowzero)
    onnx_program_to_proto = onnx_pipeline.register_stage(
        stage_onnx_program_to_proto, "onnx_program_to_proto"
    ).depends_on(rename_onnx_input_output_names)
    copy_metadata_props_from_ir_to_proto = onnx_pipeline.register_stage(
        stage_copy_metadata_from_ir_to_proto, "copy_metadata_props_from_ir_to_proto"
    ).depends_on(rename_onnx_input_output_names, onnx_program_to_proto)
    save_onnx_proto = onnx_pipeline.register_stage(
        stage_save_onnx_proto, "save_onnx_proto"
    ).depends_on(copy_metadata_props_from_ir_to_proto)
    onnx_pipeline.register_stage(
        stage_onnx_proto_to_encodings, "onnx_proto_to_encodings", capture_stage_output=True
    ).depends_on(save_onnx_proto)

    return onnx_pipeline
