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
    r"""Create the default FastForward -> ONNX -> QNN-encodings export pipeline.

    This pipeline captures a quantized FastForward model, removes runtime-only
    quantizer artifacts, emits an ONNX model, and writes a QNN-compatible
    encodings file extracted from ONNX metadata.

    High-level flow:

    ```text
    ff_model (pipeline input)
        |
        | +--> source_ff_module ---------------------+
        |                                            |
        | +--> capture_ff --> convert_captured_ff ---+
                                                     |
                                                     v
                                         cleanup_ff_quantizer_artifacts
                                                     |
                                                     v
                                                 fx_to_onnx_program
                                                     |
                                                     v
                                             add_ff_quantization_metadata
                                                     |
                                                     v
                                             alter_onnx_node_names
                                                     |
                                                     v
                                             fix_onnx_reshape_allowzero
                                                     |
                                                     v
                                         rename_onnx_input_output_names
                                                 /               \
                                                 v                 v
                                 onnx_program_to_proto      (renamed ONNX IR)
                                                 \               /
                                                 v             v
                                     copy_metadata_props_from_ir_to_proto
                                                     |
                                                     v
                                                 save_onnx_proto
                                                     |
                                                     v
                                             onnx_proto_to_encodings
    ```

    Why each stage exists:

    - `source_ff_module`:
      Keeps the original FF module available for multi-input stages that need both
      the captured graph and source module context.
    - `capture_ff`:
      Exports to FX graph form, annotates FF quantization specs, and propagates
      those specs through compatible view operations.
    - `convert_captured_ff`:
      Normalizes the captured representation to the concrete graph module format
      expected by downstream cleanup/export stages.
    - `cleanup_ff_quantizer_artifacts`:
      Removes unused quantizer references/`get_attr` artifacts so the graph is
      clean and backend-safe.
    - `fx_to_onnx_program`:
      Converts the cleaned FX graph to ONNX IR/program form.
    - `add_ff_quantization_metadata`:
      Transfers quantization specs from FX node metadata into ONNX metadata.
      This is the source of truth for downstream encodings extraction.
    - `alter_onnx_node_names`:
      Applies deterministic node renaming to avoid backend name collisions.
    - `fix_onnx_reshape_allowzero`:
      Normalizes `Reshape` allowzero semantics for backend compatibility.
    - `rename_onnx_input_output_names`:
      Applies user-provided graph input/output names (or preserves defaults).
    - `onnx_program_to_proto`:
      Materializes a protobuf model for serialization and metadata copying.
    - `copy_metadata_props_from_ir_to_proto`:
      Ensures metadata written on ONNX IR objects is preserved in the protobuf
      representation used for saving.
    - `save_onnx_proto`:
      Writes `<model_name>.onnx` to disk.
    - `onnx_proto_to_encodings`:
      Reads ONNX metadata and emits `<model_name>.encodings` in the configured
      schema handler format.

    Args:
        pipeline_kwargs: Pipeline context/configuration consumed by the registered
            stages (for example naming, export options, and schema handler).

    Returns:
        A configured `Pipeline` instance that produces ONNX and encodings artifacts
        for QNN ingestion.
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
