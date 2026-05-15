# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Any

from fastforward.export.pipeline.core import Pipeline
from fastforward.export.stages.base_pipeline_stages import (
    stage_capture_impl_ff,
    stage_convert_captured_impl_ff_qdq,
)
from fastforward.export.stages.onnx.onnx_export_stages import (
    stage_onnx_program_to_proto,
    stage_save_onnx_proto,
)
from fastforward.export.stages.onnx.onnx_qdq_export_stages import stage_fx_to_onnx_program_qdq


def qnn_onnx_qdq_pipeline(pipeline_kwargs: dict[str, Any]) -> Pipeline:
    r"""Create the FastForward -> ONNX QDQ export pipeline.

    This pipeline captures a quantized FastForward model and emits an ONNX
    model where each FF `quantize_by_tile` / `dequantize_by_tile` op is lowered
    to a standard ONNX `QuantizeLinear` / `DequantizeLinear` (QDQ) pair. Unlike
    `qnn_onnx_pipeline`, the quantization parameters live in the graph topology
    itself (as Q/DQ node inputs and initializers) rather than in side-channel
    `.encodings` metadata, so no encodings file is produced.

    High-level flow:

    ```text
    ff_model (pipeline input)
        |
        v
    capture_ff
        |
        v
    convert_captured_ff_qdq
        |
        v
    fx_to_onnx_program_qdq
        |
        v
    onnx_program_to_proto
        |
        v
    save_onnx_proto
    ```

    Why each stage exists:

    - `capture_ff`:
      Exports the FastForward QuantizedModule via `torch.export.export` and
      returns an `ExportedProgram` that retains FF quantize/dequantize ops as
      call_function nodes.
    - `convert_captured_ff_qdq`:
      Materializes an FX `GraphModule` from the captured program after running
      decompositions, but intentionally skips the FF quant-spec annotation /
      removal passes so that `fastforward::quantize_by_tile` and
      `fastforward::dequantize_by_tile` are preserved for downstream lowering.
    - `fx_to_onnx_program_qdq`:
      Wraps `fx_to_onnx_program` with a context that injects FF-specific custom
      lowerings (`ff_quantize_by_tile_onnx` / `ff_dequantize_by_tile_onnx`) into
      `onnx_export_options["custom_translation_table"]`. The wrapper builds a
      fresh context dict, so user-supplied export options are not mutated.
      Defaults `onnx_export_options["opset_version"]` to 21 when unset and
      rejects user-supplied values below 21 (INT4/INT16 in
      QuantizeLinear/DequantizeLinear were introduced in opset 21).
    - `onnx_program_to_proto`:
      Materializes a protobuf model for serialization.
    - `save_onnx_proto`:
      Writes `<model_name>.onnx` to disk.

    Args:
        pipeline_kwargs: Pipeline context/configuration consumed by the registered
            stages (for example naming, output directory, and onnx export options
            including opset version).

    Returns:
        A configured `Pipeline` instance that produces a QDQ-style ONNX artifact
        for QNN ingestion. No encodings file is produced; quantization
        parameters are embedded directly in the ONNX graph.
    """
    onnx_pipeline = Pipeline(pipeline_kwargs)

    capture_ff_stage = onnx_pipeline.register_stage(stage_capture_impl_ff, "capture_ff")
    convert_captured_ff_qdq_stage = onnx_pipeline.register_stage(
        stage_convert_captured_impl_ff_qdq, "convert_captured_ff_qdq"
    ).depends_on(capture_ff_stage)
    fx_to_onnx_program = onnx_pipeline.register_stage(
        stage_fx_to_onnx_program_qdq, "fx_to_onnx_program_qdq"
    ).depends_on(convert_captured_ff_qdq_stage)
    onnx_program_to_proto = onnx_pipeline.register_stage(
        stage_onnx_program_to_proto, "onnx_program_to_proto"
    ).depends_on(fx_to_onnx_program)
    onnx_pipeline.register_stage(
        stage_save_onnx_proto, "save_onnx_proto", capture_stage_output=True
    ).depends_on(onnx_program_to_proto)

    return onnx_pipeline
