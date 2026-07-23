# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Any

from fastforward.export.pipeline.core import Pipeline
from fastforward.export.stages.gguf.gguf_export_stages import (
    stage_apply_target_transforms,
    stage_extract_quantized_weights,
    stage_map_tensor_names,
    stage_pack_gguf_blocks,
    stage_write_gguf,
)


def gguf_llama_cpp_pipeline(pipeline_kwargs: dict[str, Any]) -> Pipeline:
    r"""Create the FastForward -> GGUF -> llama.cpp export pipeline.

    This pipeline takes an already-quantized FastForward ``QuantizedModule`` and
    writes a GGUF file whose quantized tensors preserve FastForward's learned
    scales (via ``GGUFWriter``'s ``raw_dtype`` path). It does **not** run
    calibration or GPTQ — quantization is the user's separate upstream step.
    Parameters without an initialized FastForward quantizer (embeddings, norms,
    biases) pass through as F32.

    GGUF export is weight-only, so the extraction stage is a pipeline root that
    reads the live quantized module directly rather than reusing the FX
    capture/convert/cleanup chain of the ONNX pipeline.

    High-level flow:

    ```text
    ff_model (pipeline input)
        |
        | +--> extract_quantized_weights   (root; int codes + scales + float passthroughs)
                    |
                    v
                apply_target_transforms     (per-arch RoPE Q/K permute)
                    |
                    v
                map_tensor_names            (HF -> GGUF tensor names)
                    |
                    v
                pack_gguf_blocks            (Q4_0/Q8_0 convention convert + nibble pack)
                    |
                    v
                write_gguf   [capture]      (GGUFWriter raw_dtype + metadata + vocab)
    ```

    Args:
        pipeline_kwargs: Pipeline context/configuration consumed by the stages
            (``arch_adapter``, ``quant_format``, ``model_config``, ``tokenizer``,
            plus ``output_dir``/``model_name`` from the orchestrator). Users
            select the target architecture by passing an ``ArchAdapter`` —
            :data:`fastforward.export.stages.gguf.LLAMA_ADAPTER`,
            ``QWEN2_ADAPTER``, or ``QWEN3_ADAPTER`` — or by constructing their
            own for a non-standard model layout.

    Returns:
        A configured `Pipeline` that writes a `<model_name>.gguf` artifact.
    """
    pipeline = Pipeline(pipeline_kwargs)

    extract_stage = pipeline.register_stage(
        stage_extract_quantized_weights, "extract_quantized_weights"
    )
    apply_transforms_stage = pipeline.register_stage(
        stage_apply_target_transforms, "apply_target_transforms"
    ).depends_on(extract_stage)
    map_names_stage = pipeline.register_stage(
        stage_map_tensor_names, "map_tensor_names"
    ).depends_on(apply_transforms_stage)
    pack_stage = pipeline.register_stage(stage_pack_gguf_blocks, "pack_gguf_blocks").depends_on(
        map_names_stage
    )
    pipeline.register_stage(stage_write_gguf, "write_gguf", capture_stage_output=True).depends_on(
        pack_stage
    )

    return pipeline
