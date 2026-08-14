# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Pipeline stages for the FastForward -> GGUF (llama.cpp) export path.

The stages are intentionally thin: each reads its configuration from the
pipeline ``context`` and delegates the real work to the helper modules in this
package (extraction, RoPE permute, name mapping, block packing, vocab writing).
GGUF export is weight-only, so the extraction stage is a pipeline root that
operates on the live quantized module rather than a captured FX graph.
"""

import pathlib
import re

from dataclasses import replace
from typing import Any, TypeAlias, cast

import torch

from gguf import GGMLQuantizationType, GGUFWriter

from fastforward.exceptions import ExportError
from fastforward.export.stages.gguf._config import GgufSourceConfig
from fastforward.export.stages.gguf._extract import ExtractedTensor, extract_module_tensors
from fastforward.export.stages.gguf._vocab import write_vocab
from fastforward.export.stages.gguf.adapter import ArchAdapter, GgufQuantFormat

_SampleInputsT: TypeAlias = list[tuple[tuple[Any, ...], dict[str, Any]]]


def _require_quant_format(context: dict[str, Any]) -> GgufQuantFormat:
    quant_format = context.get("quant_format")
    if quant_format is None:
        msg = "GGUF export requires 'quant_format' in the pipeline options"
        raise ExportError(msg)
    if not isinstance(quant_format, GgufQuantFormat):
        msg = "'quant_format' must be a GgufQuantFormat instance"
        raise ExportError(msg)
    return quant_format


def _require_config(context: dict[str, Any]) -> GgufSourceConfig:
    model_config = context.get("model_config")
    if model_config is None:
        msg = "GGUF export requires 'model_config' in the pipeline options"
        raise ExportError(msg)
    return cast(GgufSourceConfig, model_config)


def _require_adapter(context: dict[str, Any]) -> ArchAdapter:
    adapter = context.get("arch_adapter")
    if adapter is None:
        msg = "GGUF export requires 'arch_adapter' in the pipeline options"
        raise ExportError(msg)
    return cast(ArchAdapter, adapter)


def stage_extract_quantized_weights(
    modules: tuple[torch.nn.Module, ...],
    sample_inputs: _SampleInputsT,
    context: dict[str, Any],
) -> list[ExtractedTensor]:
    """Extract integer codes + scales (and float passthroughs) from the FF module.

    Root stage: receives the source quantized module and reads every exportable
    tensor into an architecture-neutral list of :class:`ExtractedTensor`.
    """
    del sample_inputs
    (model,) = modules
    adapter = _require_adapter(context)
    config = _require_config(context)
    quant_format = _require_quant_format(context)

    return extract_module_tensors(
        model,
        adapter=adapter,
        config=config,
        quant_format=quant_format,
    )


def stage_apply_target_transforms(
    modules: tuple[list[ExtractedTensor], ...],
    sample_inputs: _SampleInputsT,
    context: dict[str, Any],
) -> list[ExtractedTensor]:
    """Apply architecture-specific tensor transforms in adapter-defined order.

    Iterates the adapter's ``transforms`` list, applying each to every tensor.
    Built-in transforms include :func:`llama_rope_permute`; users can append
    their own to the list when constructing a custom :class:`ArchAdapter`.
    """
    del sample_inputs
    (tensors,) = modules
    adapter = _require_adapter(context)
    config = _require_config(context)
    quant_format = _require_quant_format(context)

    transformed: list[ExtractedTensor] = []
    for tensor in tensors:
        for transform in adapter.transforms:
            tensor = transform(tensor, config, quant_format)
        transformed.append(tensor)
    return transformed


def stage_map_tensor_names(
    modules: tuple[list[ExtractedTensor], ...],
    sample_inputs: _SampleInputsT,
    context: dict[str, Any],
) -> list[ExtractedTensor]:
    """Assign each tensor its GGUF name via the architecture adapter's name map."""
    del sample_inputs
    (tensors,) = modules
    adapter = _require_adapter(context)

    named: list[ExtractedTensor] = []
    for tensor in tensors:
        gguf_name = adapter.name_map(tensor.hf_name)
        if gguf_name is None:
            continue
        named.append(replace(tensor, gguf_name=gguf_name))
    return named


def stage_pack_gguf_blocks(
    modules: tuple[list[ExtractedTensor], ...],
    sample_inputs: _SampleInputsT,
    context: dict[str, Any],
) -> dict[str, dict[str, torch.Tensor]]:
    """Pack quantized tensors into GGUF block bytes; keep float tensors as-is.

    Returns a mapping with two sub-dicts keyed by GGUF tensor name:
    ``{"quantized": {name: (rows, block_bytes) uint8 tensor}, "float": {name: tensor}}``.
    """
    del sample_inputs
    (tensors,) = modules
    quant_format = _require_quant_format(context)
    block_size = quant_format.block_size

    quantized: dict[str, torch.Tensor] = {}
    float_tensors: dict[str, torch.Tensor] = {}
    for tensor in tensors:
        if tensor.kind == "quantized":
            assert tensor.int_codes is not None and tensor.scales is not None
            int_codes = tensor.int_codes.reshape(-1, block_size)
            scales = tensor.scales.reshape(-1)
            packed = quant_format.pack_fn(int_codes, scales)
            quantized[tensor.gguf_name] = packed.reshape(tensor.rows, -1)
        else:
            assert tensor.float_data is not None
            float_tensors[tensor.gguf_name] = tensor.float_data

    return {"quantized": quantized, "float": float_tensors}


def stage_write_gguf(
    modules: tuple[dict[str, dict[str, torch.Tensor]], ...],
    sample_inputs: _SampleInputsT,
    context: dict[str, Any],
) -> pathlib.Path:
    """Write the packed tensors, metadata, and vocabulary to a ``.gguf`` file.

    Uses ``GGUFWriter``'s ``raw_dtype`` path so the pre-packed quantized bytes are
    written verbatim, preserving FastForward's learned scales. This is the
    torch->numpy boundary: ``GGUFWriter.add_tensor`` expects numpy arrays.
    """
    del sample_inputs
    (packed,) = modules

    adapter = _require_adapter(context)
    config = _require_config(context)
    quant_format = _require_quant_format(context)

    raw_dtype = GGMLQuantizationType[quant_format.name]

    output_dir = pathlib.Path(context["output_dir"])
    model_name = context["model_name"]
    output_path = output_dir / f"{model_name}.gguf"

    writer = GGUFWriter(str(output_path), arch=adapter.gguf_arch)
    try:
        adapter.write_metadata(writer, config)
        writer.add_file_type(quant_format.file_type)

        tokenizer = context.get("tokenizer")
        if tokenizer is not None:
            write_vocab(writer, tokenizer, config, adapter)

        for gguf_name, block_bytes in packed["quantized"].items():
            writer.add_tensor(gguf_name, block_bytes.numpy(), raw_dtype=raw_dtype)
        for gguf_name, float_data in packed["float"].items():
            float_type = _resolve_float_type(gguf_name, adapter)
            data = _cast_float(float_data, float_type)
            writer.add_tensor(gguf_name, data.numpy(), raw_dtype=GGMLQuantizationType[float_type])

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()

    return output_path


def _resolve_float_type(gguf_name: str, adapter: ArchAdapter) -> str:
    """Determine the GGML float type for a tensor, checking overrides first."""
    for pattern, float_type in adapter.float_type_overrides.items():
        if re.fullmatch(pattern, gguf_name):
            return float_type
    return adapter.float_type


_FLOAT_DTYPES: dict[str, torch.dtype] = {
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F64": torch.float64,
}

_RAW_BYTE_TYPES = frozenset({"BF16"})


def _cast_float(data: torch.Tensor, target_type: str) -> torch.Tensor:
    """Cast float32 tensor data to the target GGML float type for writing."""
    if target_type not in _FLOAT_DTYPES:
        msg = f"Unsupported float_type '{target_type}'. Supported: {sorted(_FLOAT_DTYPES)}"
        raise ExportError(msg)
    result = data.to(_FLOAT_DTYPES[target_type])
    if target_type in _RAW_BYTE_TYPES:
        result = result.view(torch.uint8)
    return result
