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

from collections import Counter
from dataclasses import replace
from typing import Any, TypeAlias, cast

import torch

from gguf import GGMLQuantizationType, GGUFWriter

from fastforward.exceptions import ExportError
from fastforward.export.stages.gguf._config import GgufSourceConfig
from fastforward.export.stages.gguf._extract import ExtractedTensor, extract_module_tensors
from fastforward.export.stages.gguf._fusion import apply_fusions
from fastforward.export.stages.gguf._packing import default_format_registry
from fastforward.export.stages.gguf._vocab import write_vocab
from fastforward.export.stages.gguf.adapter import ArchAdapter, GgufFormatRegistry, GgufQuantFormat

_SampleInputsT: TypeAlias = list[tuple[tuple[Any, ...], dict[str, Any]]]


def _require_format_registry(context: dict[str, Any]) -> GgufFormatRegistry:
    """Resolve a format registry from the pipeline context.

    Resolution order:
    1. ``context["format_registry"]`` — user-provided registry (may include
       custom formats).
    2. ``context["quant_format"]`` — single format (backward-compat). Creates a
       registry containing only that format, preserving strict single-format
       validation.
    3. Neither — returns the default registry with all built-in formats.
    """
    registry = context.get("format_registry")
    if registry is not None:
        if not isinstance(registry, GgufFormatRegistry):
            msg = "'format_registry' must be a GgufFormatRegistry instance"
            raise ExportError(msg)
        return registry

    quant_format = context.get("quant_format")
    if quant_format is not None:
        if not isinstance(quant_format, GgufQuantFormat):
            msg = "'quant_format' must be a GgufQuantFormat instance"
            raise ExportError(msg)
        single_registry = GgufFormatRegistry()
        single_registry.register(quant_format)
        return single_registry

    return default_format_registry()


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
    format_registry = _require_format_registry(context)

    return extract_module_tensors(
        model,
        adapter=adapter,
        config=config,
        format_registry=format_registry,
    )


def stage_fuse_tensors(
    modules: tuple[list[ExtractedTensor], ...],
    sample_inputs: _SampleInputsT,
    context: dict[str, Any],
) -> list[ExtractedTensor]:
    """Fuse groups of tensors into single outputs per the adapter's fusion specs.

    No-op when ``adapter.fusions`` is empty. Otherwise, for each
    :class:`TensorFusion` spec, groups matching tensors by their captured
    layer/instance indices and concatenates them along the specified axis.
    """
    del sample_inputs
    (tensors,) = modules
    adapter = _require_adapter(context)
    return apply_fusions(tensors, adapter.fusions)


def stage_apply_target_transforms(
    modules: tuple[list[ExtractedTensor], ...],
    sample_inputs: _SampleInputsT,
    context: dict[str, Any],
) -> list[ExtractedTensor]:
    """Apply architecture-specific tensor transforms in adapter-defined order.

    Iterates the adapter's ``transforms`` list, applying each to every tensor.
    Built-in transforms include :func:`llama_rope_permute`; users can append
    their own to the list when constructing a custom :class:`ArchAdapter`.

    Each transform receives the tensor's resolved :class:`GgufQuantFormat` (for
    quantized tensors) or a default format (for float tensors, which built-in
    transforms skip).
    """
    del sample_inputs
    (tensors,) = modules
    adapter = _require_adapter(context)
    config = _require_config(context)
    format_registry = _require_format_registry(context)
    fallback_format = format_registry.first()

    transformed: list[ExtractedTensor] = []
    for tensor in tensors:
        fmt = tensor.quant_format if tensor.quant_format is not None else fallback_format
        for transform in adapter.transforms:
            tensor = transform(tensor, config, fmt)
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
) -> dict[str, dict[str, Any]]:
    """Pack quantized tensors into GGUF block bytes; keep float tensors as-is.

    Each quantized tensor is packed using its own resolved :class:`GgufQuantFormat`,
    supporting mixed quantization types in a single file.

    Returns a mapping with two sub-dicts keyed by GGUF tensor name:
    ``{"quantized": {name: (block_bytes, format)}, "float": {name: tensor}}``.
    """
    del sample_inputs, context
    (tensors,) = modules

    quantized: dict[str, tuple[torch.Tensor, GgufQuantFormat]] = {}
    float_tensors: dict[str, torch.Tensor] = {}
    for tensor in tensors:
        if tensor.kind == "quantized":
            assert tensor.int_codes is not None and tensor.scales is not None
            assert tensor.quant_format is not None
            fmt = tensor.quant_format
            int_codes = tensor.int_codes.reshape(-1, fmt.block_size)
            scales = tensor.scales.reshape(-1)
            offsets = tensor.offsets.reshape(-1) if tensor.offsets is not None else None
            packed = fmt.pack_fn(int_codes, scales, offsets)
            if packed.shape[-1] != fmt.block_bytes:
                msg = (
                    f"pack_fn for {fmt.name} returned "
                    f"{packed.shape[-1]} bytes/block, expected {fmt.block_bytes}"
                )
                raise ExportError(msg)
            quantized[tensor.gguf_name] = (packed.reshape(tensor.rows, -1), fmt)
        else:
            assert tensor.float_data is not None
            float_tensors[tensor.gguf_name] = tensor.float_data

    return {"quantized": quantized, "float": float_tensors}


def stage_write_gguf(
    modules: tuple[dict[str, dict[str, Any]], ...],
    sample_inputs: _SampleInputsT,
    context: dict[str, Any],
) -> pathlib.Path:
    """Write the packed tensors, metadata, and vocabulary to a ``.gguf`` file.

    Uses ``GGUFWriter``'s ``raw_dtype`` path so the pre-packed quantized bytes are
    written verbatim, preserving FastForward's learned scales. Each quantized
    tensor is written with its own GGML type, supporting mixed quantization.

    The GGUF header ``file_type`` is set to the dominant quantization format
    (the format with the most tensors), matching llama.cpp's "MOSTLY_Q*"
    convention.
    """
    del sample_inputs
    (packed,) = modules

    adapter = _require_adapter(context)
    config = _require_config(context)

    output_dir = pathlib.Path(context["output_dir"])
    model_name = context["model_name"]
    output_path = output_dir / f"{model_name}.gguf"

    quantized_entries: dict[str, tuple[torch.Tensor, GgufQuantFormat]] = packed["quantized"]

    format_counts: Counter[int] = Counter()
    for _, fmt in quantized_entries.values():
        format_counts[fmt.file_type] += 1

    if not format_counts:
        msg = "GGUF export produced no quantized tensors — cannot determine file_type"
        raise ExportError(msg)

    dominant_file_type = format_counts.most_common(1)[0][0]

    writer = GGUFWriter(str(output_path), arch=adapter.gguf_arch)
    try:
        adapter.write_metadata(writer, config)
        writer.add_file_type(dominant_file_type)

        tokenizer = context.get("tokenizer")
        if tokenizer is not None:
            write_vocab(writer, tokenizer, config, adapter)

        for gguf_name, (block_bytes, fmt) in quantized_entries.items():
            raw_dtype = GGMLQuantizationType[fmt.name]
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
