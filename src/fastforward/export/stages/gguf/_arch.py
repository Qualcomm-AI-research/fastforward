# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Built-in :class:`ArchAdapter` instances for the model families FastForward ships out of the box.

Only these constants — and the private helpers backing them — should know
about HuggingFace parameter paths, llama.cpp's RoPE row-interleave, or the
various llama.cpp tokenizer-pre discriminators. Users targeting an unsupported
architecture should construct their own :class:`ArchAdapter` from
:mod:`fastforward.export.stages.gguf.adapter` rather than editing this file.

- **Llama (llama3-family)** stores attention Q/K projections in an interleaved
  RoPE layout that llama.cpp undoes with a row permutation. It has no per-head
  query/key norm and uses the ``llama-bpe`` pre-tokenizer.
- **Qwen3** does *not* permute Q/K, carries per-head ``q_norm``/``k_norm``
  RMSNorm tensors, and (typically) tied input/output embeddings.
- **Qwen2 / Qwen2.5** does not permute Q/K either, but carries additive biases
  on the Q/K/V projections.
"""

from __future__ import annotations

import logging
import re

from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from gguf import GGUFWriter, RopeScalingType

from fastforward.exceptions import ExportError
from fastforward.export.stages.gguf._config import GgufSourceConfig
from fastforward.export.stages.gguf._naming import (
    llama_name_map,
    qwen2_name_map,
    qwen3_name_map,
)
from fastforward.export.stages.gguf.adapter import ArchAdapter

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from fastforward.export.stages.gguf._extract import ExtractedTensor
    from fastforward.export.stages.gguf.adapter import GgufQuantFormat

_Q_PROJ_RE = re.compile(r"model\.layers\.\d+\.self_attn\.q_proj\.weight$")
_K_PROJ_RE = re.compile(r"model\.layers\.\d+\.self_attn\.k_proj\.weight$")


def _rope_permute_row_index(n_rows: int, n_head: int) -> torch.Tensor:
    """Return the row permutation llama.cpp applies to undo interleaved RoPE.

    The transform is a pure reordering of weight rows, so it can be expressed as
    an index tensor and applied identically to integer codes and their per-row
    scales — preserving FastForward's learned scale/code pairing.
    """
    if n_rows % (n_head * 2) != 0:
        msg = (
            f"Cannot apply RoPE permute: row count {n_rows} is not divisible by "
            f"2 * n_head ({2 * n_head})"
        )
        raise ExportError(msg)
    index = torch.arange(n_rows).reshape(n_head, 2, n_rows // n_head // 2)
    return index.swapaxes(1, 2).reshape(n_rows).contiguous()


def _apply_row_permute(
    tensor: ExtractedTensor, row_index: torch.Tensor, block_size: int
) -> ExtractedTensor:
    """Reorder a quantized tensor's rows, keeping code/scale pairing intact."""
    assert tensor.int_codes is not None and tensor.scales is not None
    assert tensor.cols is not None
    if tensor.cols % block_size != 0:
        msg = (
            f"Cannot apply row permute to '{tensor.hf_name}': column count "
            f"{tensor.cols} is not divisible by block_size {block_size}"
        )
        raise ExportError(msg)
    blocks_per_row = tensor.cols // block_size

    codes = tensor.int_codes.reshape(tensor.rows, blocks_per_row, block_size)
    scales = tensor.scales.reshape(tensor.rows, blocks_per_row)
    permuted_codes = codes[row_index].reshape(-1, block_size)
    permuted_scales = scales[row_index].reshape(-1)
    return replace(tensor, int_codes=permuted_codes, scales=permuted_scales)


def llama_rope_permute(
    tensor: ExtractedTensor, config: GgufSourceConfig, quant_format: GgufQuantFormat
) -> ExtractedTensor:
    """RoPE de-interleave transform for llama.cpp's Llama loader.

    Permutes rows of Q projections by ``num_attention_heads`` and K projections
    by ``num_key_value_heads``. Non-matching tensors and float tensors are
    returned unchanged.

    This is a :data:`~fastforward.export.stages.gguf.adapter.TensorTransformT`
    and can be used in an adapter's ``transforms`` list.
    """
    if tensor.kind != "quantized":
        return tensor

    if _Q_PROJ_RE.match(tensor.hf_name):
        row_index = _rope_permute_row_index(tensor.rows, config.num_attention_heads)
    elif _K_PROJ_RE.match(tensor.hf_name):
        n_head_kv = getattr(config, "num_key_value_heads", config.num_attention_heads)
        row_index = _rope_permute_row_index(tensor.rows, n_head_kv)
    else:
        return tensor

    return _apply_row_permute(tensor, row_index, quant_format.block_size)


def _write_common_metadata(writer: GGUFWriter, config: GgufSourceConfig) -> None:
    """Write metadata shared by all supported Llama-family architectures."""
    writer.add_context_length(config.max_position_embeddings)
    writer.add_embedding_length(config.hidden_size)
    writer.add_block_count(config.num_hidden_layers)
    writer.add_feed_forward_length(config.intermediate_size)
    writer.add_head_count(config.num_attention_heads)
    writer.add_head_count_kv(getattr(config, "num_key_value_heads", config.num_attention_heads))
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is None:
        rope_theta = 1000000.0
        logger.warning("model_config has no 'rope_theta'; defaulting to %.1f", rope_theta)
    writer.add_rope_freq_base(rope_theta)
    writer.add_layer_norm_rms_eps(config.rms_norm_eps)
    writer.add_vocab_size(config.vocab_size)

    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling:
        scaling_type = rope_scaling.get("type", rope_scaling.get("rope_type", ""))
        type_map = {member.value: member for member in RopeScalingType}
        if scaling_type in type_map:
            writer.add_rope_scaling_type(type_map[scaling_type])
        if "factor" in rope_scaling:
            writer.add_rope_scaling_factor(rope_scaling["factor"])
    else:
        writer.add_rope_scaling_type(RopeScalingType.NONE)


def _write_qwen3_metadata(writer: GGUFWriter, config: GgufSourceConfig) -> None:
    """Write Qwen3 metadata (common set plus attention head dimension)."""
    _write_common_metadata(writer, config)
    head_dim = getattr(config, "head_dim", None)
    if head_dim is not None:
        writer.add_key_length(head_dim)
        writer.add_value_length(head_dim)


LLAMA_ADAPTER = ArchAdapter(
    gguf_arch="llama",
    name_map=llama_name_map,
    transforms=[llama_rope_permute],
    write_metadata=_write_common_metadata,
    tokenizer_model="gpt2",
    tokenizer_pre="llama-bpe",
)
"""Built-in adapter for llama3-family models.

Uses llama.cpp's ``llama-bpe`` pre-tokenizer. For llama2, construct a variant
with ``tokenizer_pre="default"``.
"""

QWEN2_ADAPTER = ArchAdapter(
    gguf_arch="qwen2",
    name_map=qwen2_name_map,
    transforms=[],
    write_metadata=_write_common_metadata,
    tokenizer_model="gpt2",
    tokenizer_pre="qwen2",
)
"""Built-in adapter for Qwen2 / Qwen2.5 models."""

QWEN3_ADAPTER = ArchAdapter(
    gguf_arch="qwen3",
    name_map=qwen3_name_map,
    transforms=[],
    write_metadata=_write_qwen3_metadata,
    tokenizer_model="gpt2",
    tokenizer_pre="qwen2",
)
"""Built-in adapter for Qwen3 models."""
