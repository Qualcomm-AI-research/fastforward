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
  query/key norm and uses the ``llama-bpe`` pre-tokenizer. Llama 3.1+ RoPE
  scaling is exported as per-frequency factors in ``rope_freqs.weight``.
- **Qwen3** does *not* permute Q/K, carries per-head ``q_norm``/``k_norm``
  RMSNorm tensors, and (typically) tied input/output embeddings.
- **Qwen2 / Qwen2.5** does not permute Q/K either, but carries additive biases
  on the Q/K/V projections.
- **Gemma4** (dense text decoder) mixes sliding-window and full-attention
  layers with different head dims, shares KV across its last layers, feeds
  per-layer input embeddings, and needs a synthesized ``rope_freqs`` tensor.
  It uses a byte-fallback BPE vocabulary with no pre-tokenizer.
"""

from __future__ import annotations

import math
import re

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import torch

from gguf import GGUFWriter, RopeScalingType

from fastforward.exceptions import ExportError
from fastforward.export.stages.gguf._config import GgufSourceConfig
from fastforward.export.stages.gguf._naming import (
    gemma4_name_map,
    llama_name_map,
    qwen2_name_map,
    qwen3_name_map,
)
from fastforward.export.stages.gguf._vocab import make_llama_hf_vocab_writer
from fastforward.export.stages.gguf.adapter import ArchAdapter
from fastforward.export.stages.gguf.rules import DROP_TIED_LM_HEAD, Rename, Synthesize, Transform

if TYPE_CHECKING:
    from fastforward.export.stages.gguf._extract import ExtractedTensor

_Q_PROJ_RE = re.compile(r"model\.layers\.\d+\.self_attn\.q_proj\.weight$")
_K_PROJ_RE = re.compile(r"model\.layers\.\d+\.self_attn\.k_proj\.weight$")
_QK_PROJ_PATTERN = r"model\.layers\.\d+\.self_attn\.[qk]_proj\.weight"


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
    """Reorder a quantized tensor's rows, keeping code/scale/offset pairing and shapes intact."""
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
    permuted_codes = codes[row_index].reshape(tensor.int_codes.shape)
    permuted_scales = scales[row_index].reshape(tensor.scales.shape)
    permuted_offsets = None
    if tensor.offsets is not None:
        offsets = tensor.offsets.reshape(tensor.rows, blocks_per_row)
        permuted_offsets = offsets[row_index].reshape(tensor.offsets.shape)
    return replace(
        tensor, int_codes=permuted_codes, scales=permuted_scales, offsets=permuted_offsets
    )


def llama_rope_permute(tensor: ExtractedTensor, config: GgufSourceConfig) -> ExtractedTensor:
    """RoPE de-interleave transform for llama.cpp's Llama loader.

    Permutes rows of Q projections by ``num_attention_heads`` and K projections
    by ``num_key_value_heads``, both quantized and float. Non-matching tensors
    are returned unchanged.

    This is a :data:`~fastforward.export.stages.gguf.rules.TensorTransformT`
    for use with a :class:`~fastforward.export.stages.gguf.rules.Transform`
    rule.
    """
    if _Q_PROJ_RE.match(tensor.hf_name):
        row_index = _rope_permute_row_index(tensor.rows, config.num_attention_heads)
    elif _K_PROJ_RE.match(tensor.hf_name):
        n_head_kv = getattr(config, "num_key_value_heads", config.num_attention_heads)
        row_index = _rope_permute_row_index(tensor.rows, n_head_kv)
    else:
        return tensor

    if tensor.kind != "quantized":
        assert tensor.float_data is not None
        return replace(tensor, float_data=tensor.float_data[row_index])
    assert tensor.quant_format is not None
    return _apply_row_permute(tensor, row_index, tensor.quant_format.block_size)


def _rope_theta(config: GgufSourceConfig) -> float:
    """Return the RoPE base frequency.

    transformers>=5 moves ``rope_theta`` into ``config.rope_parameters``; older
    configs (and plain namespaces) carry it as an attribute.

    Raises:
        ExportError: If the config carries neither.
    """
    rope_parameters = getattr(config, "rope_parameters", None) or {}
    rope_theta = rope_parameters.get("rope_theta", getattr(config, "rope_theta", None))
    if rope_theta is None:
        msg = "model_config has neither rope_parameters['rope_theta'] nor rope_theta"
        raise ExportError(msg)
    return float(rope_theta)


def _rope_scaling(config: GgufSourceConfig) -> dict[str, Any]:
    """Return the config's RoPE scaling parameters (empty for unscaled legacy configs).

    transformers 5 aliases ``rope_scaling`` to ``rope_parameters``, which is always
    set and has ``rope_type="default"`` for unscaled RoPE; older configs set
    ``rope_scaling`` only when scaled.
    """
    rope_parameters = getattr(config, "rope_parameters", None) or {}
    return rope_parameters or getattr(config, "rope_scaling", None) or {}


def _rope_scaling_type(rope_scaling: dict[str, Any]) -> str:
    return str(rope_scaling.get("type", rope_scaling.get("rope_type", "default")))


def llama3_rope_freqs(config: GgufSourceConfig) -> dict[str, torch.Tensor]:
    """Synthesize llama.cpp's ``rope_freqs.weight`` for Llama 3 RoPE scaling.

    Llama 3.1+ (``rope_type="llama3"``) scales each RoPE frequency by a
    wavelength-dependent factor: short wavelengths are kept, long ones divided
    by ``factor``, and the band in between interpolated. llama.cpp has no
    metadata for this and reads the per-frequency divisors from
    ``rope_freqs.weight`` instead. Mirrors llama.cpp's
    ``LlamaModel.generate_extra_tensors``.

    Returns an empty mapping for configs without Llama 3 RoPE scaling, so
    ``Synthesize(llama3_rope_freqs)`` is safe for every Llama config.

    Note:
        llama.cpp only loads ``rope_freqs.weight`` as F32. Exports with a lower
        ``float_type`` should pin it to F32 via a
        ``float_type_overrides`` entry matching ``rope_freqs.weight``.
    """
    rope_scaling = _rope_scaling(config)
    if _rope_scaling_type(rope_scaling) != "llama3":
        return {}

    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    exponents = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    inv_freq = 1.0 / _rope_theta(config) ** exponents

    factor = rope_scaling.get("factor", 8.0)
    low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
    high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
    old_context_len = rope_scaling.get("original_max_position_embeddings", 8192)

    wavelen = 2 * math.pi / inv_freq
    smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    interpolated = 1 / ((1 - smooth) / factor + smooth)
    freq_factors = torch.where(
        wavelen < old_context_len / high_freq_factor,
        1.0,
        torch.where(wavelen > old_context_len / low_freq_factor, factor, interpolated),
    )
    return {"rope_freqs.weight": freq_factors.float()}


# llama.cpp divides each RoPE frequency by its factor; a huge factor drives the
# rotation angle to ~0, leaving that dimension pair unrotated.
_UNROTATED_FREQ_FACTOR = 1e30


def gemma4_rope_freqs(config: GgufSourceConfig) -> dict[str, torch.Tensor]:
    """Synthesize llama.cpp's ``rope_freqs.weight`` for Gemma4's full-attention layers.

    Gemma4's full-attention layers use "proportional" RoPE with a
    ``partial_rotary_factor``: only the leading ``n_rot`` frequency pairs of
    each NEOX half rotate (``cc000000ss000000``). ggml's NEOX RoPE can only
    rotate a contiguous prefix (``ccss000000000000``), and the head dims cannot
    be reordered without breaking Gemma4's KV sharing. llama.cpp instead
    rotates the full head and receives per-frequency divisors: the RoPE
    scaling ``factor`` (``1.0`` when absent) for rotated pairs,
    :data:`_UNROTATED_FREQ_FACTOR` for the rest.

    The tensor is derived from config alone (``global_head_dim`` and
    ``rope_parameters["full_attention"]``) and mirrors llama.cpp's
    ``Gemma4Model.generate_extra_tensors``. Use it as
    ``Synthesize(gemma4_rope_freqs)`` in an adapter's ``tensor_rules``.

    Note:
        llama.cpp only loads ``rope_freqs.weight`` as F32, and the unrotated
        factors overflow to infinity in F16, so :data:`GEMMA4_ADAPTER` pins it
        to F32 whatever the export's ``float_type``. A matching
        ``float_type_overrides`` entry still takes precedence.

    Raises:
        ExportError: If the config has no proportional full-attention RoPE
            parameters, no ``global_head_dim``, or a non-positive ``factor``.
    """
    rope_parameters = getattr(config, "rope_parameters", None) or {}
    full_attention = rope_parameters.get("full_attention")
    if full_attention is None or full_attention.get("rope_type") != "proportional":
        msg = (
            "Gemma4 rope_freqs requires rope_parameters['full_attention'] with "
            + f"rope_type='proportional', got {full_attention!r}"
        )
        raise ExportError(msg)
    head_dim = getattr(config, "global_head_dim", None)
    if head_dim is None:
        msg = "Gemma4 rope_freqs requires 'global_head_dim' on the model config"
        raise ExportError(msg)

    # Transformers divides the proportional inverse frequencies by ``factor``,
    # which matches llama.cpp's divisor semantics.
    factor = full_attention.get("factor", 1.0)
    if not isinstance(factor, int | float) or isinstance(factor, bool) or factor <= 0:
        msg = f"Gemma4 rope_freqs requires a positive RoPE scaling 'factor', got {factor!r}"
        raise ExportError(msg)
    partial_rotary_factor = full_attention.get("partial_rotary_factor", 1.0)
    n_freqs = head_dim // 2
    n_rot = int(head_dim * partial_rotary_factor / 2)
    freq_factors = torch.full((n_freqs,), _UNROTATED_FREQ_FACTOR, dtype=torch.float32)
    freq_factors[:n_rot] = float(factor)
    return {"rope_freqs.weight": freq_factors}


def _write_common_metadata(writer: GGUFWriter, config: GgufSourceConfig) -> None:
    """Write metadata shared by all supported Llama-family architectures."""
    writer.add_context_length(config.max_position_embeddings)
    writer.add_embedding_length(config.hidden_size)
    writer.add_block_count(config.num_hidden_layers)
    writer.add_feed_forward_length(config.intermediate_size)
    writer.add_head_count(config.num_attention_heads)
    writer.add_head_count_kv(getattr(config, "num_key_value_heads", config.num_attention_heads))
    writer.add_rope_freq_base(_rope_theta(config))
    writer.add_layer_norm_rms_eps(config.rms_norm_eps)
    writer.add_vocab_size(config.vocab_size)

    rope_scaling = _rope_scaling(config)
    scaling_type = _rope_scaling_type(rope_scaling)
    type_map = {member.value: member for member in RopeScalingType}
    if scaling_type in ("default", "llama3"):
        # Llama 3 scaling lives in rope_freqs.weight (see `llama3_rope_freqs`);
        # a bare factor would make llama.cpp apply linear scaling instead.
        writer.add_rope_scaling_type(RopeScalingType.NONE)
    elif scaling_type in type_map:
        writer.add_rope_scaling_type(type_map[scaling_type])
        if "factor" in rope_scaling:
            writer.add_rope_scaling_factor(rope_scaling["factor"])
    else:
        msg = f"GGUF export does not support RoPE scaling type '{scaling_type}'"
        raise ExportError(msg)


def _write_qwen3_metadata(writer: GGUFWriter, config: GgufSourceConfig) -> None:
    """Write Qwen3 metadata (common set plus attention head dimension)."""
    _write_common_metadata(writer, config)
    head_dim = getattr(config, "head_dim", None)
    if head_dim is not None:
        writer.add_key_length(head_dim)
        writer.add_value_length(head_dim)


def _require_config(config: GgufSourceConfig, name: str) -> Any:
    value = getattr(config, name, None)
    if value is None:
        msg = f"Gemma4 GGUF export requires '{name}' on the model config"
        raise ExportError(msg)
    return value


def _write_gemma4_metadata(writer: GGUFWriter, config: GgufSourceConfig) -> None:
    """Write Gemma4 text-decoder metadata, mirroring llama.cpp's ``Gemma4Model`` converter.

    Expects the text config (``Gemma4TextConfig``, i.e. ``config.text_config``
    of a multimodal checkpoint). Full-attention layers use ``global_head_dim``
    and ``rope_parameters["full_attention"]``; sliding-window layers use
    ``head_dim`` and ``rope_parameters["sliding_attention"]``.

    Raises:
        ExportError: If a required Gemma4 config field is missing, the config
            enables the MoE block (not supported), or its KV-shared layers
            follow a layout llama.cpp cannot load.
    """
    if getattr(config, "enable_moe_block", False):
        msg = "Gemma4 GGUF export does not support MoE variants (enable_moe_block=True)"
        raise ExportError(msg)
    layer_types = _require_config(config, "layer_types")
    rope_parameters = _require_config(config, "rope_parameters")
    num_kv_shared_layers = _require_config(config, "num_kv_shared_layers")
    head_dim_full = _require_config(config, "global_head_dim")
    head_dim_swa = _require_config(config, "head_dim")
    n_layers = config.num_hidden_layers
    swa_layers = [layer_type == "sliding_attention" for layer_type in layer_types]

    # llama.cpp points KV-shared sliding-window layers at the second-to-last
    # non-shared layer and full-attention ones at the last, so those two layers
    # must be one sliding-window and one full-attention layer, in that order.
    first_shared_layer = n_layers - num_kv_shared_layers
    if num_kv_shared_layers and (
        first_shared_layer < 2
        or swa_layers[first_shared_layer - 2 : first_shared_layer] != [True, False]
    ):
        msg = (
            "llama.cpp requires the last two non-KV-shared Gemma4 layers to be a "
            + "sliding-window layer followed by a full-attention layer"
        )
        raise ExportError(msg)

    writer.add_block_count(n_layers)
    writer.add_context_length(config.max_position_embeddings)
    writer.add_embedding_length(config.hidden_size)
    writer.add_head_count(config.num_attention_heads)
    writer.add_layer_norm_rms_eps(config.rms_norm_eps)

    # Double-wide MLPs only apply to the KV-shared layers.
    n_ff = config.intermediate_size
    if getattr(config, "use_double_wide_mlp", False):
        writer.add_feed_forward_length([
            n_ff if layer < first_shared_layer else 2 * n_ff for layer in range(n_layers)
        ])
    else:
        writer.add_feed_forward_length(n_ff)

    n_head_kv_swa = getattr(config, "num_key_value_heads", config.num_attention_heads)
    n_head_kv_full = getattr(config, "num_global_key_value_heads", None)
    if n_head_kv_full is None:
        writer.add_head_count_kv(n_head_kv_swa)
    else:
        writer.add_head_count_kv([n_head_kv_swa if swa else n_head_kv_full for swa in swa_layers])

    writer.add_key_length(head_dim_full)
    writer.add_value_length(head_dim_full)
    writer.add_key_length_swa(head_dim_swa)
    writer.add_value_length_swa(head_dim_swa)
    writer.add_sliding_window(_require_config(config, "sliding_window"))
    writer.add_sliding_window_pattern(swa_layers)
    writer.add_shared_kv_layers(num_kv_shared_layers)
    writer.add_embedding_length_per_layer_input(
        getattr(config, "hidden_size_per_layer_input", None) or 0
    )
    if final_logit_softcap := getattr(config, "final_logit_softcapping", None):
        writer.add_final_logit_softcapping(final_logit_softcap)

    # Full-attention layers rotate the whole head; their partial rotation is
    # encoded in the synthesized rope_freqs tensor (see gemma4_rope_freqs).
    rope_swa = rope_parameters.get("sliding_attention", {})
    rope_theta = (rope_parameters.get("full_attention") or {}).get("rope_theta")
    if rope_theta is None:
        msg = "Gemma4 GGUF export requires rope_parameters['full_attention']['rope_theta']"
        raise ExportError(msg)
    writer.add_rope_freq_base(rope_theta)
    if (rope_theta_swa := rope_swa.get("rope_theta")) is not None:
        writer.add_rope_freq_base_swa(rope_theta_swa)
    writer.add_rope_dimension_count(head_dim_full)
    writer.add_rope_dimension_count_swa(
        int(head_dim_swa * rope_swa.get("partial_rotary_factor", 1.0))
    )


# Chat-template markers llama.cpp's chat parser must see in the output, so they
# are typed USER_DEFINED (always rendered) rather than CONTROL.
_GEMMA4_VISIBLE_TOKENS = frozenset({
    "<|channel>",
    "<channel|>",
    "<|tool_call>",
    "<tool_call|>",
    "<|tool_response>",
    "<tool_response|>",
    '<|"|>',
})


LLAMA_ADAPTER = ArchAdapter(
    gguf_arch="llama",
    tensor_rules=(
        DROP_TIED_LM_HEAD,
        Transform(_QK_PROJ_PATTERN, llama_rope_permute),
        Rename(llama_name_map),
        Synthesize(llama3_rope_freqs),
    ),
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
    tensor_rules=(DROP_TIED_LM_HEAD, Rename(qwen2_name_map)),
    write_metadata=_write_common_metadata,
    tokenizer_model="gpt2",
    tokenizer_pre="qwen2",
)
"""Built-in adapter for Qwen2 / Qwen2.5 models."""

QWEN3_ADAPTER = ArchAdapter(
    gguf_arch="qwen3",
    tensor_rules=(DROP_TIED_LM_HEAD, Rename(qwen3_name_map)),
    write_metadata=_write_qwen3_metadata,
    tokenizer_model="gpt2",
    tokenizer_pre="qwen2",
)
"""Built-in adapter for Qwen3 models."""


def _pin_gemma4_float_type(gguf_name: str, ndim: int, float_type: str) -> str | None:
    """Pin the tensors llama.cpp only loads as F32 (``rope_freqs.weight``)."""
    del ndim, float_type
    return "F32" if gguf_name == "rope_freqs.weight" else None


GEMMA4_ADAPTER = ArchAdapter(
    gguf_arch="gemma4",
    tensor_rules=(
        DROP_TIED_LM_HEAD,
        Rename(gemma4_name_map),
        Synthesize(gemma4_rope_freqs),
    ),
    write_metadata=_write_gemma4_metadata,
    tokenizer_model="gemma4",
    tokenizer_pre=None,
    write_vocab=make_llama_hf_vocab_writer(
        user_defined_tokens=_GEMMA4_VISIBLE_TOKENS, add_bos_token=True, add_space_prefix=False
    ),
    pin_float_type=_pin_gemma4_float_type,
)
"""Built-in adapter for dense Gemma4 text decoders (e.g. ``gemma-4-E4B``).

Works with both ``Gemma4ForCausalLM`` and ``Gemma4ForConditionalGeneration``;
the vision and audio towers of the latter are not exported. Pass the text
config (``config.text_config`` for multimodal checkpoints) as ``model_config``.
"""
