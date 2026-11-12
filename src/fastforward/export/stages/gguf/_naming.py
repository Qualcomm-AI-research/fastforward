# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""HuggingFace -> GGUF tensor-name mapping for supported architectures.

The mapping is architecture-specific because different model families expose
different submodules (for example Qwen3 carries per-head ``q_norm``/``k_norm``
tensors that Llama does not). Each map returns the GGUF tensor name for a given
HuggingFace parameter name, or ``None`` when the parameter has no GGUF
counterpart and should be skipped.
"""

import re

from typing import Callable, TypeAlias

_NameMapT: TypeAlias = Callable[[str], str | None]

_STATIC_MAP = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
}

# Layer-local tensors shared by every currently-supported architecture.
_COMMON_LAYER_PATTERNS = {
    r"model\.layers\.(\d+)\.input_layernorm\.weight$": "blk.{}.attn_norm.weight",
    r"model\.layers\.(\d+)\.self_attn\.q_proj\.weight$": "blk.{}.attn_q.weight",
    r"model\.layers\.(\d+)\.self_attn\.k_proj\.weight$": "blk.{}.attn_k.weight",
    r"model\.layers\.(\d+)\.self_attn\.v_proj\.weight$": "blk.{}.attn_v.weight",
    r"model\.layers\.(\d+)\.self_attn\.o_proj\.weight$": "blk.{}.attn_output.weight",
    r"model\.layers\.(\d+)\.post_attention_layernorm\.weight$": "blk.{}.ffn_norm.weight",
    r"model\.layers\.(\d+)\.mlp\.gate_proj\.weight$": "blk.{}.ffn_gate.weight",
    r"model\.layers\.(\d+)\.mlp\.up_proj\.weight$": "blk.{}.ffn_up.weight",
    r"model\.layers\.(\d+)\.mlp\.down_proj\.weight$": "blk.{}.ffn_down.weight",
}

# Qwen3 additionally carries per-head query/key RMSNorm weights.
_QWEN3_EXTRA_PATTERNS = {
    r"model\.layers\.(\d+)\.self_attn\.q_norm\.weight$": "blk.{}.attn_q_norm.weight",
    r"model\.layers\.(\d+)\.self_attn\.k_norm\.weight$": "blk.{}.attn_k_norm.weight",
}

# Qwen2 (and Qwen2.5) instead carry additive biases on the Q/K/V projections;
# llama.cpp's ``qwen2`` architecture requires these bias tensors.
_QWEN2_EXTRA_PATTERNS = {
    r"model\.layers\.(\d+)\.self_attn\.q_proj\.bias$": "blk.{}.attn_q.bias",
    r"model\.layers\.(\d+)\.self_attn\.k_proj\.bias$": "blk.{}.attn_k.bias",
    r"model\.layers\.(\d+)\.self_attn\.v_proj\.bias$": "blk.{}.attn_v.bias",
}


# Gemma4 names its norms differently from Llama (``post_attention_layernorm`` is a
# post-norm, ``pre_feedforward_layernorm`` the FFN pre-norm), adds the per-layer
# input embedding path, and stores the per-layer output scale as a buffer.
_GEMMA4_STATIC_MAP = {
    **_STATIC_MAP,
    "model.embed_tokens_per_layer.weight": "per_layer_token_embd.weight",
    "model.per_layer_model_projection.weight": "per_layer_model_proj.weight",
    "model.per_layer_projection_norm.weight": "per_layer_proj_norm.weight",
}

_GEMMA4_LAYER_PATTERNS = {
    r"model\.layers\.(\d+)\.input_layernorm\.weight$": "blk.{}.attn_norm.weight",
    r"model\.layers\.(\d+)\.self_attn\.q_proj\.weight$": "blk.{}.attn_q.weight",
    r"model\.layers\.(\d+)\.self_attn\.k_proj\.weight$": "blk.{}.attn_k.weight",
    r"model\.layers\.(\d+)\.self_attn\.v_proj\.weight$": "blk.{}.attn_v.weight",
    r"model\.layers\.(\d+)\.self_attn\.o_proj\.weight$": "blk.{}.attn_output.weight",
    r"model\.layers\.(\d+)\.self_attn\.q_norm\.weight$": "blk.{}.attn_q_norm.weight",
    r"model\.layers\.(\d+)\.self_attn\.k_norm\.weight$": "blk.{}.attn_k_norm.weight",
    r"model\.layers\.(\d+)\.post_attention_layernorm\.weight$": "blk.{}.post_attention_norm.weight",
    r"model\.layers\.(\d+)\.pre_feedforward_layernorm\.weight$": "blk.{}.ffn_norm.weight",
    r"model\.layers\.(\d+)\.post_feedforward_layernorm\.weight$": "blk.{}.post_ffw_norm.weight",
    r"model\.layers\.(\d+)\.mlp\.gate_proj\.weight$": "blk.{}.ffn_gate.weight",
    r"model\.layers\.(\d+)\.mlp\.up_proj\.weight$": "blk.{}.ffn_up.weight",
    r"model\.layers\.(\d+)\.mlp\.down_proj\.weight$": "blk.{}.ffn_down.weight",
    r"model\.layers\.(\d+)\.per_layer_input_gate\.weight$": "blk.{}.inp_gate.weight",
    r"model\.layers\.(\d+)\.per_layer_projection\.weight$": "blk.{}.proj.weight",
    r"model\.layers\.(\d+)\.post_per_layer_input_norm\.weight$": "blk.{}.post_norm.weight",
    r"model\.layers\.(\d+)\.layer_scalar$": "blk.{}.layer_output_scale.weight",
}

# Multimodal ``*ForConditionalGeneration`` models (Gemma4, Qwen2.5-VL, Qwen3-VL)
# nest the text decoder under this prefix, next to the vision (and audio)
# towers; the text-only ``*ForCausalLM`` models use plain ``model.``.
_MULTIMODAL_TEXT_PREFIX = "model.language_model."


def _build_name_map(
    layer_patterns: dict[str, str], static_map: dict[str, str] = _STATIC_MAP
) -> _NameMapT:
    compiled = [(re.compile(pattern), template) for pattern, template in layer_patterns.items()]

    def name_map(hf_name: str) -> str | None:
        if hf_name in static_map:
            return static_map[hf_name]
        for pattern, template in compiled:
            match = pattern.match(hf_name)
            if match:
                return template.format(match.group(1))
        return None

    return name_map


def _with_multimodal_text_layout(name_map: _NameMapT) -> _NameMapT:
    """Extend a text-decoder name map to the multimodal checkpoint layout.

    The returned map accepts both the text-only (``model.layers...``) and the
    multimodal (``model.language_model.layers...``) names. Vision and audio
    tower tensors don't match the text patterns and map to ``None``, so they
    are not exported.
    """

    def multimodal_name_map(hf_name: str) -> str | None:
        if hf_name.startswith(_MULTIMODAL_TEXT_PREFIX):
            hf_name = "model." + hf_name.removeprefix(_MULTIMODAL_TEXT_PREFIX)
        return name_map(hf_name)

    return multimodal_name_map


llama_name_map: _NameMapT = _build_name_map(_COMMON_LAYER_PATTERNS)
qwen3_name_map: _NameMapT = _build_name_map({**_COMMON_LAYER_PATTERNS, **_QWEN3_EXTRA_PATTERNS})
qwen2_name_map: _NameMapT = _build_name_map({**_COMMON_LAYER_PATTERNS, **_QWEN2_EXTRA_PATTERNS})

# The Qwen2.5-VL and Qwen3-VL text decoders are Qwen2 and Qwen3 decoders.
qwen2vl_name_map: _NameMapT = _with_multimodal_text_layout(qwen2_name_map)
qwen3vl_name_map: _NameMapT = _with_multimodal_text_layout(qwen3_name_map)

gemma4_name_map: _NameMapT = _with_multimodal_text_layout(
    _build_name_map(_GEMMA4_LAYER_PATTERNS, _GEMMA4_STATIC_MAP)
)
