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


def _build_name_map(layer_patterns: dict[str, str]) -> _NameMapT:
    compiled = [(re.compile(pattern), template) for pattern, template in layer_patterns.items()]

    def name_map(hf_name: str) -> str | None:
        if hf_name in _STATIC_MAP:
            return _STATIC_MAP[hf_name]
        for pattern, template in compiled:
            match = pattern.match(hf_name)
            if match:
                return template.format(match.group(1))
        return None

    return name_map


llama_name_map: _NameMapT = _build_name_map(_COMMON_LAYER_PATTERNS)
qwen3_name_map: _NameMapT = _build_name_map({**_COMMON_LAYER_PATTERNS, **_QWEN3_EXTRA_PATTERNS})
qwen2_name_map: _NameMapT = _build_name_map({**_COMMON_LAYER_PATTERNS, **_QWEN2_EXTRA_PATTERNS})
