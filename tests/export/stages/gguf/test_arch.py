# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Tests for the built-in GGUF architecture adapters.

The adapters (:data:`LLAMA_ADAPTER`, :data:`QWEN2_ADAPTER`, :data:`QWEN3_ADAPTER`)
are the pipeline's default reference implementations of the public
:class:`ArchAdapter` surface. These tests exercise the fields users are most
likely to depend on (name map, transforms).
"""

import pytest
import torch

from fastforward.exceptions import ExportError
from fastforward.export.stages.gguf import (
    GGUF_Q4_0,
    LLAMA_ADAPTER,
    QWEN2_ADAPTER,
    QWEN3_ADAPTER,
    llama_rope_permute,
)
from tests.export.stages.gguf.conftest import make_quantized_tensor


class _Config:
    def __init__(self, n_head: int, n_head_kv: int) -> None:
        self.num_attention_heads = n_head
        self.num_key_value_heads = n_head_kv
        # Additional fields kept minimal but present so the object satisfies
        # the GgufSourceConfig Protocol structurally.
        self.hidden_size = 0
        self.num_hidden_layers = 0
        self.intermediate_size = 0
        self.max_position_embeddings = 0
        self.rms_norm_eps = 1e-5
        self.vocab_size = 0


def test_llama_adapter_name_map_covers_attention_and_mlp() -> None:
    # GIVEN: the llama adapter's name map.
    name_map = LLAMA_ADAPTER.name_map

    # WHEN: mapping HuggingFace parameter paths.
    # THEN: static and layer-local names map to GGUF names.
    assert name_map("model.embed_tokens.weight") == "token_embd.weight"
    assert name_map("lm_head.weight") == "output.weight"
    assert name_map("model.layers.3.self_attn.q_proj.weight") == "blk.3.attn_q.weight"
    assert name_map("model.layers.0.mlp.down_proj.weight") == "blk.0.ffn_down.weight"

    # THEN: unsupported tensors (no per-head q/k norm in llama) return None.
    assert name_map("model.layers.0.self_attn.q_norm.weight") is None


def test_qwen3_adapter_name_map_includes_q_k_norm() -> None:
    # GIVEN: the qwen3 adapter's name map.
    name_map = QWEN3_ADAPTER.name_map

    # WHEN: mapping per-head query/key norm paths.
    # THEN: qwen3 additionally maps the per-head query/key norms.
    assert name_map("model.layers.2.self_attn.q_norm.weight") == "blk.2.attn_q_norm.weight"
    assert name_map("model.layers.2.self_attn.k_norm.weight") == "blk.2.attn_k_norm.weight"


def test_qwen2_adapter_name_map_includes_qkv_bias_not_q_k_norm() -> None:
    # GIVEN: the qwen2 adapter's name map.
    name_map = QWEN2_ADAPTER.name_map

    # WHEN: mapping bias and norm paths.
    # THEN: qwen2 maps the additive Q/K/V projection biases.
    assert name_map("model.layers.1.self_attn.q_proj.bias") == "blk.1.attn_q.bias"
    assert name_map("model.layers.1.self_attn.k_proj.bias") == "blk.1.attn_k.bias"
    assert name_map("model.layers.1.self_attn.v_proj.bias") == "blk.1.attn_v.bias"

    # THEN: qwen2 has no per-head query/key norm tensors (those are qwen3-only).
    assert name_map("model.layers.1.self_attn.q_norm.weight") is None


def test_llama_rope_permute_applies_to_q_and_k_only() -> None:
    # GIVEN: a config with 2 attention heads and 2 KV heads (produces non-trivial permute),
    # and q/k/v tensors with their original codes saved for comparison.
    config = _Config(n_head=2, n_head_kv=2)
    q_tensor = make_quantized_tensor("model.layers.0.self_attn.q_proj.weight", rows=8, cols=32)
    k_tensor = make_quantized_tensor("model.layers.0.self_attn.k_proj.weight", rows=8, cols=32)
    v_tensor = make_quantized_tensor("model.layers.0.self_attn.v_proj.weight", rows=8, cols=32)

    assert q_tensor.int_codes is not None
    assert v_tensor.int_codes is not None
    q_original = q_tensor.int_codes.clone()
    v_original = v_tensor.int_codes.clone()

    # WHEN: applying the llama rope transform to q_proj, k_proj, and v_proj tensors.
    q_result = llama_rope_permute(q_tensor, config, GGUF_Q4_0)
    k_result = llama_rope_permute(k_tensor, config, GGUF_Q4_0)
    v_result = llama_rope_permute(v_tensor, config, GGUF_Q4_0)

    # THEN: q/k are modified (rows reordered); v passes through unchanged.
    assert q_result.int_codes is not None
    assert not torch.equal(q_result.int_codes, q_original)
    assert k_result.hf_name == "model.layers.0.self_attn.k_proj.weight"
    torch.testing.assert_close(v_result.int_codes, v_original)


def test_llama_rope_permute_is_the_interleave_deinterleave() -> None:
    # GIVEN: a q_proj with 2 heads over 8 rows (4 rows per head).
    config = _Config(n_head=2, n_head_kv=2)
    block_size = GGUF_Q4_0.block_size
    tensor = make_quantized_tensor("model.layers.0.self_attn.q_proj.weight", rows=8, cols=32)
    assert tensor.int_codes is not None
    original_codes = tensor.int_codes.reshape(8, 1, block_size).clone()

    # WHEN: applying the transform.
    result = llama_rope_permute(tensor, config, GGUF_Q4_0)

    # THEN: each head's rows are de-interleaved into (even-half, odd-half) order,
    # matching llama.cpp's converter.
    assert result.int_codes is not None
    permuted = result.int_codes.reshape(8, 1, block_size)
    expected = original_codes[torch.tensor([0, 2, 1, 3, 4, 6, 5, 7])]
    torch.testing.assert_close(permuted, expected)


def test_llama_rope_permute_rejects_indivisible_row_count() -> None:
    # GIVEN: a row count not divisible by 2 * n_head.
    config = _Config(n_head=3, n_head_kv=3)
    tensor = make_quantized_tensor("model.layers.0.self_attn.q_proj.weight", rows=8, cols=32)

    # WHEN / THEN: the transform refuses rather than silently corrupting rows.
    with pytest.raises(ExportError, match="not divisible"):
        llama_rope_permute(tensor, config, GGUF_Q4_0)


def test_qwen_adapters_have_no_transforms() -> None:
    # GIVEN: the built-in qwen2 and qwen3 adapters.
    # THEN: neither carries any transforms (no RoPE permute needed).
    assert QWEN2_ADAPTER.transforms == []
    assert QWEN3_ADAPTER.transforms == []


def test_llama_adapter_has_rope_permute_transform() -> None:
    # GIVEN: the built-in llama adapter.
    # THEN: it has exactly one transform — the rope permute.
    assert len(LLAMA_ADAPTER.transforms) == 1
    assert LLAMA_ADAPTER.transforms[0] is llama_rope_permute


def test_adapters_carry_tokenizer_discriminators() -> None:
    # GIVEN: the built-in llama, qwen2, and qwen3 adapters.
    # THEN: each carries the llama.cpp tokenizer-pre string matching its
    # architecture's convention (llama-bpe for llama3-family, qwen2 for both qwens).
    assert LLAMA_ADAPTER.tokenizer_pre == "llama-bpe"
    assert QWEN2_ADAPTER.tokenizer_pre == "qwen2"
    assert QWEN3_ADAPTER.tokenizer_pre == "qwen2"
    for adapter in (LLAMA_ADAPTER, QWEN2_ADAPTER, QWEN3_ADAPTER):
        assert adapter.tokenizer_model == "gpt2"
