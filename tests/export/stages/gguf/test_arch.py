# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Tests for the built-in GGUF architecture adapters.

The adapters (:data:`LLAMA_ADAPTER`, :data:`QWEN2_ADAPTER`, :data:`QWEN3_ADAPTER`)
are the pipeline's default reference implementations of the public
:class:`ArchAdapter` surface. These tests exercise the fields users are most
likely to depend on (tensor rules, transforms).
"""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
import torch

from fastforward.exceptions import ExportError
from fastforward.export.stages.gguf import (
    GGUF_Q4_0,
    GGUF_Q4_1,
    LLAMA_ADAPTER,
    QWEN2_ADAPTER,
    QWEN3_ADAPTER,
    ArchAdapter,
    ExtractedTensor,
    Transform,
    llama_rope_permute,
)
from fastforward.export.stages.gguf.rules import apply_tensor_rules
from gguf import GGUFWriter, RopeScalingType
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


def _gguf_name(adapter: ArchAdapter, hf_name: str) -> str | None:
    """Run the adapter's tensor rules on one float tensor and return its GGUF name."""
    tensor = ExtractedTensor(hf_name=hf_name, kind="float", rows=2, float_data=torch.zeros(2))
    result = apply_tensor_rules([tensor], adapter.tensor_rules, _Config(n_head=1, n_head_kv=1))
    return result[0].gguf_name if result else None


def _metadata_config(**rope_fields: Any) -> SimpleNamespace:
    """A config with every field the metadata writers read, plus `rope_fields`."""
    return SimpleNamespace(
        max_position_embeddings=128,
        hidden_size=32,
        num_hidden_layers=2,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=64,
        **rope_fields,
    )


def test_llama_adapter_name_map_covers_attention_and_mlp() -> None:
    # GIVEN: the llama adapter's tensor rules.
    def name_map(hf_name: str) -> str | None:
        return _gguf_name(LLAMA_ADAPTER, hf_name)

    # WHEN: mapping HuggingFace parameter paths.
    # THEN: static and layer-local names map to GGUF names.
    assert name_map("model.embed_tokens.weight") == "token_embd.weight"
    assert name_map("lm_head.weight") == "output.weight"
    assert name_map("model.layers.3.self_attn.q_proj.weight") == "blk.3.attn_q.weight"
    assert name_map("model.layers.0.mlp.down_proj.weight") == "blk.0.ffn_down.weight"

    # THEN: unsupported tensors (no per-head q/k norm in llama) return None.
    assert name_map("model.layers.0.self_attn.q_norm.weight") is None


def test_qwen3_adapter_name_map_includes_q_k_norm() -> None:
    # GIVEN: the qwen3 adapter's tensor rules.
    def name_map(hf_name: str) -> str | None:
        return _gguf_name(QWEN3_ADAPTER, hf_name)

    # WHEN: mapping per-head query/key norm paths.
    # THEN: qwen3 additionally maps the per-head query/key norms.
    assert name_map("model.layers.2.self_attn.q_norm.weight") == "blk.2.attn_q_norm.weight"
    assert name_map("model.layers.2.self_attn.k_norm.weight") == "blk.2.attn_k_norm.weight"


def test_qwen2_adapter_name_map_includes_qkv_bias_not_q_k_norm() -> None:
    # GIVEN: the qwen2 adapter's tensor rules.
    def name_map(hf_name: str) -> str | None:
        return _gguf_name(QWEN2_ADAPTER, hf_name)

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
    q_result = llama_rope_permute(q_tensor, config)
    k_result = llama_rope_permute(k_tensor, config)
    v_result = llama_rope_permute(v_tensor, config)

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
    result = llama_rope_permute(tensor, config)

    # THEN: each head's rows are de-interleaved into (even-half, odd-half) order,
    # matching llama.cpp's converter.
    assert result.int_codes is not None
    permuted = result.int_codes.reshape(8, 1, block_size)
    expected = original_codes[torch.tensor([0, 2, 1, 3, 4, 6, 5, 7])]
    torch.testing.assert_close(permuted, expected)


def test_llama_rope_permute_applies_to_float_projections() -> None:
    # GIVEN: an unquantized q_proj with 2 heads over 8 rows.
    config = _Config(n_head=2, n_head_kv=2)
    data = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
    tensor = ExtractedTensor(
        hf_name="model.layers.0.self_attn.q_proj.weight",
        kind="float",
        rows=8,
        cols=4,
        float_data=data,
    )

    # WHEN: applying the transform.
    result = llama_rope_permute(tensor, config)

    # THEN: its rows get the same de-interleave as quantized projections.
    torch.testing.assert_close(result.float_data, data[torch.tensor([0, 2, 1, 3, 4, 6, 5, 7])])


def test_llama_rope_permute_moves_offsets_with_scales() -> None:
    # GIVEN: an asymmetric k_proj whose per-block offsets are tied to its scales.
    config = _Config(n_head=2, n_head_kv=2)
    tensor = make_quantized_tensor("model.layers.0.self_attn.k_proj.weight", rows=8, cols=32)
    assert tensor.scales is not None
    tensor = replace(tensor, offsets=tensor.scales * 10, quant_format=GGUF_Q4_1)

    # WHEN: applying the transform.
    result = llama_rope_permute(tensor, config)

    # THEN: every block keeps its (scale, offset) pair.
    assert result.scales is not None
    torch.testing.assert_close(result.offsets, result.scales * 10)


def test_llama_rope_permute_keeps_extracted_shapes() -> None:
    # GIVEN: a q_proj shaped like extraction output: (rows, cols) codes and
    # (rows, blocks_per_row) scales.
    config = _Config(n_head=2, n_head_kv=2)
    flat = make_quantized_tensor("model.layers.0.self_attn.q_proj.weight", rows=8, cols=64)
    assert flat.int_codes is not None and flat.scales is not None
    tensor = replace(
        flat, int_codes=flat.int_codes.reshape(8, 64), scales=flat.scales.reshape(8, 2)
    )

    # WHEN: applying the transform.
    result = llama_rope_permute(tensor, config)

    # THEN: the row-major layout is kept, so later rules (e.g. a column fusion)
    # see the same shapes as for unpermuted tensors.
    assert result.int_codes is not None and result.scales is not None
    assert result.int_codes.shape == (8, 64)
    assert result.scales.shape == (8, 2)


def test_llama_rope_permute_rejects_indivisible_row_count() -> None:
    # GIVEN: a row count not divisible by 2 * n_head.
    config = _Config(n_head=3, n_head_kv=3)
    tensor = make_quantized_tensor("model.layers.0.self_attn.q_proj.weight", rows=8, cols=32)

    # WHEN / THEN: the transform refuses rather than silently corrupting rows.
    with pytest.raises(ExportError, match="not divisible"):
        llama_rope_permute(tensor, config)


def test_qwen_adapters_have_no_transforms() -> None:
    # GIVEN: the built-in qwen2 and qwen3 adapters.
    # THEN: neither carries a Transform rule (no RoPE permute needed).
    for adapter in (QWEN2_ADAPTER, QWEN3_ADAPTER):
        assert not any(isinstance(rule, Transform) for rule in adapter.tensor_rules)


def test_llama_adapter_has_rope_permute_transform() -> None:
    # GIVEN: the built-in llama adapter.
    transforms = [rule for rule in LLAMA_ADAPTER.tensor_rules if isinstance(rule, Transform)]

    # THEN: it has exactly one transform — the rope permute.
    assert len(transforms) == 1
    assert transforms[0].fn is llama_rope_permute


def test_llama_adapter_permutes_quantized_q_proj_before_renaming() -> None:
    # GIVEN: a quantized q_proj tensor and a 2-head config.
    config = _Config(n_head=2, n_head_kv=2)
    tensor = make_quantized_tensor("model.layers.0.self_attn.q_proj.weight", rows=8, cols=32)

    # WHEN: running the llama adapter's tensor rules.
    (result,) = apply_tensor_rules([tensor], LLAMA_ADAPTER.tensor_rules, config)

    # THEN: the tensor is renamed and its rows are permuted like the bare transform.
    expected = llama_rope_permute(tensor, config)
    assert result.gguf_name == "blk.0.attn_q.weight"
    torch.testing.assert_close(result.int_codes, expected.int_codes)


def test_adapters_carry_tokenizer_discriminators() -> None:
    # GIVEN: the built-in llama, qwen2, and qwen3 adapters.
    # THEN: each carries the llama.cpp tokenizer-pre string matching its
    # architecture's convention (llama-bpe for llama3-family, qwen2 for both qwens).
    assert LLAMA_ADAPTER.tokenizer_pre == "llama-bpe"
    assert QWEN2_ADAPTER.tokenizer_pre == "qwen2"
    assert QWEN3_ADAPTER.tokenizer_pre == "qwen2"
    for adapter in (LLAMA_ADAPTER, QWEN2_ADAPTER, QWEN3_ADAPTER):
        assert adapter.tokenizer_model == "gpt2"


def test_llama_metadata_reads_rope_theta_from_transformers_config() -> None:
    # GIVEN: a transformers config, which keeps rope_theta in rope_parameters.
    # transformers is not part of the CI `test` dependency group.
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        rope_parameters={"rope_type": "default", "rope_theta": 500000.0}
    )
    writer = GGUFWriter(None, arch=LLAMA_ADAPTER.gguf_arch)

    # WHEN: writing the llama metadata.
    LLAMA_ADAPTER.write_metadata(writer, config)

    # THEN: the GGUF RoPE base is the model's.
    metadata = {key: value.value for key, value in writer.kv_data[0].items()}
    assert metadata["llama.rope.freq_base"] == 500000.0


@pytest.mark.parametrize(
    "rope_fields",
    [
        {"rope_parameters": {"rope_type": "default", "rope_theta": 500000.0}},
        {"rope_theta": 500000.0},
    ],
    ids=["transformers-5-rope-parameters", "legacy-rope-theta"],
)
def test_metadata_writes_rope_theta_from_config(rope_fields: dict[str, Any]) -> None:
    # GIVEN: a config carrying the RoPE base in one of the two supported places.
    config = _metadata_config(**rope_fields)
    writer = mock.create_autospec(GGUFWriter, instance=True)

    # WHEN: writing the llama metadata.
    LLAMA_ADAPTER.write_metadata(writer, config)

    # THEN: the config's RoPE base is written, not a default.
    writer.add_rope_freq_base.assert_called_once_with(500000.0)


@pytest.mark.parametrize(
    "rope_fields",
    [
        {"rope_parameters": {"rope_type": "default", "rope_theta": 500000.0}},
        {"rope_theta": 500000.0},
        {"rope_theta": 500000.0, "rope_scaling": None},
    ],
    ids=["transformers-5-rope-parameters", "legacy-no-rope-scaling", "legacy-rope-scaling-none"],
)
def test_metadata_writes_no_rope_scaling_for_unscaled_rope(rope_fields: dict[str, Any]) -> None:
    # GIVEN: a config with unscaled RoPE.
    config = _metadata_config(**rope_fields)
    writer = mock.create_autospec(GGUFWriter, instance=True)

    # WHEN: writing the llama metadata.
    LLAMA_ADAPTER.write_metadata(writer, config)

    # THEN: the scaling type is "none" and no scaling factor is written.
    writer.add_rope_scaling_type.assert_called_once_with(RopeScalingType.NONE)
    writer.add_rope_scaling_factor.assert_not_called()


@pytest.mark.parametrize(
    "rope_fields",
    [
        {"rope_parameters": {"rope_type": "linear", "factor": 2.0, "rope_theta": 500000.0}},
        {"rope_theta": 500000.0, "rope_scaling": {"type": "linear", "factor": 2.0}},
    ],
    ids=["transformers-5-rope-parameters", "legacy-rope-scaling"],
)
def test_metadata_writes_rope_scaling_from_config(rope_fields: dict[str, Any]) -> None:
    # GIVEN: a config with linearly scaled RoPE.
    config = _metadata_config(**rope_fields)
    writer = mock.create_autospec(GGUFWriter, instance=True)

    # WHEN: writing the llama metadata.
    LLAMA_ADAPTER.write_metadata(writer, config)

    # THEN: the scaling type and factor are written.
    writer.add_rope_scaling_type.assert_called_once_with(RopeScalingType.LINEAR)
    writer.add_rope_scaling_factor.assert_called_once_with(2.0)


def test_metadata_without_rope_theta_raises() -> None:
    # GIVEN: a config with neither rope_parameters nor rope_theta.
    config = _metadata_config()
    writer = mock.create_autospec(GGUFWriter, instance=True)

    # WHEN / THEN: the metadata writer refuses instead of guessing a RoPE base.
    with pytest.raises(ExportError, match="rope_theta"):
        LLAMA_ADAPTER.write_metadata(writer, config)
