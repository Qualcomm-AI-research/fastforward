# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Tests for the built-in GGUF architecture adapters.

The adapters (:data:`LLAMA_ADAPTER`, :data:`QWEN2_ADAPTER`, :data:`QWEN3_ADAPTER`,
:data:`GEMMA4_ADAPTER`, :data:`QWEN25VL_ADAPTER`, :data:`QWEN3VL_ADAPTER`) are the
pipeline's default reference implementations of the public :class:`ArchAdapter`
surface. The mmproj adapters are tested in ``test_mmproj.py``. These tests exercise the fields users
are most likely to depend on (tensor rules, transforms, metadata).
"""

import importlib.util

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

import pytest
import torch

from fastforward.exceptions import ExportError
from fastforward.export.stages.gguf import (
    GEMMA4_ADAPTER,
    GGUF_Q4_0,
    GGUF_Q4_1,
    LLAMA_ADAPTER,
    QWEN2_ADAPTER,
    QWEN3_ADAPTER,
    QWEN3VL_ADAPTER,
    QWEN25VL_ADAPTER,
    ArchAdapter,
    ExtractedTensor,
    GgufSourceConfig,
    Transform,
    gemma4_rope_freqs,
    llama3_rope_freqs,
    llama_rope_permute,
    text_config_view,
)
from fastforward.export.stages.gguf.rules import apply_tensor_rules
from gguf import GGUFWriter, RopeScalingType
from tests.export.stages.gguf.conftest import make_quantized_tensor

# transformers is not part of the CI `test` dependency group.
requires_transformers = pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None, reason="requires `transformers`"
)


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


def _gguf_name(
    adapter: ArchAdapter, hf_name: str, config: GgufSourceConfig | None = None
) -> str | None:
    """Run the adapter's tensor rules on one float tensor and return its GGUF name."""
    tensor = ExtractedTensor(hf_name=hf_name, kind="float", rows=2, float_data=torch.zeros(2))
    config = config or _Config(n_head=1, n_head_kv=1)
    result = apply_tensor_rules([tensor], adapter.tensor_rules, config)
    return next((t.gguf_name for t in result if t.hf_name == hf_name), None)


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


@pytest.mark.parametrize(
    ("adapter", "tokenizer_model", "tokenizer_pre"),
    [
        (LLAMA_ADAPTER, "gpt2", "llama-bpe"),
        (QWEN2_ADAPTER, "gpt2", "qwen2"),
        (QWEN3_ADAPTER, "gpt2", "qwen2"),
        (GEMMA4_ADAPTER, "gemma4", None),
    ],
    ids=["llama", "qwen2", "qwen3", "gemma4"],
)
def test_adapter_carries_tokenizer_discriminators(
    adapter: ArchAdapter, tokenizer_model: str, tokenizer_pre: str | None
) -> None:
    # GIVEN: a built-in adapter.
    # THEN: it carries the llama.cpp tokenizer-model and pre-tokenizer strings
    # matching its architecture's convention (gemma4 has no pre-tokenizer).
    assert adapter.tokenizer_model == tokenizer_model
    assert adapter.tokenizer_pre == tokenizer_pre


@requires_transformers
def test_llama_metadata_reads_rope_theta_from_transformers_config() -> None:
    # GIVEN: a transformers config, which keeps rope_theta in rope_parameters.
    config = _llama_config(rope_parameters={"rope_type": "default", "rope_theta": 500000.0})
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
        {
            "rope_parameters": {
                "type": "mrope",
                "mrope_section": [16, 24, 24],
                "rope_theta": 1e6,
                "rope_type": "default",
            }
        },
        {"rope_theta": 1e6, "rope_scaling": {"type": "mrope", "mrope_section": [16, 24, 24]}},
    ],
    ids=[
        "transformers-5-rope-parameters",
        "legacy-no-rope-scaling",
        "legacy-rope-scaling-none",
        "qwen2.5-vl-stale-legacy-type",
        "legacy-mrope",
    ],
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


_LLAMA3_ROPE = {
    "rope_type": "llama3",
    "rope_theta": 500000.0,
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}


def _llama_config(**kwargs: Any) -> Any:
    from transformers import LlamaConfig

    return LlamaConfig(**kwargs)


def _llama3_config() -> Any:
    return _llama_config(
        hidden_size=256,
        num_attention_heads=2,
        max_position_embeddings=131072,
        rope_parameters=_LLAMA3_ROPE,
    )


@requires_transformers
def test_llama3_rope_freqs_matches_transformers_scaling() -> None:
    # GIVEN: a Llama 3.1-style config (128-dim heads, llama3 RoPE scaling).
    config = _llama3_config()

    # WHEN: synthesizing the rope_freqs tensor.
    freqs = llama3_rope_freqs(config)["rope_freqs.weight"]

    # THEN: dividing the unscaled frequencies by it gives transformers' llama3
    # frequencies, covering the kept, interpolated, and fully scaled bands.
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    head_dim = 128
    exponents = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    unscaled = 1.0 / 500000.0**exponents
    scaled, _ = ROPE_INIT_FUNCTIONS["llama3"](config, "cpu")
    torch.testing.assert_close(freqs, unscaled / scaled)
    assert freqs.min() == 1.0 and freqs.max() == 8.0


@requires_transformers
def test_llama3_rope_freqs_is_empty_without_llama3_scaling() -> None:
    # GIVEN: a Llama config with default RoPE.
    config = _llama_config()

    # THEN: no rope_freqs tensor is synthesized.
    assert llama3_rope_freqs(config) == {}


@requires_transformers
def test_llama_metadata_does_not_write_llama3_factor_as_linear_scaling() -> None:
    # GIVEN: a Llama 3.1-style config.
    writer = GGUFWriter(None, arch=LLAMA_ADAPTER.gguf_arch)

    # WHEN: writing the llama metadata.
    LLAMA_ADAPTER.write_metadata(writer, _llama3_config())

    # THEN: no scaling type or factor is written (llama.cpp would read a bare
    # factor as linear scaling); the scaling comes from rope_freqs.weight.
    metadata = {key: value.value for key, value in writer.kv_data[0].items()}
    assert metadata["llama.rope.scaling.type"] == "none"
    assert "llama.rope.scaling.factor" not in metadata


@requires_transformers
def test_llama_metadata_rejects_unsupported_rope_scaling() -> None:
    # GIVEN: a config with a RoPE scaling type llama.cpp cannot express.
    config = _llama_config(rope_parameters={"rope_type": "dynamic", "factor": 2.0})
    writer = GGUFWriter(None, arch=LLAMA_ADAPTER.gguf_arch)

    # WHEN/THEN: writing the metadata raises instead of writing a wrong factor.
    with pytest.raises(ExportError, match="RoPE scaling type 'dynamic'"):
        LLAMA_ADAPTER.write_metadata(writer, config)


def _gemma4_rope_parameters(full_attention: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "full_attention": full_attention,
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
    }


def _gemma4_config(
    full_attention: dict[str, Any] | None, global_head_dim: int | None
) -> GgufSourceConfig:
    return cast(
        GgufSourceConfig,
        SimpleNamespace(
            global_head_dim=global_head_dim,
            rope_parameters=_gemma4_rope_parameters(full_attention),
        ),
    )


def _gemma4_e4b_config(**overrides: Any) -> GgufSourceConfig:
    """The text config of ``google/gemma-4-E4B-it``, with optional field overrides."""
    fields: dict[str, Any] = {
        "num_hidden_layers": 42,
        "max_position_embeddings": 131072,
        "hidden_size": 2560,
        "intermediate_size": 10240,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "global_head_dim": 512,
        "rms_norm_eps": 1e-6,
        "vocab_size": 262144,
        "layer_types": (["sliding_attention"] * 5 + ["full_attention"]) * 7,
        "rope_parameters": _gemma4_rope_parameters({
            "rope_type": "proportional",
            "partial_rotary_factor": 0.25,
            "rope_theta": 1e6,
        }),
        "sliding_window": 512,
        "num_kv_shared_layers": 18,
        "hidden_size_per_layer_input": 256,
        "tie_word_embeddings": True,
    }
    return cast(GgufSourceConfig, SimpleNamespace(**{**fields, **overrides}))


def test_gemma4_adapter_name_map_covers_gemma4_tensors() -> None:
    # GIVEN: the gemma4 adapter's tensor rules.
    def name_map(hf_name: str) -> str | None:
        return _gguf_name(GEMMA4_ADAPTER, hf_name, _gemma4_e4b_config())

    # WHEN: mapping HuggingFace parameter and buffer paths.
    # THEN: parameter and buffer paths are correctly mapped to llama.cpp format.
    assert name_map("model.layers.3.post_attention_layernorm.weight") == (
        "blk.3.post_attention_norm.weight"
    )
    assert name_map("model.layers.3.pre_feedforward_layernorm.weight") == "blk.3.ffn_norm.weight"
    assert name_map("model.embed_tokens_per_layer.weight") == "per_layer_token_embd.weight"
    assert name_map("model.layers.3.per_layer_projection.weight") == "blk.3.proj.weight"
    assert name_map("model.layers.3.layer_scalar") == "blk.3.layer_output_scale.weight"

    # THEN: the multimodal layout maps the same; its towers and the tied head are dropped.
    assert name_map("model.language_model.layers.0.self_attn.q_proj.weight") == (
        "blk.0.attn_q.weight"
    )
    assert name_map("model.vision_tower.encoder.layers.0.self_attn.q_proj.weight") is None
    assert name_map("lm_head.weight") is None


def test_gemma4_metadata_matches_reference_gguf() -> None:
    # GIVEN: an in-memory GGUF writer.
    writer = GGUFWriter(None, arch=GEMMA4_ADAPTER.gguf_arch)

    # WHEN: writing the gemma-4-E4B metadata.
    GEMMA4_ADAPTER.write_metadata(writer, _gemma4_e4b_config())

    # THEN: the gemma4-specific keys match google/gemma-4-E4B-it-qat-q4_0-gguf.
    metadata = {key: value.value for key, value in writer.kv_data[0].items()}
    assert metadata["gemma4.attention.key_length"] == 512
    assert metadata["gemma4.attention.key_length_swa"] == 256
    assert metadata["gemma4.attention.sliding_window_pattern"] == ([True] * 5 + [False]) * 7
    assert metadata["gemma4.attention.shared_kv_layers"] == 18
    assert metadata["gemma4.embedding_length_per_layer_input"] == 256
    assert metadata["gemma4.rope.freq_base_swa"] == 10000.0
    assert metadata["gemma4.rope.dimension_count_swa"] == 256


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"enable_moe_block": True}, "MoE"),
        ({"global_head_dim": None}, "'global_head_dim'"),
        (
            {"rope_parameters": _gemma4_rope_parameters({"rope_type": "proportional"})},
            "'rope_theta'",
        ),
        ({"num_kv_shared_layers": 19}, "sliding-window layer followed by a full-attention"),
    ],
    ids=["moe", "missing-field", "missing-rope-theta", "kv-sharing-unsupported-by-llama-cpp"],
)
def test_gemma4_metadata_rejects_unsupported_config(overrides: dict[str, Any], match: str) -> None:
    # GIVEN: an unsupported gemma4 config.
    writer = GGUFWriter(None, arch=GEMMA4_ADAPTER.gguf_arch)

    # WHEN/THEN: writing its metadata raises.
    with pytest.raises(ExportError, match=match):
        GEMMA4_ADAPTER.write_metadata(writer, _gemma4_e4b_config(**overrides))


def test_gemma4_adapter_pins_rope_freqs_to_f32() -> None:
    # GIVEN: the gemma4 adapter's float-type pin.
    pin = GEMMA4_ADAPTER.pin_float_type
    assert pin is not None

    # WHEN/THEN: only rope_freqs.weight is pinned, to F32, whatever the default.
    assert pin("rope_freqs.weight", 1, "F16") == "F32"
    assert pin("blk.0.attn_norm.weight", 1, "F16") is None


def test_gemma4_rope_freqs_matches_llama_cpp_converter() -> None:
    # GIVEN: the gemma-4-E4B full-attention RoPE config (512-dim heads, 1/4 rotated).
    config = _gemma4_config(
        {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1e6},
        global_head_dim=512,
    )

    # WHEN: synthesizing the rope_freqs tensor.
    tensors = gemma4_rope_freqs(config)

    # THEN: one float32 entry per frequency pair: 64 rotated pairs (factor 1.0)
    # followed by 192 unrotated ones (factor 1e30), as llama.cpp's converter emits.
    assert list(tensors) == ["rope_freqs.weight"]
    freqs = tensors["rope_freqs.weight"]
    assert freqs.dtype == torch.float32
    expected = torch.tensor([1.0] * 64 + [1e30] * 192, dtype=torch.float32)
    torch.testing.assert_close(freqs, expected)


def test_gemma4_rope_freqs_applies_rope_scaling_factor() -> None:
    # GIVEN: a full-attention RoPE config with scaling factor 2.0.
    config = _gemma4_config(
        {"rope_type": "proportional", "partial_rotary_factor": 0.25, "factor": 2.0},
        global_head_dim=512,
    )

    # WHEN: synthesizing the rope_freqs tensor.
    freqs = gemma4_rope_freqs(config)["rope_freqs.weight"]

    # THEN: rotated pairs are divided by the factor; unrotated pairs are unchanged.
    expected = torch.tensor([2.0] * 64 + [1e30] * 192, dtype=torch.float32)
    torch.testing.assert_close(freqs, expected)


@pytest.mark.parametrize(
    ("full_attention", "global_head_dim", "match"),
    [
        ({"rope_type": "proportional", "factor": 0.0}, 512, "positive RoPE scaling 'factor'"),
        ({"rope_type": "proportional", "factor": "2"}, 512, "positive RoPE scaling 'factor'"),
        (None, 512, "rope_type='proportional'"),
        ({"rope_type": "default"}, 512, "rope_type='proportional'"),
        ({"rope_type": "proportional"}, None, "global_head_dim"),
    ],
    ids=[
        "zero-factor",
        "non-numeric-factor",
        "missing-full-attention",
        "non-proportional",
        "missing-head-dim",
    ],
)
def test_gemma4_rope_freqs_rejects_unsupported_config(
    full_attention: dict[str, Any] | None, global_head_dim: int | None, match: str
) -> None:
    # GIVEN: a config lacking the proportional full-attention RoPE inputs.
    config = _gemma4_config(full_attention, global_head_dim)

    # WHEN/THEN: synthesizing raises.
    with pytest.raises(ExportError, match=match):
        gemma4_rope_freqs(config)


def _vl_config(tie_word_embeddings: bool = True, **text_overrides: Any) -> SimpleNamespace:
    """A top-level vision-language config whose text_config has no tie flag."""
    text_config = _metadata_config(
        rope_parameters={"rope_type": "default", "rope_theta": 1e6, "mrope_section": [4, 2, 2]},
        **text_overrides,
    )
    return SimpleNamespace(
        text_config=text_config,
        vision_config=SimpleNamespace(deepstack_visual_indexes=[0, 2]),
        tie_word_embeddings=tie_word_embeddings,
    )


def test_text_config_view_reads_text_config_then_top_level() -> None:
    # GIVEN: a multimodal config whose text_config sets one field to None.
    config = SimpleNamespace(
        text_config=SimpleNamespace(hidden_size=32, tie_word_embeddings=None),
        tie_word_embeddings=True,
        hidden_size=999,
    )

    # WHEN: viewing it as a text config.
    view: Any = text_config_view(config)

    # THEN: text_config wins, None and missing fields fall back to the top level,
    # and fields missing on both raise AttributeError (so getattr defaults work).
    assert view.hidden_size == 32
    assert view.tie_word_embeddings is True
    assert view.text_config is config.text_config
    assert getattr(view, "head_dim", "default") == "default"


def test_text_config_view_passes_text_only_config_through() -> None:
    # GIVEN: a config without a text_config.
    config = _metadata_config()

    # WHEN / THEN: it is returned unchanged.
    assert text_config_view(config) is config


@pytest.mark.parametrize(
    ("adapter", "extra_names"),
    [
        (QWEN25VL_ADAPTER, {"self_attn.q_proj.bias": "attn_q.bias"}),
        (QWEN3VL_ADAPTER, {"self_attn.q_norm.weight": "attn_q_norm.weight"}),
    ],
    ids=["qwen2.5-vl", "qwen3-vl"],
)
def test_qwen_vl_adapter_name_map_exports_only_the_text_decoder(
    adapter: ArchAdapter, extra_names: dict[str, str]
) -> None:
    # GIVEN: the resolved top-level config of an untied vision-language checkpoint.
    config = adapter.resolve_config(_vl_config(tie_word_embeddings=False))

    def name_map(hf_name: str) -> str | None:
        return _gguf_name(adapter, hf_name, config)

    # WHEN / THEN: the nested text decoder maps like its Qwen2/Qwen3 counterpart.
    prefix = "model.language_model."
    assert name_map(f"{prefix}embed_tokens.weight") == "token_embd.weight"
    assert name_map(f"{prefix}norm.weight") == "output_norm.weight"
    assert name_map(f"{prefix}layers.1.mlp.down_proj.weight") == "blk.1.ffn_down.weight"
    for hf_suffix, gguf_suffix in extra_names.items():
        assert name_map(f"{prefix}layers.1.{hf_suffix}") == f"blk.1.{gguf_suffix}"
    assert name_map("lm_head.weight") == "output.weight"

    # THEN: the vision tower is not exported.
    assert name_map("model.visual.blocks.0.attn.qkv.weight") is None
    assert name_map("model.visual.merger.mlp.0.weight") is None


@pytest.mark.parametrize(
    "adapter", [QWEN25VL_ADAPTER, QWEN3VL_ADAPTER], ids=["qwen2.5-vl", "qwen3-vl"]
)
def test_qwen_vl_adapter_drops_lm_head_tied_on_top_level_config(adapter: ArchAdapter) -> None:
    # GIVEN: a config that ties the embeddings only at the top level, as
    # transformers 5 does for Qwen2.5-VL.
    config = adapter.resolve_config(_vl_config(tie_word_embeddings=True))

    # WHEN / THEN: the tied lm_head is dropped.
    assert _gguf_name(adapter, "lm_head.weight", config) is None


@pytest.mark.parametrize(
    "adapter", [QWEN25VL_ADAPTER, QWEN3VL_ADAPTER], ids=["qwen2.5-vl", "qwen3-vl"]
)
def test_qwen_vl_adapter_rejects_bare_text_config(adapter: ArchAdapter) -> None:
    # GIVEN: the text_config of a vision-language checkpoint.
    text_config = _vl_config().text_config

    # WHEN / THEN: resolving it raises instead of losing the top-level fields.
    with pytest.raises(ExportError, match="top-level model config"):
        adapter.resolve_config(text_config)


def _qwen25vl_7b_config() -> Any:
    # Qwen2.5-VL checkpoints name multimodal RoPE in the legacy `type` key.
    return SimpleNamespace(
        text_config=SimpleNamespace(
            vocab_size=152064,
            hidden_size=3584,
            intermediate_size=18944,
            num_hidden_layers=28,
            num_attention_heads=28,
            num_key_value_heads=4,
            max_position_embeddings=128000,
            rms_norm_eps=1e-6,
            rope_theta=1e6,
            rope_scaling={"type": "mrope", "mrope_section": [16, 24, 24]},
        ),
        vision_config=SimpleNamespace(),
        tie_word_embeddings=False,
    )


def _qwen3vl_2b_config() -> Any:
    return SimpleNamespace(
        text_config=SimpleNamespace(
            vocab_size=151936,
            hidden_size=2048,
            intermediate_size=6144,
            num_hidden_layers=28,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            max_position_embeddings=262144,
            rms_norm_eps=1e-6,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 5e6,
                "mrope_section": [24, 20, 20],
                "mrope_interleaved": True,
            },
        ),
        vision_config=SimpleNamespace(deepstack_visual_indexes=[5, 11, 17]),
        tie_word_embeddings=True,
    )


@pytest.mark.parametrize(
    ("adapter", "make_config", "expected"),
    [
        (
            QWEN25VL_ADAPTER,
            _qwen25vl_7b_config,
            {
                "qwen2vl.block_count": 28,
                "qwen2vl.context_length": 128000,
                "qwen2vl.embedding_length": 3584,
                "qwen2vl.feed_forward_length": 18944,
                "qwen2vl.attention.head_count": 28,
                "qwen2vl.attention.head_count_kv": 4,
                "qwen2vl.rope.freq_base": 1e6,
                "qwen2vl.rope.dimension_sections": [16, 24, 24, 0],
            },
        ),
        (
            QWEN3VL_ADAPTER,
            _qwen3vl_2b_config,
            {
                "qwen3vl.block_count": 28,
                "qwen3vl.context_length": 262144,
                "qwen3vl.embedding_length": 2048,
                "qwen3vl.feed_forward_length": 6144,
                "qwen3vl.attention.head_count": 16,
                "qwen3vl.attention.head_count_kv": 8,
                "qwen3vl.attention.key_length": 128,
                "qwen3vl.attention.value_length": 128,
                "qwen3vl.rope.freq_base": 5e6,
                "qwen3vl.rope.dimension_sections": [24, 20, 20, 0],
                "qwen3vl.n_deepstack_layers": 3,
            },
        ),
    ],
    ids=["qwen2.5-vl-7b", "qwen3-vl-2b"],
)
def test_qwen_vl_metadata_matches_reference_gguf(
    adapter: ArchAdapter, make_config: Any, expected: dict[str, Any]
) -> None:
    # GIVEN: the top-level config of a released checkpoint.
    config = adapter.resolve_config(make_config())
    writer = GGUFWriter(None, arch=adapter.gguf_arch)

    # WHEN: writing the text-decoder metadata.
    adapter.write_metadata(writer, config)

    # THEN: the architecture keys match the llama.cpp-converted GGUF that AI Hub
    # ships (unsloth/Qwen2.5-VL-7B-Instruct-GGUF, unsloth/Qwen3-VL-2B-Instruct-GGUF),
    # and the legacy `type="mrope"` is not mistaken for RoPE scaling.
    metadata = {key: value.value for key, value in writer.kv_data[0].items()}
    assert {key: metadata.get(key) for key in expected} == expected
    assert metadata[f"{adapter.gguf_arch}.attention.layer_norm_rms_epsilon"] == pytest.approx(1e-6)
    assert metadata[f"{adapter.gguf_arch}.rope.scaling.type"] == "none"


@pytest.mark.parametrize(
    ("mrope_section", "match"),
    [
        (None, "1 to 4 entries"),
        ([2, 2, 2, 1, 1], "1 to 4 entries"),
        ([4, 4, 4], "covers 24 rotary dims, but the head dimension is 16"),
    ],
    ids=["missing", "too-many", "wrong-head-dim"],
)
def test_qwen_vl_metadata_rejects_invalid_mrope_section(
    mrope_section: list[int] | None, match: str
) -> None:
    # GIVEN: a config (head dim 16) with an unusable mrope_section.
    rope_parameters: dict[str, Any] = {"rope_type": "default", "rope_theta": 1e6}
    if mrope_section is not None:
        rope_parameters["mrope_section"] = mrope_section
    config = _vl_config()
    config.text_config.rope_parameters = rope_parameters
    writer = GGUFWriter(None, arch=QWEN25VL_ADAPTER.gguf_arch)

    # WHEN / THEN: writing the metadata raises.
    with pytest.raises(ExportError, match=match):
        QWEN25VL_ADAPTER.write_metadata(writer, QWEN25VL_ADAPTER.resolve_config(config))


def test_qwen3vl_metadata_defaults_mrope_section() -> None:
    # GIVEN: a Qwen3-VL config (head dim 128) that omits mrope_section, as
    # transformers' defaults do.
    config = _vl_config(head_dim=128)
    config.text_config.rope_parameters = {"rope_type": "default", "rope_theta": 5e6}
    config.text_config.num_hidden_layers = 3
    writer = GGUFWriter(None, arch=QWEN3VL_ADAPTER.gguf_arch)

    # WHEN: writing the metadata.
    QWEN3VL_ADAPTER.write_metadata(writer, QWEN3VL_ADAPTER.resolve_config(config))

    # THEN: transformers' default sections are written.
    metadata = {key: value.value for key, value in writer.kv_data[0].items()}
    assert metadata["qwen3vl.rope.dimension_sections"] == [24, 20, 20, 0]


@pytest.mark.parametrize("n_deepstack_layers", [2, 3])
def test_qwen3vl_metadata_rejects_deepstack_layers_not_below_decoder_layers(
    n_deepstack_layers: int,
) -> None:
    # GIVEN: a two-layer Qwen3-VL config with as many deepstack layers as, or
    # more than, decoder layers, which crashes llama.cpp's multi-token decoding.
    config = _vl_config()
    config.text_config.num_hidden_layers = 2
    config.vision_config.deepstack_visual_indexes = list(range(n_deepstack_layers))
    writer = GGUFWriter(None, arch=QWEN3VL_ADAPTER.gguf_arch)

    # WHEN / THEN: writing the metadata raises.
    with pytest.raises(ExportError, match="needs more decoder layers than deepstack layers"):
        QWEN3VL_ADAPTER.write_metadata(writer, QWEN3VL_ADAPTER.resolve_config(config))
