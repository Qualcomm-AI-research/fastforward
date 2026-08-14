# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Tests for the GGUF export pipeline stages (transforms, naming, packing).

These cover the stage wiring and the RoPE-permute/name-map/pack composition
without requiring the optional ``gguf`` package. The write stage's dependency
handling is tested via its error paths.
"""

from typing import Any
from unittest import mock

import pytest
import torch

from fastforward.exceptions import ExportError
from fastforward.export.stages.gguf import GGUF_Q4_0, LLAMA_ADAPTER, QWEN3_ADAPTER, ArchAdapter
from fastforward.export.stages.gguf._extract import ExtractedTensor
from fastforward.export.stages.gguf.gguf_export_stages import (
    _cast_float,
    _resolve_float_type,
    stage_apply_target_transforms,
    stage_map_tensor_names,
    stage_pack_gguf_blocks,
    stage_write_gguf,
)
from tests.export.stages.gguf.conftest import make_quantized_tensor

BLOCK_SIZE = GGUF_Q4_0.block_size


class _Config:
    def __init__(self) -> None:
        self.num_attention_heads = 2
        self.num_key_value_heads = 2
        self.tie_word_embeddings = False


def test_apply_transforms_permutes_llama_q_proj() -> None:
    # GIVEN: a llama q_proj tensor (8 rows, 2 heads => de-interleave order).
    tensor = make_quantized_tensor("model.layers.0.self_attn.q_proj.weight", rows=8, cols=32)
    assert tensor.int_codes is not None
    original_codes = tensor.int_codes.reshape(8, 1, BLOCK_SIZE).clone()
    context = {"arch_adapter": LLAMA_ADAPTER, "model_config": _Config(), "quant_format": GGUF_Q4_0}

    # WHEN: applying target transforms.
    (result,) = stage_apply_target_transforms(([tensor],), [], context)

    # THEN: rows are reordered by llama.cpp's RoPE de-interleave permutation.
    assert result.int_codes is not None
    permuted = result.int_codes.reshape(8, 1, BLOCK_SIZE)
    expected = original_codes[torch.tensor([0, 2, 1, 3, 4, 6, 5, 7])]
    torch.testing.assert_close(permuted, expected)


def test_apply_transforms_noop_for_qwen3() -> None:
    # GIVEN: the same tensor but a qwen3 context (no permute).
    tensor = make_quantized_tensor("model.layers.0.self_attn.q_proj.weight", rows=8, cols=32)
    assert tensor.int_codes is not None
    original = tensor.int_codes.clone()
    context = {"arch_adapter": QWEN3_ADAPTER, "model_config": _Config(), "quant_format": GGUF_Q4_0}

    # WHEN: applying transforms.
    (result,) = stage_apply_target_transforms(([tensor],), [], context)

    # THEN: codes are unchanged.
    torch.testing.assert_close(result.int_codes, original)


def test_map_tensor_names_assigns_gguf_names_and_drops_unmapped() -> None:
    # GIVEN: one mappable and one unmappable tensor.
    keep = make_quantized_tensor("model.layers.1.self_attn.v_proj.weight", rows=4, cols=32)
    drop = make_quantized_tensor("model.rotary_emb.inv_freq", rows=1, cols=32)
    context = {"arch_adapter": LLAMA_ADAPTER}

    # WHEN: mapping names.
    result = stage_map_tensor_names(([keep, drop],), [], context)

    # THEN: only the mappable tensor survives, tagged with its GGUF name.
    assert len(result) == 1
    assert result[0].gguf_name == "blk.1.attn_v.weight"


def test_pack_gguf_blocks_splits_quantized_and_float() -> None:
    # GIVEN: one quantized tensor and one float tensor, both with GGUF names.
    quant = make_quantized_tensor("model.layers.0.mlp.up_proj.weight", rows=4, cols=32)
    quant.gguf_name = "blk.0.ffn_up.weight"
    float_tensor = ExtractedTensor(
        hf_name="model.norm.weight",
        kind="float",
        rows=32,
        float_data=torch.ones(32),
        gguf_name="output_norm.weight",
    )
    context = {"quant_format": GGUF_Q4_0}

    # WHEN: packing.
    packed = stage_pack_gguf_blocks(([quant, float_tensor],), [], context)

    # THEN: the quantized tensor becomes (rows, 18) uint8 block bytes; the float
    # tensor passes through unchanged.
    assert packed["quantized"]["blk.0.ffn_up.weight"].shape == (4, 18)
    assert packed["quantized"]["blk.0.ffn_up.weight"].dtype == torch.uint8
    torch.testing.assert_close(packed["float"]["output_norm.weight"], torch.ones(32))


def test_pack_gguf_blocks_rejects_invalid_quant_format() -> None:
    # GIVEN: a quant_format that is not a GgufQuantFormat instance.
    context = {"quant_format": "Q2_K"}

    # WHEN / THEN: packing raises a helpful ExportError.
    with pytest.raises(ExportError, match="must be a GgufQuantFormat"):
        stage_pack_gguf_blocks(([],), [], context)


def test_write_gguf_requires_model_config() -> None:
    # GIVEN: a context missing model_config.
    context = {
        "arch_adapter": LLAMA_ADAPTER,
        "quant_format": GGUF_Q4_0,
        "output_dir": ".",
        "model_name": "m",
    }
    packed: dict[str, dict[str, Any]] = {"quantized": {}, "float": {}}

    # WHEN / THEN: the write stage refuses without a config (before importing gguf).
    with pytest.raises(ExportError, match="requires 'model_config'"):
        stage_write_gguf((packed,), [], context)


def test_write_gguf_requires_adapter() -> None:
    # GIVEN: a context missing arch_adapter.
    context = {
        "model_config": _Config(),
        "quant_format": GGUF_Q4_0,
        "output_dir": ".",
        "model_name": "m",
    }
    packed: dict[str, dict[str, Any]] = {"quantized": {}, "float": {}}

    # WHEN / THEN: the write stage refuses without an adapter.
    with pytest.raises(ExportError, match="requires 'arch_adapter'"):
        stage_write_gguf((packed,), [], context)


def test_write_gguf_closes_writer_on_exception(tmp_path: Any) -> None:
    # GIVEN: a mocked GGUFWriter that raises mid-write (during add_tensor). The
    # stage promises to close the writer even on failure so the target file
    # handle is released.
    context = {
        "arch_adapter": LLAMA_ADAPTER,
        "model_config": _make_full_config(),
        "quant_format": GGUF_Q4_0,
        "output_dir": tmp_path,
        "model_name": "boom",
    }
    packed: dict[str, dict[str, Any]] = {
        "quantized": {"blk.0.attn_q.weight": torch.zeros(4, 18, dtype=torch.uint8)},
        "float": {},
    }

    mock_writer = mock.MagicMock()
    mock_writer.add_tensor.side_effect = RuntimeError("simulated write failure")

    # WHEN: invoking the write stage with the rigged writer.
    with mock.patch(
        "fastforward.export.stages.gguf.gguf_export_stages.GGUFWriter",
        return_value=mock_writer,
    ):
        with pytest.raises(RuntimeError, match="simulated write failure"):
            stage_write_gguf((packed,), [], context)

    # THEN: close() was still called on the writer.
    mock_writer.close.assert_called_once()


def _make_full_config() -> Any:
    """A namespace populated enough for LLAMA_ADAPTER.write_metadata to succeed."""

    class C:
        max_position_embeddings = 128
        hidden_size = 32
        num_hidden_layers = 2
        intermediate_size = 64
        num_attention_heads = 2
        num_key_value_heads = 2
        rope_theta = 10000.0
        rms_norm_eps = 1e-5
        vocab_size = 64
        tie_word_embeddings = False
        rope_scaling = None

    return C()


def test_resolve_float_type_default() -> None:
    # GIVEN: an adapter with F16 default and no overrides.
    adapter = ArchAdapter(
        gguf_arch="test",
        name_map=lambda n: n,
        transforms=[],
        write_metadata=lambda w, c: None,
        tokenizer_model="gpt2",
        tokenizer_pre="default",
        float_type="F16",
    )

    # WHEN: resolving a tensor with no override match.
    result = _resolve_float_type("blk.0.ffn_up.weight", adapter)

    # THEN: falls back to the default.
    assert result == "F16"


def test_resolve_float_type_override_matches() -> None:
    # GIVEN: an adapter with F16 default but norms overridden to F32.
    adapter = ArchAdapter(
        gguf_arch="test",
        name_map=lambda n: n,
        transforms=[],
        write_metadata=lambda w, c: None,
        tokenizer_model="gpt2",
        tokenizer_pre="default",
        float_type="F16",
        float_type_overrides={r".*norm.*": "F32"},
    )

    # WHEN: resolving tensors.
    norm_result = _resolve_float_type("blk.0.attn_norm.weight", adapter)
    weight_result = _resolve_float_type("blk.0.ffn_up.weight", adapter)

    # THEN: norm matches the override, other falls back to default.
    assert norm_result == "F32"
    assert weight_result == "F16"


def test_resolve_float_type_first_override_wins() -> None:
    # GIVEN: an adapter with multiple overrides that could both match.
    adapter = ArchAdapter(
        gguf_arch="test",
        name_map=lambda n: n,
        transforms=[],
        write_metadata=lambda w, c: None,
        tokenizer_model="gpt2",
        tokenizer_pre="default",
        float_type="F32",
        float_type_overrides={
            r"blk\.0\..*": "F16",
            r".*norm.*": "BF16",
        },
    )

    # WHEN: resolving a tensor that matches both patterns.
    result = _resolve_float_type("blk.0.attn_norm.weight", adapter)

    # THEN: first matching pattern wins.
    assert result == "F16"


def test_cast_float_f32_noop() -> None:
    # GIVEN: float32 data.
    data = torch.randn(4, 8)

    # WHEN: casting to F32.
    result = _cast_float(data, "F32")

    # THEN: returns the same tensor unchanged.
    assert result is data


def test_cast_float_f16() -> None:
    # GIVEN: float32 data.
    data = torch.randn(4, 8)

    # WHEN: casting to F16.
    result = _cast_float(data, "F16")

    # THEN: result is float16.
    assert result.dtype == torch.float16
    assert result.shape == (4, 8)


def test_cast_float_bf16() -> None:
    # GIVEN: float32 data.
    data = torch.randn(4, 8)

    # WHEN: casting to BF16.
    result = _cast_float(data, "BF16")

    # THEN: result is uint8 raw bytes (2 bytes per element).
    assert result.dtype == torch.uint8
    assert result.shape == (4, 16)


def test_cast_float_unsupported_raises() -> None:
    # GIVEN: float32 data and an invalid target type.
    data = torch.randn(4, 8)

    # WHEN / THEN: casting to an unsupported type raises ExportError.
    with pytest.raises(ExportError, match="Unsupported float_type 'MXFP4'"):
        _cast_float(data, "MXFP4")
