# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Tests for the GGUF export pipeline stages (tensor rules, packing, writing).

These cover the stage wiring and the tensor-rule/pack composition
without requiring the optional ``gguf`` package. The write stage's dependency
handling is tested via its error paths.
"""

from typing import Any
from unittest import mock

import gguf
import pytest
import torch

from fastforward.exceptions import ExportError
from fastforward.export.stages.gguf import (
    GGUF_Q4_0,
    GGUF_Q4_1,
    GGUF_Q8_0,
    LLAMA_ADAPTER,
    QWEN3_ADAPTER,
)
from fastforward.export.stages.gguf._extract import ExtractedTensor
from fastforward.export.stages.gguf._packing import default_format_registry
from fastforward.export.stages.gguf.adapter import GgufFormatRegistry
from fastforward.export.stages.gguf.gguf_export_stages import (
    _cast_float,
    stage_apply_tensor_rules,
    stage_extract_quantized_weights,
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


def test_apply_tensor_rules_permutes_and_names_llama_q_proj() -> None:
    # GIVEN: a llama q_proj tensor (8 rows, 2 heads => de-interleave order).
    tensor = make_quantized_tensor("model.layers.0.self_attn.q_proj.weight", rows=8, cols=32)
    assert tensor.int_codes is not None
    original_codes = tensor.int_codes.reshape(8, 1, BLOCK_SIZE).clone()
    context = {"arch_adapter": LLAMA_ADAPTER, "model_config": _Config()}

    # WHEN: applying the adapter's tensor rules.
    (result,) = stage_apply_tensor_rules(([tensor],), [], context)

    # THEN: the tensor is named and its rows are reordered by llama.cpp's RoPE
    # de-interleave permutation.
    assert result.gguf_name == "blk.0.attn_q.weight"
    assert result.int_codes is not None
    permuted = result.int_codes.reshape(8, 1, BLOCK_SIZE)
    expected = original_codes[torch.tensor([0, 2, 1, 3, 4, 6, 5, 7])]
    torch.testing.assert_close(permuted, expected)


def test_apply_tensor_rules_does_not_permute_for_qwen3() -> None:
    # GIVEN: the same tensor but a qwen3 context (no permute).
    tensor = make_quantized_tensor("model.layers.0.self_attn.q_proj.weight", rows=8, cols=32)
    assert tensor.int_codes is not None
    original = tensor.int_codes.clone()
    context = {"arch_adapter": QWEN3_ADAPTER, "model_config": _Config()}

    # WHEN: applying the adapter's tensor rules.
    (result,) = stage_apply_tensor_rules(([tensor],), [], context)

    # THEN: the tensor is named and its codes are unchanged.
    assert result.gguf_name == "blk.0.attn_q.weight"
    torch.testing.assert_close(result.int_codes, original)


def test_apply_tensor_rules_drops_unnamed_tensors() -> None:
    # GIVEN: one mappable and one unmappable tensor.
    keep = make_quantized_tensor("model.layers.1.self_attn.v_proj.weight", rows=4, cols=32)
    drop = make_quantized_tensor("model.rotary_emb.inv_freq", rows=1, cols=32)
    context = {"arch_adapter": LLAMA_ADAPTER, "model_config": _Config()}

    # WHEN: applying the adapter's tensor rules.
    result = stage_apply_tensor_rules(([keep, drop],), [], context)

    # THEN: only the mappable tensor survives, tagged with its GGUF name.
    assert len(result) == 1
    assert result[0].gguf_name == "blk.1.attn_v.weight"


def test_apply_tensor_rules_requires_model_config() -> None:
    # GIVEN: a context without model_config.
    context = {"arch_adapter": LLAMA_ADAPTER}

    # WHEN/THEN: applying the rules raises because they receive the config.
    with pytest.raises(ExportError, match="model_config"):
        stage_apply_tensor_rules(([],), [], context)


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
    context: dict[str, object] = {}

    # WHEN: packing.
    packed = stage_pack_gguf_blocks(([quant, float_tensor],), [], context)

    # THEN: the quantized tensor becomes (block_bytes, format) tuple; the float
    # tensor passes through unchanged.
    block_bytes, fmt = packed["quantized"]["blk.0.ffn_up.weight"]
    assert block_bytes.shape == (4, 18)
    assert block_bytes.dtype == torch.uint8
    assert fmt.name == "Q4_0"
    torch.testing.assert_close(packed["float"]["output_norm.weight"], torch.ones(32))


def test_extract_stage_rejects_invalid_quant_format() -> None:
    # GIVEN: a quant_format that is not a GgufQuantFormat instance.
    context: dict[str, object] = {
        "quant_format": "Q2_K",
        "arch_adapter": LLAMA_ADAPTER,
        "model_config": _Config(),
    }

    # WHEN / THEN: extraction raises a helpful ExportError.
    with pytest.raises(ExportError, match="must be a GgufQuantFormat"):
        stage_extract_quantized_weights((torch.nn.Linear(4, 4),), [], context)


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
        "quantized": {"blk.0.attn_q.weight": (torch.zeros(4, 18, dtype=torch.uint8), GGUF_Q4_0)},
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


def test_write_gguf_applies_float_type_policy_from_context(tmp_path: Any) -> None:
    # GIVEN: a context carrying an F16 default with norms overridden to F32,
    # and packed data with one norm and one non-norm float tensor.
    context = {
        "arch_adapter": LLAMA_ADAPTER,
        "model_config": _make_full_config(),
        "output_dir": tmp_path,
        "model_name": "policy",
        "float_type": "F16",
        "float_type_overrides": {r".*norm.*": "F32"},
    }
    packed: dict[str, dict[str, Any]] = {
        "quantized": {"blk.0.attn_q.weight": (torch.zeros(4, 18, dtype=torch.uint8), GGUF_Q4_0)},
        "float": {"output_norm.weight": torch.ones(32), "token_embd.weight": torch.ones(64, 32)},
    }
    mock_writer = mock.MagicMock()

    # WHEN: invoking the write stage.
    with mock.patch(
        "fastforward.export.stages.gguf.gguf_export_stages.GGUFWriter",
        return_value=mock_writer,
    ):
        stage_write_gguf((packed,), [], context)

    # THEN: each float tensor is written with the type the context's policy selects.
    raw_dtypes = {
        call.args[0]: call.kwargs["raw_dtype"] for call in mock_writer.add_tensor.call_args_list
    }
    assert raw_dtypes["output_norm.weight"] == gguf.GGMLQuantizationType.F32
    assert raw_dtypes["token_embd.weight"] == gguf.GGMLQuantizationType.F16


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


# --- GgufFormatRegistry tests ---


def test_registry_register_and_resolve() -> None:
    # GIVEN: an empty registry.
    registry = GgufFormatRegistry()

    # WHEN: registering Q4_0.
    registry.register(GGUF_Q4_0)

    # THEN: resolve finds it by matching properties.
    assert registry.resolve(4, True, 32) is GGUF_Q4_0


def test_registry_resolve_returns_none_on_miss() -> None:
    # GIVEN: a registry with only Q4_0.
    registry = GgufFormatRegistry()
    registry.register(GGUF_Q4_0)

    # WHEN: resolving properties that don't match.
    result = registry.resolve(8, True, 32)

    # THEN: returns None.
    assert result is None


def test_registry_first_returns_first_registered() -> None:
    # GIVEN: a registry with Q4_0 and Q8_0 registered in that order.
    registry = GgufFormatRegistry()
    registry.register(GGUF_Q4_0)
    registry.register(GGUF_Q8_0)

    # THEN: first() returns Q4_0.
    assert registry.first() is GGUF_Q4_0


def test_default_format_registry_contains_builtins() -> None:
    # GIVEN: the default registry.
    registry = default_format_registry()

    # THEN: all built-in formats are resolvable.
    assert registry.resolve(4, True, 32) is GGUF_Q4_0
    assert registry.resolve(8, True, 32) is GGUF_Q8_0
    assert registry.resolve(4, False, 32) is GGUF_Q4_1


# --- Mixed-precision pipeline tests ---


def _make_q8_tensor(gguf_name: str, rows: int, cols: int) -> ExtractedTensor:
    """Build a minimal Q8_0 quantized ExtractedTensor."""
    block_size = GGUF_Q8_0.block_size
    n_blocks = (rows * cols) // block_size
    return ExtractedTensor(
        hf_name=gguf_name,
        kind="quantized",
        rows=rows,
        cols=cols,
        int_codes=torch.randint(-128, 127, (n_blocks, block_size), dtype=torch.int8),
        scales=torch.rand(n_blocks) * 0.95 + 0.05,
        gguf_name=gguf_name,
        quant_format=GGUF_Q8_0,
    )


def test_pack_gguf_blocks_mixed_precision() -> None:
    # GIVEN: one Q4_0 tensor and one Q8_0 tensor, both with GGUF names set.
    q4_tensor = make_quantized_tensor("model.layers.0.mlp.up_proj.weight", rows=4, cols=32)
    q4_tensor.gguf_name = "blk.0.ffn_up.weight"

    q8_tensor = _make_q8_tensor("blk.0.attn_q.weight", rows=4, cols=32)

    context: dict[str, object] = {}

    # WHEN: packing.
    packed = stage_pack_gguf_blocks(([q4_tensor, q8_tensor],), [], context)

    # THEN: each tensor is packed with its own format.
    block_bytes_q4, fmt_q4 = packed["quantized"]["blk.0.ffn_up.weight"]
    assert fmt_q4.name == "Q4_0"
    assert block_bytes_q4.shape == (4, GGUF_Q4_0.block_bytes)

    block_bytes_q8, fmt_q8 = packed["quantized"]["blk.0.attn_q.weight"]
    assert fmt_q8.name == "Q8_0"
    assert block_bytes_q8.shape == (4, GGUF_Q8_0.block_bytes)


def test_write_gguf_mixed_precision_dominant_file_type(tmp_path: Any) -> None:
    # GIVEN: packed data with 2 Q4_0 tensors and 1 Q8_0 tensor.
    context = {
        "arch_adapter": LLAMA_ADAPTER,
        "model_config": _make_full_config(),
        "output_dir": tmp_path,
        "model_name": "mixed",
    }
    packed: dict[str, dict[str, Any]] = {
        "quantized": {
            "blk.0.attn_q.weight": (torch.zeros(4, 18, dtype=torch.uint8), GGUF_Q4_0),
            "blk.0.attn_k.weight": (torch.zeros(4, 18, dtype=torch.uint8), GGUF_Q4_0),
            "blk.0.ffn_up.weight": (torch.zeros(4, 34, dtype=torch.uint8), GGUF_Q8_0),
        },
        "float": {},
    }

    mock_writer = mock.MagicMock()

    # WHEN: invoking the write stage.
    with mock.patch(
        "fastforward.export.stages.gguf.gguf_export_stages.GGUFWriter",
        return_value=mock_writer,
    ):
        stage_write_gguf((packed,), [], context)

    # THEN: file_type is set to Q4_0's (dominant: 2 out of 3 tensors).
    mock_writer.add_file_type.assert_called_once_with(GGUF_Q4_0.file_type)
