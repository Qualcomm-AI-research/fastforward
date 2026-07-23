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
from fastforward.export.stages.gguf import GGUF_Q4_0, LLAMA_ADAPTER, QWEN3_ADAPTER
from fastforward.export.stages.gguf._extract import ExtractedTensor
from fastforward.export.stages.gguf.gguf_export_stages import (
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
