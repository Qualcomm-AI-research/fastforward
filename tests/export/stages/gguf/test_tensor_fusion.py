# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Tests for the :class:`Fuse` tensor rule in the GGUF export pipeline."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import fastforward as ff
import pytest
import torch

from fastforward.export.stages.gguf import GGUF_Q4_0, GGUF_Q8_0, ArchAdapter, Fuse
from fastforward.export.stages.gguf._extract import ExtractedTensor, _is_claimed

# Fuse ignores the source config.
_CONFIG: Any = SimpleNamespace()


def _make_quantized_tensor(
    hf_name: str, rows: int, cols: int, block_size: int = 32
) -> ExtractedTensor:
    """Create a fake quantized ExtractedTensor for testing."""
    blocks_per_row = cols // block_size
    int_codes = torch.randint(-8, 8, (rows * blocks_per_row, block_size), dtype=torch.int8)
    scales = torch.randn(rows, blocks_per_row)
    return ExtractedTensor(
        hf_name=hf_name,
        kind="quantized",
        rows=rows,
        cols=cols,
        int_codes=int_codes,
        scales=scales,
    )


def _make_float_tensor(hf_name: str, rows: int, cols: int) -> ExtractedTensor:
    """Create a fake float ExtractedTensor for testing."""
    return ExtractedTensor(
        hf_name=hf_name,
        kind="float",
        rows=rows,
        cols=cols,
        float_data=torch.randn(rows, cols),
    )


def test_fusion_without_matches_is_noop() -> None:
    """A fusion whose sources match nothing returns tensors unchanged."""
    # GIVEN: a list of tensors none of which match the fusion's sources.
    tensors = [
        _make_quantized_tensor("layer.0.q.weight", 128, 128),
        _make_quantized_tensor("layer.0.k.weight", 64, 128),
    ]
    fusion = Fuse(sources=(r"block\.(\d+)\.a", r"block\.(\d+)\.b"), target_name=r"block.\1.ab")

    # WHEN: applying the fusion.
    result = fusion.apply(tensors, _CONFIG)

    # THEN: the same tensors are returned in the same order.
    assert len(result) == len(tensors)
    assert all(out is inp for out, inp in zip(result, tensors))


def test_basic_qkv_fusion() -> None:
    """Row-concat of Q, K, V tensors produces correct fused shape."""
    # GIVEN: Q, K, V quantized tensors and a norm float tensor for one layer.
    q = _make_quantized_tensor("model.layers.0.self_attn.q_proj.weight", 128, 128)
    k = _make_quantized_tensor("model.layers.0.self_attn.k_proj.weight", 64, 128)
    v = _make_quantized_tensor("model.layers.0.self_attn.v_proj.weight", 64, 128)
    norm = _make_float_tensor("model.layers.0.input_layernorm.weight", 128, 1)

    fusion = Fuse(
        sources=(
            r"model\.layers\.(\d+)\.self_attn\.q_proj\.weight",
            r"model\.layers\.(\d+)\.self_attn\.k_proj\.weight",
            r"model\.layers\.(\d+)\.self_attn\.v_proj\.weight",
        ),
        target_name=r"model.layers.\1.self_attn.qkv_proj.weight",
        axis=0,
    )

    # WHEN: applying the QKV fusion.
    result = fusion.apply([q, k, v, norm], _CONFIG)

    # THEN: Q, K, V are replaced by a single fused tensor; norm passes through.
    assert len(result) == 2
    fused = next(t for t in result if "qkv" in t.hf_name)
    assert fused.hf_name == "model.layers.0.self_attn.qkv_proj.weight"
    assert fused.rows == 128 + 64 + 64
    assert fused.cols == 128
    assert fused.kind == "quantized"
    assert any(t.hf_name == "model.layers.0.input_layernorm.weight" for t in result)


def test_multi_layer_regex_fusion() -> None:
    """Pattern-based fusion works across multiple layers."""
    # GIVEN: Q, K, V tensors across 3 layers.
    tensors = []
    for layer_idx in range(3):
        tensors.append(
            _make_quantized_tensor(f"model.layers.{layer_idx}.self_attn.q_proj.weight", 128, 128)
        )
        tensors.append(
            _make_quantized_tensor(f"model.layers.{layer_idx}.self_attn.k_proj.weight", 64, 128)
        )
        tensors.append(
            _make_quantized_tensor(f"model.layers.{layer_idx}.self_attn.v_proj.weight", 64, 128)
        )

    fusion = Fuse(
        sources=(
            r"model\.layers\.(\d+)\.self_attn\.q_proj\.weight",
            r"model\.layers\.(\d+)\.self_attn\.k_proj\.weight",
            r"model\.layers\.(\d+)\.self_attn\.v_proj\.weight",
        ),
        target_name=r"model.layers.\1.self_attn.qkv_proj.weight",
        axis=0,
    )

    # WHEN: applying fusion with regex capture groups.
    result = fusion.apply(tensors, _CONFIG)

    # THEN: each layer produces one fused tensor with combined row count.
    assert len(result) == 3
    for layer_idx in range(3):
        fused = next(
            t for t in result if t.hf_name == f"model.layers.{layer_idx}.self_attn.qkv_proj.weight"
        )
        assert fused.rows == 256
        assert fused.cols == 128


def test_float_tensor_fusion() -> None:
    """Float tensors can be fused too."""
    # GIVEN: two float tensors matching a fusion spec.
    a = _make_float_tensor("block.0.norm_a.weight", 128, 1)
    b = _make_float_tensor("block.0.norm_b.weight", 64, 1)

    fusion = Fuse(
        sources=(
            r"block\.(\d+)\.norm_a\.weight",
            r"block\.(\d+)\.norm_b\.weight",
        ),
        target_name=r"block.\1.norm_fused.weight",
        axis=0,
    )

    # WHEN: applying fusion.
    result = fusion.apply([a, b], _CONFIG)

    # THEN: the fused tensor has combined rows and is still float.
    assert len(result) == 1
    assert result[0].rows == 192
    assert result[0].kind == "float"


def test_mixed_kind_raises() -> None:
    """Fusing quantized + float tensors raises ExportError."""
    # GIVEN: one quantized and one float tensor matching the same fusion spec.
    q = _make_quantized_tensor("layer.0.q.weight", 128, 128)
    f = _make_float_tensor("layer.0.k.weight", 64, 128)

    fusion = Fuse(
        sources=(
            r"layer\.(\d+)\.q\.weight",
            r"layer\.(\d+)\.k\.weight",
        ),
        target_name=r"layer.\1.qk.weight",
        axis=0,
    )

    # WHEN/THEN: applying fusion raises due to mixed kinds.
    with pytest.raises(ff.exceptions.ExportError, match="mixed kinds"):
        fusion.apply([q, f], _CONFIG)


def test_missing_source_raises() -> None:
    """A partial match (some sources found, others not) raises ExportError."""
    # GIVEN: only one of the two required sources is present.
    q = _make_quantized_tensor("layer.0.q.weight", 128, 128)

    fusion = Fuse(
        sources=(
            r"layer\.(\d+)\.q\.weight",
            r"layer\.(\d+)\.k\.weight",
        ),
        target_name=r"layer.\1.qk.weight",
        axis=0,
    )

    # WHEN/THEN: applying fusion raises due to incomplete match.
    with pytest.raises(ff.exceptions.ExportError, match="incomplete"):
        fusion.apply([q], _CONFIG)


def test_incompatible_cols_raises() -> None:
    """Row-concat with different column counts raises ExportError."""
    # GIVEN: two tensors with mismatched column counts.
    q = _make_quantized_tensor("layer.0.q.weight", 128, 128)
    k = _make_quantized_tensor("layer.0.k.weight", 64, 64)

    fusion = Fuse(
        sources=(
            r"layer\.(\d+)\.q\.weight",
            r"layer\.(\d+)\.k\.weight",
        ),
        target_name=r"layer.\1.qk.weight",
        axis=0,
    )

    # WHEN/THEN: applying fusion raises due to incompatible shapes.
    with pytest.raises(ff.exceptions.ExportError, match="column counts"):
        fusion.apply([q, k], _CONFIG)


def test_incompatible_float_cols_raises() -> None:
    """Float shapes are validated from their data, even without ``cols``."""
    # GIVEN: two extracted-style float tensors (no ``cols``) with different widths.
    a = ExtractedTensor("layer.0.a.weight", kind="float", rows=4, float_data=torch.zeros(4, 8))
    b = ExtractedTensor("layer.0.b.weight", kind="float", rows=4, float_data=torch.zeros(4, 16))

    fusion = Fuse(
        sources=(r"layer\.(\d+)\.a\.weight", r"layer\.(\d+)\.b\.weight"),
        target_name=r"layer.\1.ab.weight",
        axis=0,
    )

    # WHEN/THEN: applying fusion raises an ExportError rather than a torch error.
    with pytest.raises(ff.exceptions.ExportError, match="column counts"):
        fusion.apply([a, b], _CONFIG)


def test_extraction_includes_fusion_sources() -> None:
    """Parameters matching fusion source patterns are claimed for extraction."""
    # GIVEN: an adapter whose only rule is a q/k/v fusion.
    adapter = ArchAdapter(
        gguf_arch="test",
        tensor_rules=(
            Fuse(
                sources=(
                    r"model\.layers\.(\d+)\.self_attn\.q_proj\.weight",
                    r"model\.layers\.(\d+)\.self_attn\.k_proj\.weight",
                    r"model\.layers\.(\d+)\.self_attn\.v_proj\.weight",
                ),
                target_name=r"model.layers.\1.self_attn.qkv_proj.weight",
                axis=0,
            ),
        ),
        write_metadata=lambda w, c: None,
        tokenizer_model="gpt2",
        tokenizer_pre="default",
    )

    # WHEN: checking individual parameter names against the adapter's rules.
    # THEN: q/k/v are claimed as fusion sources; o_proj and norms are not.
    assert _is_claimed("model.layers.0.self_attn.q_proj.weight", adapter)
    assert _is_claimed("model.layers.0.self_attn.k_proj.weight", adapter)
    assert _is_claimed("model.layers.2.self_attn.v_proj.weight", adapter)
    assert not _is_claimed("model.layers.0.self_attn.o_proj.weight", adapter)
    assert not _is_claimed("model.norm.weight", adapter)


def test_mixed_quant_format_fusion_raises() -> None:
    """Fusing tensors with different quant_format values raises ExportError."""
    # GIVEN: two quantized tensors with different quant formats.
    q = _make_quantized_tensor("layer.0.q.weight", 128, 128)
    k = _make_quantized_tensor("layer.0.k.weight", 64, 128)
    q.quant_format = GGUF_Q4_0
    k.quant_format = GGUF_Q8_0

    fusion = Fuse(
        sources=(
            r"layer\.(\d+)\.q\.weight",
            r"layer\.(\d+)\.k\.weight",
        ),
        target_name=r"layer.\1.qk.weight",
        axis=0,
    )

    # WHEN/THEN: applying fusion raises due to conflicting quant formats.
    with pytest.raises(ff.exceptions.ExportError, match="different quant formats"):
        fusion.apply([q, k], _CONFIG)


def test_fusion_propagates_quant_format() -> None:
    """Fusing tensors with the same quant_format propagates it to the result."""
    # GIVEN: two quantized tensors sharing the same quant format.
    q = _make_quantized_tensor("layer.0.q.weight", 128, 128)
    k = _make_quantized_tensor("layer.0.k.weight", 64, 128)
    q.quant_format = GGUF_Q4_0
    k.quant_format = GGUF_Q4_0

    fusion = Fuse(
        sources=(
            r"layer\.(\d+)\.q\.weight",
            r"layer\.(\d+)\.k\.weight",
        ),
        target_name=r"layer.\1.qk.weight",
        axis=0,
    )

    # WHEN: applying fusion.
    result = fusion.apply([q, k], _CONFIG)

    # THEN: the fused tensor carries the shared format.
    assert len(result) == 1
    assert result[0].quant_format is GGUF_Q4_0
