# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Tests for the tensor fusion stage in the GGUF export pipeline."""

from __future__ import annotations

import fastforward as ff
import pytest
import torch

from fastforward.export.stages.gguf import ArchAdapter, TensorFusion
from fastforward.export.stages.gguf._extract import ExtractedTensor, _is_fusion_source
from fastforward.export.stages.gguf._fusion import apply_fusions


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


def test_no_fusions_is_noop() -> None:
    """An empty fusions list returns tensors unchanged."""
    # GIVEN: a list of tensors and no fusion specs.
    tensors = [
        _make_quantized_tensor("layer.0.q.weight", 128, 128),
        _make_quantized_tensor("layer.0.k.weight", 64, 128),
    ]

    # WHEN: applying an empty fusions list.
    result = apply_fusions(tensors, [])

    # THEN: the exact same list is returned.
    assert result is tensors


def test_basic_qkv_fusion() -> None:
    """Row-concat of Q, K, V tensors produces correct fused shape."""
    # GIVEN: Q, K, V quantized tensors and a norm float tensor for one layer.
    q = _make_quantized_tensor("model.layers.0.self_attn.q_proj.weight", 128, 128)
    k = _make_quantized_tensor("model.layers.0.self_attn.k_proj.weight", 64, 128)
    v = _make_quantized_tensor("model.layers.0.self_attn.v_proj.weight", 64, 128)
    norm = _make_float_tensor("model.layers.0.input_layernorm.weight", 128, 1)

    fusion = TensorFusion(
        sources=(
            r"model\.layers\.(\d+)\.self_attn\.q_proj\.weight",
            r"model\.layers\.(\d+)\.self_attn\.k_proj\.weight",
            r"model\.layers\.(\d+)\.self_attn\.v_proj\.weight",
        ),
        target_name=r"model.layers.\1.self_attn.qkv_proj.weight",
        axis=0,
    )

    # WHEN: applying the QKV fusion.
    result = apply_fusions([q, k, v, norm], [fusion])

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

    fusion = TensorFusion(
        sources=(
            r"model\.layers\.(\d+)\.self_attn\.q_proj\.weight",
            r"model\.layers\.(\d+)\.self_attn\.k_proj\.weight",
            r"model\.layers\.(\d+)\.self_attn\.v_proj\.weight",
        ),
        target_name=r"model.layers.\1.self_attn.qkv_proj.weight",
        axis=0,
    )

    # WHEN: applying fusion with regex capture groups.
    result = apply_fusions(tensors, [fusion])

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

    fusion = TensorFusion(
        sources=(
            r"block\.(\d+)\.norm_a\.weight",
            r"block\.(\d+)\.norm_b\.weight",
        ),
        target_name=r"block.\1.norm_fused.weight",
        axis=0,
    )

    # WHEN: applying fusion.
    result = apply_fusions([a, b], [fusion])

    # THEN: the fused tensor has combined rows and is still float.
    assert len(result) == 1
    assert result[0].rows == 192
    assert result[0].kind == "float"


def test_mixed_kind_raises() -> None:
    """Fusing quantized + float tensors raises ExportError."""
    # GIVEN: one quantized and one float tensor matching the same fusion spec.
    q = _make_quantized_tensor("layer.0.q.weight", 128, 128)
    f = _make_float_tensor("layer.0.k.weight", 64, 128)

    fusion = TensorFusion(
        sources=(
            r"layer\.(\d+)\.q\.weight",
            r"layer\.(\d+)\.k\.weight",
        ),
        target_name=r"layer.\1.qk.weight",
        axis=0,
    )

    # WHEN/THEN: applying fusion raises due to mixed kinds.
    with pytest.raises(ff.exceptions.ExportError, match="mixed kinds"):
        apply_fusions([q, f], [fusion])


def test_missing_source_raises() -> None:
    """A partial match (some sources found, others not) raises ExportError."""
    # GIVEN: only one of the two required sources is present.
    q = _make_quantized_tensor("layer.0.q.weight", 128, 128)

    fusion = TensorFusion(
        sources=(
            r"layer\.(\d+)\.q\.weight",
            r"layer\.(\d+)\.k\.weight",
        ),
        target_name=r"layer.\1.qk.weight",
        axis=0,
    )

    # WHEN/THEN: applying fusion raises due to incomplete match.
    with pytest.raises(ff.exceptions.ExportError, match="incomplete"):
        apply_fusions([q], [fusion])


def test_incompatible_cols_raises() -> None:
    """Row-concat with different column counts raises ExportError."""
    # GIVEN: two tensors with mismatched column counts.
    q = _make_quantized_tensor("layer.0.q.weight", 128, 128)
    k = _make_quantized_tensor("layer.0.k.weight", 64, 64)

    fusion = TensorFusion(
        sources=(
            r"layer\.(\d+)\.q\.weight",
            r"layer\.(\d+)\.k\.weight",
        ),
        target_name=r"layer.\1.qk.weight",
        axis=0,
    )

    # WHEN/THEN: applying fusion raises due to incompatible shapes.
    with pytest.raises(ff.exceptions.ExportError, match="column counts"):
        apply_fusions([q, k], [fusion])


def test_extraction_includes_fusion_sources() -> None:
    """Parameters matching fusion source patterns are extracted even without a name_map entry."""

    # GIVEN: a name_map that only knows about o_proj (not q/k/v), but an adapter
    # whose fusions list declares q/k/v as fusion sources.
    def name_map(hf_name: str) -> str | None:
        if "o_proj" in hf_name:
            return "blk.0.attn_output.weight"
        return None

    adapter = ArchAdapter(
        gguf_arch="test",
        name_map=name_map,
        transforms=[],
        write_metadata=lambda w, c: None,
        tokenizer_model="gpt2",
        tokenizer_pre="default",
        fusions=[
            TensorFusion(
                sources=(
                    r"model\.layers\.(\d+)\.self_attn\.q_proj\.weight",
                    r"model\.layers\.(\d+)\.self_attn\.k_proj\.weight",
                    r"model\.layers\.(\d+)\.self_attn\.v_proj\.weight",
                ),
                target_name=r"model.layers.\1.self_attn.qkv_proj.weight",
                axis=0,
            ),
        ],
    )

    # WHEN: checking individual parameter names against the adapter's fusions.
    # THEN: q/k/v are recognized as fusion sources; o_proj is not.
    assert _is_fusion_source("model.layers.0.self_attn.q_proj.weight", adapter)
    assert _is_fusion_source("model.layers.0.self_attn.k_proj.weight", adapter)
    assert _is_fusion_source("model.layers.2.self_attn.v_proj.weight", adapter)
    assert not _is_fusion_source("model.layers.0.self_attn.o_proj.weight", adapter)
    assert not _is_fusion_source("model.norm.weight", adapter)
