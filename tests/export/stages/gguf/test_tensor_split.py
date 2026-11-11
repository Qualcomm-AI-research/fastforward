# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Tests for the :class:`Split` tensor rule in the GGUF export pipeline."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from fastforward.exceptions import ExportError
from fastforward.export.stages.gguf import (
    GGUF_Q4_0,
    GGUF_Q4_1,
    GGUF_Q8_0,
    ArchAdapter,
    ExtractedTensor,
    Fuse,
    GgufQuantFormat,
    Split,
)
from fastforward.export.stages.gguf._extract import _is_claimed
from fastforward.export.stages.gguf.gguf_export_stages import stage_pack_gguf_blocks

# Split only reads the config through a `sizes` callable.
_CONFIG: Any = SimpleNamespace(q_dim=64, kv_dim=32)

_QKV_PATTERN = r"visual\.blocks\.(\d+)\.attn\.qkv\.(weight|bias)"
_QKV_TARGETS = (
    r"visual.blocks.\1.attn.q.\2",
    r"visual.blocks.\1.attn.k.\2",
    r"visual.blocks.\1.attn.v.\2",
)


def _gqa_sizes(config: Any) -> tuple[int, int, int]:
    return (config.q_dim, config.kv_dim, config.kv_dim)


def _float_tensor(hf_name: str, data: torch.Tensor) -> ExtractedTensor:
    return ExtractedTensor(
        hf_name=hf_name,
        kind="float",
        rows=data.shape[0],
        cols=data.shape[1] if data.ndim > 1 else None,
        float_data=data,
    )


def _quantized_tensor(
    hf_name: str, rows: int, cols: int, quant_format: GgufQuantFormat = GGUF_Q4_0
) -> ExtractedTensor:
    """A quantized tensor laid out like extraction produces it: codes (rows, cols)."""
    blocks_per_row = cols // quant_format.block_size
    offsets = None
    if not quant_format.symmetric:
        offsets = torch.randint(-8, 8, (rows, blocks_per_row)).float()
    return ExtractedTensor(
        hf_name=hf_name,
        kind="quantized",
        rows=rows,
        cols=cols,
        int_codes=torch.randint(-8, 8, (rows, cols), dtype=torch.int8),
        scales=torch.rand(rows, blocks_per_row) + 0.05,
        offsets=offsets,
        quant_format=quant_format,
    )


def _by_name(tensors: list[ExtractedTensor]) -> dict[str, ExtractedTensor]:
    return {tensor.hf_name: tensor for tensor in tensors}


def _float_data(tensor: ExtractedTensor) -> torch.Tensor:
    assert tensor.float_data is not None
    return tensor.float_data


def test_split_cuts_float_weight_and_bias_into_equal_parts_per_layer() -> None:
    # GIVEN: fused qkv weights and biases for two layers.
    weights = [torch.randn(96, 16), torch.randn(96, 16)]
    biases = [torch.randn(96), torch.randn(96)]
    tensors = [
        tensor
        for layer in (0, 1)
        for tensor in (
            _float_tensor(f"visual.blocks.{layer}.attn.qkv.weight", weights[layer]),
            _float_tensor(f"visual.blocks.{layer}.attn.qkv.bias", biases[layer]),
        )
    ]

    # WHEN: splitting them into q, k and v.
    result = _by_name(Split(_QKV_PATTERN, _QKV_TARGETS).apply(tensors, _CONFIG))

    # THEN: every layer's weight and bias is cut into three consecutive 32-row parts,
    # named through the back-references.
    for layer in (0, 1):
        for index, part in enumerate("qkv"):
            rows = slice(32 * index, 32 * (index + 1))
            weight = result[f"visual.blocks.{layer}.attn.{part}.weight"]
            bias = result[f"visual.blocks.{layer}.attn.{part}.bias"]
            assert weight.float_data is not None and bias.float_data is not None
            torch.testing.assert_close(weight.float_data, weights[layer][rows])
            torch.testing.assert_close(bias.float_data, biases[layer][rows])
            assert (weight.rows, weight.cols) == (32, 16)
            assert (bias.rows, bias.cols) == (32, None)
            assert weight.gguf_name == ""
    assert len(result) == 12


def test_split_replaces_source_in_place_and_keeps_other_tensors() -> None:
    # GIVEN: a fused qkv weight between two unrelated tensors.
    before = _float_tensor("visual.patch_embed.proj.weight", torch.randn(8, 4))
    qkv = _float_tensor("visual.blocks.0.attn.qkv.weight", torch.randn(6, 4))
    after = _float_tensor("visual.blocks.0.norm1.weight", torch.randn(4))

    # WHEN: splitting the qkv weight.
    result = Split(_QKV_PATTERN, _QKV_TARGETS).apply([before, qkv, after], _CONFIG)

    # THEN: the parts take the source's position and the other tensors pass through.
    assert [tensor.hf_name for tensor in result] == [
        "visual.patch_embed.proj.weight",
        "visual.blocks.0.attn.q.weight",
        "visual.blocks.0.attn.k.weight",
        "visual.blocks.0.attn.v.weight",
        "visual.blocks.0.norm1.weight",
    ]
    assert result[0] is before and result[-1] is after


def test_split_without_matches_is_noop() -> None:
    # GIVEN: tensors none of which match the split pattern.
    tensors = [_float_tensor("model.layers.0.self_attn.q_proj.weight", torch.randn(4, 4))]

    # WHEN: applying the split.
    result = Split(_QKV_PATTERN, _QKV_TARGETS).apply(tensors, _CONFIG)

    # THEN: the same tensors are returned.
    assert len(result) == 1 and result[0] is tensors[0]


def test_split_squeezes_conv3d_patch_embedding_along_time() -> None:
    # GIVEN: a Conv3D patch embedding [C, 3, kt=2, P, P].
    data = torch.randn(8, 3, 2, 4, 4)
    tensor = _float_tensor("visual.patch_embed.proj.weight", data)
    rule = Split(
        r"visual\.patch_embed\.proj\.weight",
        ("patch.0", "patch.1"),
        axis=2,
        squeeze=True,
    )

    # WHEN: splitting it along the time axis.
    result = rule.apply([tensor], _CONFIG)

    # THEN: each part is one time slice as a Conv2D kernel [C, 3, P, P].
    assert [part.hf_name for part in result] == ["patch.0", "patch.1"]
    for index, part in enumerate(result):
        assert part.float_data is not None
        assert part.float_data.is_contiguous()
        torch.testing.assert_close(part.float_data, data[:, :, index])
        assert (part.rows, part.cols) == (8, 3)


def test_split_takes_unequal_sizes_from_config() -> None:
    # GIVEN: a fused GQA projection whose q part is wider than k and v.
    data = torch.randn(128, 8)
    tensor = _float_tensor("visual.blocks.0.attn.qkv.weight", data)
    rule = Split(
        _QKV_PATTERN,
        _QKV_TARGETS,
        sizes=_gqa_sizes,
    )

    # WHEN: splitting it with sizes read from the config.
    result = rule.apply([tensor], _CONFIG)

    # THEN: the parts have the configured sizes and cover the tensor in order.
    assert [part.rows for part in result] == [64, 32, 32]
    torch.testing.assert_close(torch.cat([_float_data(part) for part in result]), data)


def test_split_reads_sizes_only_when_a_tensor_matches() -> None:
    # GIVEN: a split whose sizes read a vision config, and a text-only config
    # without one.
    def vision_sizes(config: Any) -> tuple[int, int, int]:
        return _gqa_sizes(config.vision_config)

    rule = Split(_QKV_PATTERN, _QKV_TARGETS, sizes=vision_sizes)
    tensor = _float_tensor("model.layers.0.self_attn.q_proj.weight", torch.randn(4, 4))

    # WHEN: applying it to tensors none of which match.
    result = rule.apply([tensor], SimpleNamespace())

    # THEN: sizes is never called and the tensors pass through.
    assert len(result) == 1 and result[0] is tensor


def test_split_negative_axis_counts_from_the_end() -> None:
    # GIVEN: a float tensor split along its last axis.
    data = torch.randn(4, 6)
    rule = Split(r"w", ("a", "b"), axis=-1)

    # WHEN: applying the split.
    result = rule.apply([_float_tensor("w", data)], _CONFIG)

    # THEN: the columns are divided between the parts.
    assert [part.cols for part in result] == [3, 3]


@pytest.mark.parametrize("quant_format", [GGUF_Q4_0, GGUF_Q4_1], ids=["Q4_0", "Q4_1"])
def test_split_quantized_rows_pack_to_matching_rows_of_source_bytes(
    quant_format: GgufQuantFormat,
) -> None:
    # GIVEN: a quantized fused qkv weight, 3 x 32 rows of 64 columns.
    source = _quantized_tensor("visual.blocks.0.attn.qkv.weight", 96, 64, quant_format)

    # WHEN: splitting it and packing the source and the parts.
    parts = Split(_QKV_PATTERN, _QKV_TARGETS).apply([source], _CONFIG)
    packed_source = stage_pack_gguf_blocks(([replace(source, gguf_name="qkv")],), [], {})
    packed_parts = stage_pack_gguf_blocks(
        ([replace(part, gguf_name=part.hf_name) for part in parts],), [], {}
    )

    # THEN: every part keeps the format and column count, with a third of the rows.
    for part in parts:
        assert part.kind == "quantized"
        assert part.quant_format is quant_format
        assert (part.rows, part.cols) == (32, 64)
        assert (part.offsets is None) == quant_format.symmetric

    # THEN: each part's packed bytes are exactly the matching rows of the source's.
    source_bytes, _ = packed_source["quantized"]["qkv"]
    for index, part in enumerate(parts):
        part_bytes, part_format = packed_parts["quantized"][part.hf_name]
        assert part_format is quant_format
        torch.testing.assert_close(part_bytes, source_bytes[32 * index : 32 * (index + 1)])


def _column_major(data: torch.Tensor) -> torch.Tensor:
    """The same 2-D values with column-major strides, so the last stride is not 1."""
    return data.t().clone(memory_format=torch.contiguous_format).t()


@pytest.mark.parametrize(
    "quant_format", [GGUF_Q4_0, GGUF_Q4_1, GGUF_Q8_0], ids=["Q4_0", "Q4_1", "Q8_0"]
)
def test_split_quantized_single_block_parts_of_strided_scales_pack(
    quant_format: GgufQuantFormat,
) -> None:
    # GIVEN: a quantized tensor with one block per row whose scales and offsets
    # are column-major, as a custom rule could leave them.
    source = _quantized_tensor("w", 2, quant_format.block_size, quant_format)
    assert source.scales is not None
    source = replace(
        source,
        scales=_column_major(source.scales),
        offsets=None if source.offsets is None else _column_major(source.offsets),
    )
    assert source.scales is not None and source.scales.stride() == (1, 2)

    # WHEN: splitting it into single-block parts and packing the source and the parts.
    parts = Split(r"w", ("a", "b")).apply([source], _CONFIG)
    packed_source = stage_pack_gguf_blocks(([replace(source, gguf_name="w")],), [], {})
    packed_parts = stage_pack_gguf_blocks(
        ([replace(part, gguf_name=part.hf_name) for part in parts],), [], {}
    )

    # THEN: each part packs to the matching block of the source's bytes.
    source_bytes, _ = packed_source["quantized"]["w"]
    for index, part in enumerate(parts):
        part_bytes, _ = packed_parts["quantized"][part.hf_name]
        torch.testing.assert_close(part_bytes, source_bytes[index : index + 1])


def test_split_float_single_element_parts_of_strided_source_are_row_major() -> None:
    # GIVEN: a strided float bias view.
    bias = torch.randn(2, 2)[:, 0]
    assert bias.stride() == (2,)

    # WHEN: splitting it into single-element parts.
    parts = Split(r"b", ("b0", "b1")).apply([_float_tensor("b", bias)], _CONFIG)

    # THEN: every part has a unit stride, which the raw-byte (BF16) write needs.
    for index, part in enumerate(parts):
        assert _float_data(part).stride() == (1,)
        torch.testing.assert_close(_float_data(part), bias[index : index + 1])


@pytest.mark.parametrize("kind", ["float", "quantized"])
def test_split_inverts_fuse(kind: str) -> None:
    # GIVEN: per-layer q, k and v projections of different heights.
    def make(name: str, rows: int) -> ExtractedTensor:
        if kind == "quantized":
            return _quantized_tensor(name, rows, 64)
        return _float_tensor(name, torch.randn(rows, 64))

    originals = [
        make("blk.0.q.weight", 64),
        make("blk.0.k.weight", 32),
        make("blk.0.v.weight", 32),
    ]
    fuse = Fuse(
        sources=(r"blk\.(\d+)\.q\.weight", r"blk\.(\d+)\.k\.weight", r"blk\.(\d+)\.v\.weight"),
        target_name=r"blk.\1.qkv.weight",
    )
    split = Split(
        r"blk\.(\d+)\.qkv\.weight",
        (r"blk.\1.q.weight", r"blk.\1.k.weight", r"blk.\1.v.weight"),
        sizes=lambda config: (64, 32, 32),
    )

    # WHEN: fusing them and splitting the result back.
    result = split.apply(fuse.apply(originals, _CONFIG), _CONFIG)

    # THEN: the original tensors come back unchanged.
    assert [tensor.hf_name for tensor in result] == [tensor.hf_name for tensor in originals]
    for original, restored in zip(originals, result):
        assert (restored.rows, restored.cols) == (original.rows, original.cols)
        for field in ("float_data", "int_codes", "scales"):
            expected, actual = getattr(original, field), getattr(restored, field)
            if expected is None:
                assert actual is None
            else:
                torch.testing.assert_close(actual, expected)


def test_split_claims_matching_parameters() -> None:
    # GIVEN: an adapter whose only rule is a split.
    adapter = ArchAdapter(
        gguf_arch="test",
        tensor_rules=(Split(_QKV_PATTERN, _QKV_TARGETS),),
        write_metadata=lambda writer, config: None,
        tokenizer_model="gpt2",
        tokenizer_pre=None,
    )

    # THEN: the split claims the parameters it matches, so extraction reads them.
    assert _is_claimed("visual.blocks.3.attn.qkv.weight", adapter)
    assert _is_claimed("visual.blocks.3.attn.qkv.bias", adapter)
    assert not _is_claimed("visual.blocks.3.attn.proj.weight", adapter)


def test_split_requires_at_least_two_targets() -> None:
    # WHEN / THEN: building a split with a single target raises.
    with pytest.raises(ExportError, match="at least two targets"):
        Split(r"w", ("a",))


@pytest.mark.parametrize(
    ("tensor", "rule", "match"),
    [
        pytest.param(
            _float_tensor("w", torch.randn(5, 4)),
            Split(r"w", ("a", "b")),
            "not divisible by 2",
            id="indivisible",
        ),
        pytest.param(
            _float_tensor("w", torch.randn(6, 4)),
            Split(r"w", ("a", "b"), sizes=lambda config: (2, 2, 2)),
            "got 3 sizes for 2 targets",
            id="sizes-count",
        ),
        pytest.param(
            _float_tensor("w", torch.randn(6, 4)),
            Split(r"w", ("a", "b"), sizes=lambda config: (2, 3)),
            "add up to",
            id="sizes-sum",
        ),
        pytest.param(
            _float_tensor("w", torch.randn(6, 4)),
            Split(r"w", ("a", "b"), sizes=lambda config: (6, 0)),
            "must be positive",
            id="sizes-zero",
        ),
        pytest.param(
            _float_tensor("w", torch.randn(6, 4)),
            Split(r"w", ("a", "b"), axis=2),
            "along axis 2",
            id="axis-out-of-range",
        ),
        pytest.param(
            _float_tensor("w", torch.randn(4, 4)),
            Split(r"w", ("a", "b"), squeeze=True),
            "with squeeze",
            id="squeeze-size-not-1",
        ),
        pytest.param(
            _float_tensor("w", torch.randn(2)),
            Split(r"w", ("a", "b"), squeeze=True),
            "keep at least one dimension",
            id="squeeze-to-scalar",
        ),
        pytest.param(
            _quantized_tensor("w", 4, 64),
            Split(r"w", ("a", "b"), axis=1),
            "only be split along axis 0",
            id="quantized-axis-1",
        ),
        pytest.param(
            _quantized_tensor("w", 2, 64),
            Split(r"w", ("a", "b"), squeeze=True),
            "with squeeze",
            id="quantized-squeeze",
        ),
    ],
)
def test_split_rejects_invalid_split(tensor: ExtractedTensor, rule: Split, match: str) -> None:
    # WHEN / THEN: applying a split the tensor cannot satisfy raises.
    with pytest.raises(ExportError, match=match):
        rule.apply([tensor], _CONFIG)
