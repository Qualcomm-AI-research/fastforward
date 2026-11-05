# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Tests for the GGUF tensor rules and :func:`apply_tensor_rules`."""

import logging

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from fastforward.exceptions import ExportError
from fastforward.export.stages.gguf import (
    DROP_TIED_LM_HEAD,
    Drop,
    ExtractedTensor,
    Fuse,
    Rename,
    Synthesize,
    Transform,
)
from fastforward.export.stages.gguf.rules import apply_tensor_rules
from tests.export.stages.gguf.conftest import make_quantized_tensor

_CONFIG: Any = SimpleNamespace(hidden_size=4, tie_word_embeddings=False)


def _float_tensor(hf_name: str, gguf_name: str = "") -> ExtractedTensor:
    return ExtractedTensor(
        hf_name=hf_name, kind="float", rows=2, float_data=torch.zeros(2), gguf_name=gguf_name
    )


def _upper_name_map(hf_name: str) -> str | None:
    return hf_name.upper() if hf_name.startswith("keep") else None


def test_rename_sets_gguf_names_and_keeps_unmapped_names() -> None:
    # GIVEN: a mapped tensor and an unmapped one that an earlier rule already named.
    mapped = _float_tensor("keep.weight")
    unmapped = _float_tensor("other.weight", gguf_name="earlier.weight")

    # WHEN: applying the rename.
    result = Rename(_upper_name_map).apply([mapped, unmapped], _CONFIG)

    # THEN: the mapped tensor is named; the unmapped one keeps its earlier name.
    assert [t.gguf_name for t in result] == ["KEEP.WEIGHT", "earlier.weight"]
    assert mapped.gguf_name == ""


def test_rename_claims_mapped_parameters_only() -> None:
    # GIVEN: a rename rule.
    rule = Rename(_upper_name_map)

    # THEN: it claims exactly the parameters its name map maps.
    assert rule.claims("keep.weight")
    assert not rule.claims("other.weight")


def test_transform_applies_only_to_matching_tensors() -> None:
    # GIVEN: a transform that adds one to tensors matching "a.*".
    def add_one(tensor: ExtractedTensor, config: Any) -> ExtractedTensor:
        del config
        assert tensor.float_data is not None
        return replace(tensor, float_data=tensor.float_data + 1)

    rule = Transform(r"a\..*", add_one)
    tensors = [_float_tensor("a.weight"), _float_tensor("b.weight")]

    # WHEN: applying it.
    result = rule.apply(tensors, _CONFIG)

    # THEN: only the matching tensor is rewritten; the rule claims nothing.
    torch.testing.assert_close(result[0].float_data, torch.ones(2))
    torch.testing.assert_close(result[1].float_data, torch.zeros(2))
    assert not rule.claims("a.weight")


def test_drop_removes_full_matches_only() -> None:
    # GIVEN: a drop rule for "x.weight" and tensors with similar names.
    tensors = [_float_tensor("x.weight"), _float_tensor("x.weight.extra")]

    # WHEN: applying it.
    result = Drop(r"x\.weight").apply(tensors, _CONFIG)

    # THEN: only the full match is removed.
    assert [t.hf_name for t in result] == ["x.weight.extra"]


@pytest.mark.parametrize(("tied", "expected"), [(True, []), (False, ["lm_head.weight"])])
def test_drop_tied_lm_head_follows_config(tied: bool, expected: list[str]) -> None:
    # GIVEN: an lm_head tensor and a config with or without tied embeddings.
    config: Any = SimpleNamespace(tie_word_embeddings=tied)

    # WHEN: applying the built-in tied-head drop.
    result = DROP_TIED_LM_HEAD.apply([_float_tensor("lm_head.weight")], config)

    # THEN: the head is dropped only when the embeddings are tied.
    assert [t.hf_name for t in result] == expected


def test_drop_tied_lm_head_without_tie_attribute_keeps_head() -> None:
    # GIVEN: a config without a tie_word_embeddings attribute.
    config: Any = SimpleNamespace()

    # WHEN: applying the built-in tied-head drop.
    result = DROP_TIED_LM_HEAD.apply([_float_tensor("lm_head.weight")], config)

    # THEN: the head is kept.
    assert len(result) == 1


def test_synthesize_appends_named_float_tensors() -> None:
    # GIVEN: a synthesize rule deriving a bf16 tensor from the config.
    rule = Synthesize(
        lambda config: {"rope_freqs.weight": torch.ones(config.hidden_size, dtype=torch.bfloat16)}
    )
    existing = _float_tensor("a.weight", gguf_name="a.weight")

    # WHEN: applying it.
    result = rule.apply([existing], _CONFIG)

    # THEN: the synthesized tensor follows the existing ones as a named float
    # tensor, keeping its dtype until the write stage casts it.
    assert [t.gguf_name for t in result] == ["a.weight", "rope_freqs.weight"]
    synthesized = result[1]
    assert synthesized.kind == "float"
    assert synthesized.rows == 4
    assert synthesized.float_data is not None
    assert synthesized.float_data.dtype == torch.bfloat16
    torch.testing.assert_close(synthesized.float_data, torch.ones(4, dtype=torch.bfloat16))


@pytest.mark.parametrize(
    "data",
    [torch.ones(4, dtype=torch.int32), torch.tensor(1.0)],
    ids=["integer", "zero-dim"],
)
def test_synthesize_rejects_invalid_tensor(data: torch.Tensor) -> None:
    # GIVEN: a synthesize rule returning a tensor that is not a >=1-D float tensor.
    rule = Synthesize(lambda config: {"bad.weight": data})

    # WHEN/THEN: applying it raises.
    with pytest.raises(ExportError, match="float tensor"):
        rule.apply([], _CONFIG)


def test_apply_tensor_rules_runs_rules_in_order() -> None:
    # GIVEN: a fusion of q/k followed by a rename of the fused tensor only.
    tensors = [
        make_quantized_tensor("layer.0.q.weight", rows=4, cols=32),
        make_quantized_tensor("layer.0.k.weight", rows=4, cols=32),
    ]
    rules = (
        Fuse(
            sources=(r"layer\.(\d+)\.q\.weight", r"layer\.(\d+)\.k\.weight"), target_name=r"l.\1.qk"
        ),
        Rename(lambda hf_name: "blk.0.attn_qk.weight" if hf_name == "l.0.qk" else None),
    )

    # WHEN: applying the rules.
    (result,) = apply_tensor_rules(tensors, rules, _CONFIG)

    # THEN: the rename saw the fusion's output.
    assert result.gguf_name == "blk.0.attn_qk.weight"
    assert result.rows == 8


def test_apply_tensor_rules_drops_unnamed_tensors_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # GIVEN: a named and an unnamed tensor.
    tensors = [_float_tensor("keep.weight"), _float_tensor("other.weight")]

    # WHEN: applying a rename that only names one of them.
    with caplog.at_level(logging.WARNING, logger="fastforward.export.stages.gguf.rules"):
        result = apply_tensor_rules(tensors, (Rename(_upper_name_map),), _CONFIG)

    # THEN: the unnamed tensor is dropped and reported.
    assert [t.gguf_name for t in result] == ["KEEP.WEIGHT"]
    (record,) = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert "other.weight" in record.getMessage()


def test_apply_tensor_rules_rejects_duplicate_gguf_names() -> None:
    # GIVEN: rules that give two tensors the same GGUF name.
    tensors = [_float_tensor("keep.a")]
    rules = (
        Rename(lambda hf_name: "blk.0.attn_q.weight"),
        Synthesize(lambda config: {"blk.0.attn_q.weight": torch.ones(4)}),
    )

    # WHEN/THEN: applying them raises.
    with pytest.raises(ExportError, match="duplicate GGUF tensor names: blk.0.attn_q.weight"):
        apply_tensor_rules(tensors, rules, _CONFIG)
