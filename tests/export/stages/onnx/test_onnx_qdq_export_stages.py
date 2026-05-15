# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Any

import pytest
import torch

from fastforward.exceptions import ExportError
from fastforward.export.stages.onnx.onnx_qdq_export_stages import (
    _QDQ_MIN_OPSET_VERSION,
    _resolve_qdq_lowerings,
    _with_ff_qdq_lowerings,
)


def test_with_ff_qdq_lowerings_injects_ff_lowerings_without_mutating_input() -> None:
    # GIVEN a user context for exporting
    user_table = {"sentinel_op": "sentinel_lowering"}
    user_options = {"opset_version": 21, "custom_translation_table": user_table}
    user_context = {"onnx_export_options": user_options, "model_name": "m"}

    # WHEN applying the QDQ lowerings and returning the new context
    new_context = _with_ff_qdq_lowerings(user_context)
    # THEN the new context is not the original context and the onnx epxort options do
    # not match the original (new translations have been added)
    assert new_context is not user_context
    assert new_context["onnx_export_options"] is not user_options
    assert new_context["onnx_export_options"]["custom_translation_table"] is not user_table
    # FF lowerings are present in the merged table; user entries survive.
    # The lowerings are produced by `_resolve_qdq_lowerings(opset_version)`
    # which is `lru_cache`-d, so identity comparison against a fresh resolve
    # call at the same opset is well-defined.
    expected_q, expected_dq = _resolve_qdq_lowerings(_QDQ_MIN_OPSET_VERSION)
    merged_table = new_context["onnx_export_options"]["custom_translation_table"]
    assert merged_table[torch.ops.fastforward.quantize_by_tile.default] is expected_q
    assert merged_table[torch.ops.fastforward.dequantize_by_tile.default] is expected_dq
    assert merged_table["sentinel_op"] == "sentinel_lowering"

    # THEN the user context remains unchanged
    assert user_context == {
        "onnx_export_options": {
            "opset_version": 21,
            "custom_translation_table": {"sentinel_op": "sentinel_lowering"},
        },
        "model_name": "m",
    }
    assert "custom_translation_table" in user_options
    assert user_table == {"sentinel_op": "sentinel_lowering"}
    assert new_context["model_name"] == "m"
    assert new_context["onnx_export_options"]["opset_version"] == 21


def test_with_ff_qdq_lowerings_preserves_user_override_for_ff_op() -> None:
    # GIVEN a user context with a custom Q lowering for the quantize_by_tile op.
    def _custom_q_lowering(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return None

    user_context = {
        "onnx_export_options": {
            "custom_translation_table": {
                torch.ops.fastforward.quantize_by_tile.default: _custom_q_lowering,
            },
        },
    }

    # WHEN applying the QDQ lowerings.
    new_context = _with_ff_qdq_lowerings(user_context)

    # THEN the user-supplied lowering for the overridden op wins, while the FF
    # default is still applied for the op the user did not override.
    table = new_context["onnx_export_options"]["custom_translation_table"]
    assert table[torch.ops.fastforward.quantize_by_tile.default] is _custom_q_lowering
    _, expected_dq = _resolve_qdq_lowerings(_QDQ_MIN_OPSET_VERSION)
    assert table[torch.ops.fastforward.dequantize_by_tile.default] is expected_dq


@pytest.mark.parametrize(
    "bad_options",
    ["not-a-dict", 42, ["list", "of", "things"]],
)
def test_with_ff_qdq_lowerings_raises_on_non_dict_options(bad_options: Any) -> None:
    # GIVEN a context where onnx_export_options is not a dict.
    # WHEN/THEN applying QDQ lowerings raises a TypeError.
    with pytest.raises(TypeError, match="`onnx_export_options` must be a dictionary"):
        _with_ff_qdq_lowerings({"onnx_export_options": bad_options})


def test_with_ff_qdq_lowerings_defaults_opset_version_when_unset() -> None:
    # GIVEN a context with no opset_version set.
    # WHEN applying QDQ lowerings.
    new_context = _with_ff_qdq_lowerings({})

    # THEN opset_version defaults to the QDQ minimum.
    assert new_context["onnx_export_options"]["opset_version"] == _QDQ_MIN_OPSET_VERSION


def test_with_ff_qdq_lowerings_preserves_user_supplied_opset_at_or_above_minimum() -> None:
    # GIVEN a context with an opset_version above the minimum.
    # WHEN applying QDQ lowerings.
    new_context = _with_ff_qdq_lowerings({
        "onnx_export_options": {"opset_version": _QDQ_MIN_OPSET_VERSION + 2}
    })

    # THEN the user-supplied opset_version is preserved unchanged.
    assert new_context["onnx_export_options"]["opset_version"] == _QDQ_MIN_OPSET_VERSION + 2


@pytest.mark.parametrize("bad_opset", [_QDQ_MIN_OPSET_VERSION - 1, 18, 17, 13])
def test_with_ff_qdq_lowerings_rejects_opset_below_minimum(bad_opset: int) -> None:
    # GIVEN a context with an opset_version below the QDQ minimum.
    # WHEN/THEN applying QDQ lowerings raises an ExportError.
    with pytest.raises(ExportError, match="opset_version >= 21"):
        _with_ff_qdq_lowerings({"onnx_export_options": {"opset_version": bad_opset}})
