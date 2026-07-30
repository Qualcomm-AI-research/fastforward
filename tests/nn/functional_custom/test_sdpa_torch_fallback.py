# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Tests for the `sdpa_torch_fallback` feature.

When the `sdpa_torch_fallback` flag is enabled and the caller
passes no active quantizer, `fastforward.nn.functional.scaled_dot_product_attention`
should short-circuit to `torch.nn.functional.scaled_dot_product_attention` for speed.
Any active quantizer (i.e. a `Quantizer` that is not `None` and not a
`QuantizerStub`) must disable the shortcut and route through the normal
fastforward dispatch path.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import fastforward
import pytest
import torch

from fastforward.exceptions import QuantizationError
from fastforward.flags import (
    get_sdpa_torch_fallback_allowed,
    sdpa_torch_fallback_allowed,
    set_sdpa_torch_fallback_allowed,
)
from fastforward.nn import QuantizerStub
from fastforward.nn import functional as FF
from fastforward.nn.quantizer import Quantizer
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from ._utils import _make_attn_inputs

torch.backends.cudnn.deterministic = True


# ------------------------------------------------------------------------------
# FIXTURES
# ------------------------------------------------------------------------------


@pytest.fixture
def spy_fallback(monkeypatch: pytest.MonkeyPatch) -> Mock:
    spy = Mock(wraps=F.scaled_dot_product_attention)
    monkeypatch.setattr(F, "scaled_dot_product_attention", spy)
    return spy


# ------------------------------------------------------------------------------
# FLAG TESTS
# ------------------------------------------------------------------------------


def test_fallback_flag_defaults_to_false() -> None:
    # GIVEN an unmodified process state
    # WHEN the flag is read
    # THEN it is False
    assert not get_sdpa_torch_fallback_allowed()


def test_fallback_flag_setter_and_context_manager() -> None:
    # GIVEN the flag is at its default value (False)
    assert not get_sdpa_torch_fallback_allowed()

    # WHEN/THEN nested contexts and setters behave like the other flags
    with sdpa_torch_fallback_allowed(True):
        assert get_sdpa_torch_fallback_allowed()
        with sdpa_torch_fallback_allowed(False):
            assert not get_sdpa_torch_fallback_allowed()
            set_sdpa_torch_fallback_allowed(True)
            assert get_sdpa_torch_fallback_allowed()
            set_sdpa_torch_fallback_allowed(False)
            assert not get_sdpa_torch_fallback_allowed()
            with sdpa_torch_fallback_allowed(True):
                assert get_sdpa_torch_fallback_allowed()
            assert not get_sdpa_torch_fallback_allowed()
        assert get_sdpa_torch_fallback_allowed()
    assert not get_sdpa_torch_fallback_allowed()


def test_fallback_flag_exposed_on_fastforward_namespace() -> None:
    # GIVEN fastforward re-exports flag helpers at the package level
    # WHEN reading via the package namespace
    # THEN they resolve to the same objects as `fastforward.flags`
    assert fastforward.get_sdpa_torch_fallback_allowed is get_sdpa_torch_fallback_allowed
    assert fastforward.set_sdpa_torch_fallback_allowed is set_sdpa_torch_fallback_allowed
    assert fastforward.sdpa_torch_fallback_allowed is sdpa_torch_fallback_allowed


# ------------------------------------------------------------------------------
# BRANCHING TESTS
# ------------------------------------------------------------------------------


def test_fallback_not_called_when_flag_is_false(spy_fallback: Mock) -> None:
    # GIVEN the fallback flag defaults to False
    q, k, v, _, _ = _make_attn_inputs("self-attn", 1, False, "cpu")

    # WHEN SDPA is called with no quantizers and the flag disabled
    with sdpa_torch_fallback_allowed(False), torch.no_grad():
        FF.scaled_dot_product_attention(q, k, v, strict_quantization=False)

    # THEN the fastforward dispatch path runs, not the torch fallback
    assert not spy_fallback.called


def test_fallback_called_when_flag_true_and_no_quantizer_args(
    spy_fallback: Mock,
) -> None:
    # GIVEN the fallback flag is enabled and no quantizer args are provided
    q, k, v, _, _ = _make_attn_inputs("self-attn", 1, False, "cpu")

    # WHEN SDPA is called
    with sdpa_torch_fallback_allowed(True), torch.no_grad():
        FF.scaled_dot_product_attention(q, k, v, strict_quantization=False)

    # THEN the torch fallback runs exactly once
    assert spy_fallback.called
    assert spy_fallback.call_count == 1


def test_fallback_called_when_all_quantizers_are_none(spy_fallback: Mock) -> None:
    # GIVEN the fallback flag is enabled and every quantizer arg is None
    q, k, v, _, _ = _make_attn_inputs("self-attn", 1, False, "cpu")

    # WHEN SDPA is called with all quantizer arguments explicitly set to None
    with sdpa_torch_fallback_allowed(True), torch.no_grad():
        FF.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_scores_quantizer=None,
            attn_mask_quantizer=None,
            masked_scores_quantizer=None,
            attn_weights_quantizer=None,
            scaled_query_quantizer=None,
            scaled_key_quantizer=None,
            dropout_quantizer=None,
            output_quantizer=None,
            strict_quantization=False,
        )

    # THEN the torch fallback runs (None is treated as inactive)
    assert spy_fallback.called


def test_fallback_called_when_all_quantizers_are_stubs(spy_fallback: Mock) -> None:
    # GIVEN the fallback flag is enabled and every quantizer arg is a stub
    q, k, v, _, _ = _make_attn_inputs("self-attn", 1, False, "cpu")
    stub = QuantizerStub()

    # WHEN SDPA is called with only stub quantizers
    with sdpa_torch_fallback_allowed(True), torch.no_grad():
        FF.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_scores_quantizer=stub,
            attn_mask_quantizer=stub,
            masked_scores_quantizer=stub,
            attn_weights_quantizer=stub,
            scaled_query_quantizer=stub,
            scaled_key_quantizer=stub,
            dropout_quantizer=stub,
            output_quantizer=stub,
            strict_quantization=False,
        )

    # THEN the torch fallback runs (QuantizerStub is treated as inactive)
    assert spy_fallback.called


@pytest.mark.parametrize(
    "quantizer_kwarg",
    [
        "attn_scores_quantizer",
        "attn_mask_quantizer",
        "masked_scores_quantizer",
        "attn_weights_quantizer",
        "scaled_query_quantizer",
        "scaled_key_quantizer",
        "dropout_quantizer",
        "output_quantizer",
    ],
)
def test_fallback_not_called_when_any_named_quantizer_is_active(
    spy_fallback: Mock, quantizer_kwarg: str
) -> None:
    # GIVEN  the fallback flag is enabled and one named quantizer is active
    q, k, v, _, _ = _make_attn_inputs("self-attn", 1, False, "cpu")
    active = _ActiveQuantizer()
    kwargs: dict[str, Any] = {quantizer_kwarg: active}

    # WHEN SDPA is called with a single active quantizer
    with sdpa_torch_fallback_allowed(True), torch.no_grad():
        FF.scaled_dot_product_attention(q, k, v, strict_quantization=False, **kwargs)

    # THEN the torch fallback is skipped (dispatch handles the active quantizer)
    assert not spy_fallback.called


def test_fallback_not_called_when_extra_kwarg_quantizer_is_active(
    spy_fallback: Mock,
) -> None:
    # GIVEN an active Quantizer supplied only through **kwargs
    q, k, v, _, _ = _make_attn_inputs("self-attn", 1, False, "cpu")
    active = _ActiveQuantizer()

    # WHEN SDPA is called with the extra quantizer in kwargs
    with sdpa_torch_fallback_allowed(True), torch.no_grad():
        try:
            FF.scaled_dot_product_attention(
                q, k, v, strict_quantization=False, extra_quantizer=active
            )
        except TypeError:
            pass  # extra_quantizer does not exists and the fasforward sdpa call will fail

    # THEN the fallback is skipped — kwargs quantizers must count as "active"
    assert not spy_fallback.called


def test_fallback_called_when_kwargs_have_only_stub(spy_fallback: Mock) -> None:
    # GIVEN a stub Quantizer supplied only through **kwargs
    q, k, v, _, _ = _make_attn_inputs("self-attn", 1, False, "cpu")

    # WHEN SDPA is called with the stub in kwargs
    with sdpa_torch_fallback_allowed(True), torch.no_grad():
        try:
            FF.scaled_dot_product_attention(
                q, k, v, strict_quantization=False, extra_quantizer=QuantizerStub()
            )
        except TypeError:
            pass  # extra_quantizer does not exists and the fasforward sdpa call would fail

    # THEN the fallback still runs (stubs are inactive)
    assert spy_fallback.called


# ------------------------------------------------------------------------------
# PER-CALL `sdpa_torch_fallback` PARAMETER
# ------------------------------------------------------------------------------


def test_fallback_param_true_overrides_global_false(spy_fallback: Mock) -> None:
    # GIVEN the global flag is False
    q, k, v, _, _ = _make_attn_inputs("self-attn", 1, False, "cpu")
    assert not get_sdpa_torch_fallback_allowed()

    # WHEN the call passes `sdpa_torch_fallback=True` explicitly
    with sdpa_torch_fallback_allowed(False), torch.no_grad():
        FF.scaled_dot_product_attention(
            q,
            k,
            v,
            strict_quantization=False,
            sdpa_torch_fallback=True,
        )

    # THEN the per-call override wins and the fallback fires
    assert spy_fallback.called


def test_fallback_param_false_overrides_global_true(spy_fallback: Mock) -> None:
    # GIVEN the global flag is enabled
    q, k, v, _, _ = _make_attn_inputs("self-attn", 1, False, "cpu")

    # WHEN the call passes `sdpa_torch_fallback=False` explicitly
    with sdpa_torch_fallback_allowed(True), torch.no_grad():
        FF.scaled_dot_product_attention(
            q,
            k,
            v,
            strict_quantization=False,
            sdpa_torch_fallback=False,
        )

    # THEN the per-call override wins and the fallback does not fire
    assert not spy_fallback.called


def test_fallback_param_none_reads_global_flag_true(spy_fallback: Mock) -> None:
    # GIVEN the global flag is enabled and the call passes None (the default)
    set_sdpa_torch_fallback_allowed(True)
    q, k, v, _, _ = _make_attn_inputs("self-attn", 1, False, "cpu")

    # WHEN SDPA is invoked without overriding the param
    with torch.no_grad():
        FF.scaled_dot_product_attention(
            q,
            k,
            v,
            strict_quantization=False,
            sdpa_torch_fallback=None,
        )

    # THEN the global flag is honored and the fallback fires
    assert spy_fallback.called


def test_fallback_param_none_reads_global_flag_false(spy_fallback: Mock) -> None:
    # GIVEN the global flag is disabled and the call passes None (the default)
    set_sdpa_torch_fallback_allowed(False)
    q, k, v, _, _ = _make_attn_inputs("self-attn", 1, False, "cpu")
    assert not get_sdpa_torch_fallback_allowed()

    # WHEN SDPA is invoked without overriding the param
    with torch.no_grad():
        FF.scaled_dot_product_attention(
            q,
            k,
            v,
            strict_quantization=False,
            sdpa_torch_fallback=None,
        )

    # THEN the global flag is honored and the fallback does not fire
    assert not spy_fallback.called


# ------------------------------------------------------------------------------
# BIT-EXACT PARITY WITH torch.nn.functional.scaled_dot_product_attention
# ------------------------------------------------------------------------------

_ATTN_MASK_OPTS = [False, "float", "bool", "causal"]
_GQA_VALUES = [1, 4] if torch.__version__ >= "2.5" else [1]


@pytest.mark.skipif(torch.__version__ < "2.5", reason="requires torch>=2.5 for enable_gqa")
@pytest.mark.parametrize("use_attn_mask", _ATTN_MASK_OPTS, ids=lambda m: f"attn_mask={m}")
@pytest.mark.parametrize("groups", _GQA_VALUES, ids=lambda g: f"gqa={g}" if g > 1 else "no-gqa")
@pytest.mark.parametrize("scale", [None, 0.1], ids=lambda s: f"scale={s}")
def test_fallback_output_matches_torch_sdpa_bit_exact(
    use_attn_mask: bool | str, groups: int, scale: float | None
) -> None:
    # GIVEN torch scaled_dot_product_attention output for a given input
    q, k, v, attn_mask, is_causal = _make_attn_inputs("self-attn", groups, use_attn_mask, "cpu")

    with sdpa_kernel(backends=[SDPBackend.MATH]), torch.no_grad():
        out_torch = F.scaled_dot_product_attention(
            q, k, v, attn_mask, is_causal=is_causal, scale=scale, enable_gqa=groups > 1
        )

    # WHEN fastforward SDPA is called with the fallback enabled and no quantizers
    with sdpa_torch_fallback_allowed(True), torch.no_grad():
        out_ff = FF.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=groups > 1,
            strict_quantization=False,
        )

    # THEN the fastforward output equals torch's output bit-exactly
    assert out_ff is not None
    assert torch.equal(out_ff, out_torch)


# @pytest.mark.skipif(torch.__version__ >= "2.5", reason="test for torch torch<2.5 (no gqa support)")
@pytest.mark.parametrize("use_attn_mask", _ATTN_MASK_OPTS, ids=lambda m: f"attn_mask={m}")
@pytest.mark.parametrize("scale", [None, 0.1], ids=lambda s: f"scale={s}")
def test_fallback_output_matches_torch_sdpa_bit_exact__no_gqa(
    use_attn_mask: bool | str, scale: float | None
) -> None:
    # GIVEN torch scaled_dot_product_attention output for a given input
    q, k, v, attn_mask, is_causal = _make_attn_inputs("self-attn", 1, use_attn_mask, "cpu")

    with sdpa_kernel(backends=[SDPBackend.MATH]), torch.no_grad():
        out_torch = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask,
            is_causal=is_causal,
            scale=scale,
        )

    # WHEN fastforward SDPA is called with the fallback enabled and no quantizers
    with sdpa_torch_fallback_allowed(True), torch.no_grad():
        out_ff = FF.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            scale=scale,
            strict_quantization=False,
        )

    # THEN the fastforward output equals torch's output bit-exactly
    assert out_ff is not None
    assert torch.equal(out_ff, out_torch)


# ------------------------------------------------------------------------------
# PARAMETER FORWARDING TESTS
# ------------------------------------------------------------------------------


def test_fallback_forwards_all_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    # GIVEN a spy that records the kwargs passed to torch scaled_dot_product_attention
    captured: dict[str, Any] = {}

    def _capture(*_: Any, **kwargs: Any) -> torch.Tensor:
        captured.update(kwargs)
        return torch.zeros(1, 2, 3, 4)

    monkeypatch.setattr(torch, "scaled_dot_product_attention", _capture, raising=False)
    monkeypatch.setattr(F, "scaled_dot_product_attention", _capture)
    q, k, v, _, _ = _make_attn_inputs("self-attn", 1, False, "cpu")
    mask = torch.rand(1, 1, 4, 4) >= 0.0

    # WHEN SDPA is invoked with the fallback enabled and several non-default args
    with sdpa_torch_fallback_allowed(True), torch.no_grad():
        FF.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.25,
            is_causal=False,
            scale=0.5,
            enable_gqa=False,
            strict_quantization=False,
        )

    # THEN every SDPA argument reached torch with the correct value
    assert captured["query"] is q
    assert captured["key"] is k
    assert captured["value"] is v
    assert captured["attn_mask"] is mask
    assert captured["dropout_p"] == 0.25
    assert captured["is_causal"] is False
    assert captured["scale"] == 0.5
    # `enable_gqa` is only forwarded on torch>=2.5, so only assert when present
    if "enable_gqa" in captured:
        assert captured["enable_gqa"] is False


def test_fallback_forwards_is_causal(monkeypatch: pytest.MonkeyPatch) -> None:
    # GIVEN a spy that records the kwargs passed to torch scaled_dot_product_attention
    captured: dict[str, Any] = {}

    def _capture(*_: Any, **kwargs: Any) -> torch.Tensor:
        captured.update(kwargs)
        return torch.zeros_like(kwargs["query"])

    monkeypatch.setattr(torch, "scaled_dot_product_attention", _capture, raising=False)
    monkeypatch.setattr(F, "scaled_dot_product_attention", _capture)
    q, k, v, _, _ = _make_attn_inputs("self-attn", 1, False, "cpu")

    # WHEN SDPA is invoked with is_causal=True
    with sdpa_torch_fallback_allowed(True), torch.no_grad():
        FF.scaled_dot_product_attention(q, k, v, is_causal=True, strict_quantization=False)

    # THEN torch receives is_causal=True and no attn_mask
    assert captured["is_causal"] is True
    assert captured["attn_mask"] is None


# ------------------------------------------------------------------------------
# INTERACTION WITH strict_quantization
# ------------------------------------------------------------------------------


def test_fallback_blocked_when_strict_quantization_true_kwarg(
    spy_fallback: Mock,
) -> None:
    # GIVEN the fallback flag is enabled but the call requests strict quantization
    q, k, v, _, _ = _make_attn_inputs("self-attn", 1, False, "cpu")

    # WHEN SDPA is invoked with explicit `strict_quantization=True`
    #       (raises because dispatch is entered without an output_quantizer —
    #       which is precisely what proves the fallback path was skipped)
    with sdpa_torch_fallback_allowed(True), torch.no_grad():
        with pytest.raises(QuantizationError):
            FF.scaled_dot_product_attention(q, k, v, strict_quantization=True)

    # THEN strict quantization takes precedence — the fallback is skipped
    assert not spy_fallback.called


def test_fallback_blocked_when_strict_quantization_true_context(
    spy_fallback: Mock,
) -> None:
    # GIVEN `strict_quantization(True)` is set globally via context manager
    q, k, v, _, _ = _make_attn_inputs("self-attn", 1, False, "cpu")

    # WHEN the fallback flag is enabled at the same time
    with (
        fastforward.strict_quantization(True),
        sdpa_torch_fallback_allowed(True),
        torch.no_grad(),
    ):
        with pytest.raises(QuantizationError):
            FF.scaled_dot_product_attention(q, k, v)

    # THEN strict quantization blocks the shortcut
    assert not spy_fallback.called


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------


class _ActiveQuantizer(Quantizer):
    """A non-stub Quantizer that acts as identity.

    Used in tests that only need `_is_active` to return True; the forward is
    identity so no range initialization is required. Using an uninitialized
    `LinearQuantizer` here would crash inside the dispatch/math path even
    though the test's intent is to verify the fallback branch is *skipped*.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
