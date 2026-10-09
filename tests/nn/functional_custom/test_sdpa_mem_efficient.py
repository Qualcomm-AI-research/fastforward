# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Tests for the memory-efficient `scaled_dot_product_attention` kernel.

The memory-efficient kernel splits the query into chunks of target positions and
attends one chunk at a time, so the intermediate attention tensors never grow
with the full target sequence length. Every chunk still holds complete attention
rows, hence the kernel must reproduce the result of the MATH kernel -- for
unquantized as well as for quantized inputs -- while keeping peak memory
proportional to the chunk size instead of the sequence length.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator
from unittest.mock import Mock

import fastforward
import pytest
import torch

from fastforward import estimate_ranges, range_setting
from fastforward.dispatcher import dispatch
from fastforward.exceptions import QuantizationError
from fastforward.nn import functional as FF
from fastforward.nn.functional.custom.sdpa import (
    scaled_dot_product_attention_math,
    scaled_dot_product_attention_mem_efficient,
    sdpa_mem_efficient,
)
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from ._utils import (
    _make_attn_inputs,
    _make_sdpa_input_tensors,
    _print_abs_max_err,
    _QuantizedSDPA,
    assert_max_abs_err,
)

torch.backends.cudnn.deterministic = True

_ATTN_MASK_OPTS = [False, "float", "bool", "causal"]

# Compare numerically: as strings "2.15" < "2.5", which would silently drop GQA.
_TORCH_VERSION = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
GROUP_ATTN_VALUES = [1, 4] if _TORCH_VERSION >= (2, 5) else [1]

# The shared input builders produce a target sequence length of 4 (self
# attention) and 8 (cross attention), so these chunk sizes split the query into
# several chunks, with the largest one leaving a partial final chunk.
_QUERY_CHUNK_SIZES = [1, 3]

# `_make_attn_inputs` builds inputs with 4 attention heads (before GQA
# repeats query heads further), so these split the heads into several groups,
# with the largest one leaving a partial final group. 100 is comfortably
# larger than any head count used below, so it represents "no head-chunking".
_HEAD_CHUNK_SIZES = [1, 3, 100]


# ------------------------------------------------------------------------------
# CONTEXT MANAGER TESTS
# ------------------------------------------------------------------------------


def _settings() -> tuple[bool, int | None, int | None, bool]:
    """Return the current (enabled, query_chunk_size, head_chunk_size, checkpoint) SDPA settings.

    Reading the settings through a function keeps the type checker from assuming
    that class state observed once cannot change later on.
    """
    return (
        sdpa_mem_efficient.enabled,
        sdpa_mem_efficient.query_chunk_size,
        sdpa_mem_efficient.head_chunk_size,
        sdpa_mem_efficient.checkpoint,
    )


def test_mem_efficient_kernel_disabled_by_default() -> None:
    # GIVEN an unmodified process state
    # WHEN the settings are read
    enabled, query_chunk_size, head_chunk_size, checkpoint = _settings()

    # THEN the memory-efficient kernel is not selected, is set up to checkpoint,
    #       attends the whole query at once, and chunks one head at a time (the
    #       more memory-efficient default, since a single head's Q/K/V slice is
    #       contiguous while a query chunk spanning every head is not)
    assert not enabled
    assert checkpoint
    assert query_chunk_size is None
    assert head_chunk_size == 1


def test_context_manager_restores_previous_settings() -> None:
    # GIVEN the default settings
    default_settings = _settings()

    # WHEN contexts are nested
    with sdpa_mem_efficient(query_chunk_size=64, checkpoint=False):
        assert _settings() == (True, 64, default_settings[2], False)

        with sdpa_mem_efficient(False, query_chunk_size=8, head_chunk_size=2):
            # THEN settings that are not given keep the value of the outer context
            assert _settings() == (False, 8, 2, False)

        assert _settings() == (True, 64, default_settings[2], False)

    # THEN the original settings are restored
    assert _settings() == default_settings


def test_context_manager_instance_can_be_reused() -> None:
    # GIVEN a single context manager instance
    default_settings = _settings()
    context = sdpa_mem_efficient(query_chunk_size=32, head_chunk_size=2)

    # WHEN it is entered twice, including nested
    with context:
        with context:
            assert _settings() == (True, 32, 2, True)
        assert _settings() == (True, 32, 2, True)

    # THEN the original settings are restored
    assert _settings() == default_settings


@pytest.mark.parametrize("query_chunk_size", [0, -1])
def test_non_positive_query_chunk_size_raises(query_chunk_size: int) -> None:
    # GIVEN a non-positive chunk size
    # WHEN a context manager is created with it
    # THEN a ValueError is raised
    with pytest.raises(ValueError, match="query_chunk_size"):
        sdpa_mem_efficient(query_chunk_size=query_chunk_size)


@pytest.mark.parametrize("head_chunk_size", [0, -1])
def test_non_positive_head_chunk_size_raises(head_chunk_size: int) -> None:
    # GIVEN a non-positive chunk size
    # WHEN a context manager is created with it
    # THEN a ValueError is raised
    with pytest.raises(ValueError, match="head_chunk_size"):
        sdpa_mem_efficient(head_chunk_size=head_chunk_size)


def test_context_manager_exposed_on_fastforward_namespace() -> None:
    # GIVEN fastforward re-exports the SDPA settings at the package level
    # WHEN reading via the package namespace
    # THEN it resolves to the same object
    assert fastforward.sdpa_mem_efficient is sdpa_mem_efficient


# ------------------------------------------------------------------------------
# DISPATCH TESTS
# ------------------------------------------------------------------------------


def test_dispatch_selects_kernel_based_on_settings() -> None:
    # GIVEN the memory-efficient kernel is disabled
    # WHEN the operator is dispatched
    # THEN the MATH kernel is selected
    assert dispatch("scaled_dot_product_attention") is scaled_dot_product_attention_math

    # WHEN the memory-efficient kernel is enabled
    with sdpa_mem_efficient():
        # THEN it is selected instead
        assert (
            dispatch("scaled_dot_product_attention") is scaled_dot_product_attention_mem_efficient
        )

    # THEN the MATH kernel is selected again afterwards
    assert dispatch("scaled_dot_product_attention") is scaled_dot_product_attention_math


def test_query_that_fits_a_single_chunk_delegates_to_math_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # GIVEN a query that is not longer than the chunk size, and all heads fit in
    #       a single group (overriding the default head_chunk_size=1)
    q, k, v, attn_mask, is_causal = _make_attn_inputs("self_attn", 1, "float", "cpu")
    spy = Mock(wraps=scaled_dot_product_attention_math)
    monkeypatch.setattr(
        "fastforward.nn.functional.custom.sdpa.scaled_dot_product_attention_math", spy
    )

    # WHEN the memory-efficient kernel is used
    with (
        sdpa_mem_efficient(query_chunk_size=q.size(-2), head_chunk_size=q.size(-3)),
        torch.no_grad(),
    ):
        out = FF.scaled_dot_product_attention(
            q, k, v, attn_mask, is_causal=is_causal, strict_quantization=False
        )

    # THEN it hands the whole query to the MATH kernel, which computes the same result
    assert spy.call_count == 1
    with torch.no_grad():
        expected = scaled_dot_product_attention_math(
            query=q, key=k, value=v, attn_mask=attn_mask, strict_quantization=False
        )
    assert torch.all(out == expected)


@pytest.mark.parametrize("head_chunk_size", [None, 1])
def test_empty_query_matches_math_kernel(head_chunk_size: int | None) -> None:
    # GIVEN a query without any positions, which cannot be split into chunks
    q = torch.randn(2, 2, 0, 4)
    k = torch.randn(2, 2, 6, 4)
    v = torch.randn(2, 2, 6, 4)

    # WHEN the memory-efficient kernel is used, also with several head groups
    with sdpa_mem_efficient(head_chunk_size=head_chunk_size), torch.no_grad():
        out = FF.scaled_dot_product_attention(q, k, v, strict_quantization=False)

    # THEN the output is empty and equal to the MATH kernel output
    with torch.no_grad():
        expected = scaled_dot_product_attention_math(
            query=q, key=k, value=v, strict_quantization=False
        )
    assert out.shape == expected.shape == q.shape
    assert torch.all(out == expected)


def test_head_chunk_size_prevents_math_kernel_shortcut(monkeypatch: pytest.MonkeyPatch) -> None:
    # GIVEN a query that is not longer than the chunk size, but more heads than
    #       the configured head group size
    q, k, v, attn_mask, is_causal = _make_attn_inputs("self_attn", 1, "float", "cpu")
    spy = Mock(wraps=scaled_dot_product_attention_math)
    monkeypatch.setattr(
        "fastforward.nn.functional.custom.sdpa.scaled_dot_product_attention_math", spy
    )

    # WHEN the memory-efficient kernel is used
    with (
        sdpa_mem_efficient(query_chunk_size=q.size(-2), head_chunk_size=1),
        torch.no_grad(),
    ):
        out = FF.scaled_dot_product_attention(
            q, k, v, attn_mask, is_causal=is_causal, strict_quantization=False
        )

    # THEN the query is not handed to the MATH kernel wholesale -- it is still
    #      chunked by head -- but the result is unaffected
    assert spy.call_count == 0
    with torch.no_grad():
        expected = scaled_dot_product_attention_math(
            query=q, key=k, value=v, attn_mask=attn_mask, strict_quantization=False
        )
    assert_max_abs_err(expected, out, atol=1e-6)


# ------------------------------------------------------------------------------
# EQUIVALENCE WITH THE MATH KERNEL
# ------------------------------------------------------------------------------


@pytest.mark.parametrize("head_chunk_size", [None, *_HEAD_CHUNK_SIZES])
@pytest.mark.parametrize("query_chunk_size", _QUERY_CHUNK_SIZES)
@pytest.mark.parametrize("use_attn_mask", _ATTN_MASK_OPTS, ids=lambda mask: f"attn_mask={mask}")
@pytest.mark.parametrize("groups", GROUP_ATTN_VALUES, ids=lambda g: f"gqa={g}" if g > 1 else "")
@pytest.mark.parametrize("scale", [None, 0.1], ids=lambda s: f"scale={s}" if s else "")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("input_type", ["cross_attn", "self_attn"])
def test_unquantized_attn_matches_math_kernel(
    input_type: str,
    device: torch.device,
    scale: float | None,
    groups: int,
    use_attn_mask: bool | str,
    query_chunk_size: int,
    head_chunk_size: int | None,
) -> None:
    # GIVEN the output of the fastforward MATH kernel
    q, k, v, attn_mask, is_causal = _make_attn_inputs(input_type, groups, use_attn_mask, device)
    kwargs: dict[str, Any] = {
        "attn_mask": attn_mask,
        "scale": scale,
        "is_causal": is_causal,
        "enable_gqa": groups > 1,
        "strict_quantization": False,
    }
    with torch.no_grad():
        out_math = FF.scaled_dot_product_attention(q, k, v, **kwargs)

    # WHEN the memory-efficient kernel processes the same inputs in chunks, split
    #      by query position and, independently, by attention head
    with (
        sdpa_mem_efficient(query_chunk_size=query_chunk_size, head_chunk_size=head_chunk_size),
        torch.no_grad(),
    ):
        out_mem_efficient = FF.scaled_dot_product_attention(q, k, v, **kwargs)

    # THEN both kernels agree up to the accumulation order of the matmuls
    _print_abs_max_err(out_mem_efficient, out_math, 1e-6)
    assert_max_abs_err(out_math, out_mem_efficient, atol=1e-6)


@pytest.mark.parametrize("query_chunk_size", _QUERY_CHUNK_SIZES)
@pytest.mark.parametrize("use_attn_mask", _ATTN_MASK_OPTS, ids=lambda mask: f"attn_mask={mask}")
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16], ids=lambda dt: str(dt).split(".")[-1]
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_low_precision_attn_matches_math_kernel(
    device: torch.device,
    dtype: torch.dtype,
    use_attn_mask: bool | str,
    query_chunk_size: int,
) -> None:
    # GIVEN low-precision inputs, which both kernels upcast the same way
    q, k, v, attn_mask, is_causal = _make_attn_inputs("self_attn", 1, use_attn_mask, device, dtype)
    kwargs: dict[str, Any] = {
        "attn_mask": attn_mask,
        "is_causal": is_causal,
        "strict_quantization": False,
    }
    with torch.no_grad():
        out_math = FF.scaled_dot_product_attention(q, k, v, **kwargs)

    # WHEN the memory-efficient kernel processes the same inputs in chunks
    with sdpa_mem_efficient(query_chunk_size=query_chunk_size), torch.no_grad():
        out_mem_efficient = FF.scaled_dot_product_attention(q, k, v, **kwargs)

    # THEN the outputs are equal and are cast back to the input dtype
    assert out_mem_efficient.dtype == dtype
    _print_abs_max_err(out_mem_efficient, out_math, 1e-3)
    assert_max_abs_err(out_math, out_mem_efficient, atol=1e-3)


@pytest.mark.parametrize("head_chunk_size", [None, 1])
@pytest.mark.parametrize("query_chunk_size", _QUERY_CHUNK_SIZES)
@pytest.mark.parametrize("use_attn_mask", ["none", "float", "bool", "causal"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_unbatched_2d_query_matches_math_kernel(
    device: str, use_attn_mask: str, query_chunk_size: int, head_chunk_size: int | None
) -> None:
    # GIVEN 2-d (L, E) query and (S, E) key and value, which have neither batch
    #       nor head dimensions and are supported by the MATH kernel
    L, S, E = 8, 6, 4
    q = torch.randn(L, E, device=device)
    k = torch.randn(S, E, device=device)
    v = torch.randn(S, E, device=device)
    kwargs: dict[str, Any] = {"strict_quantization": False}
    if use_attn_mask == "float":
        kwargs["attn_mask"] = torch.rand(L, S, device=device) - 0.5
    elif use_attn_mask == "bool":
        kwargs["attn_mask"] = torch.rand(L, S, device=device) > 0.3
        kwargs["attn_mask"][:, 0] = True  # Keep at least one visible position per row
    elif use_attn_mask == "causal":
        kwargs["is_causal"] = True

    with torch.no_grad():
        out_math = scaled_dot_product_attention_math(query=q, key=k, value=v, **kwargs)

    # WHEN the memory-efficient kernel processes the query in chunks
    with (
        sdpa_mem_efficient(query_chunk_size=query_chunk_size, head_chunk_size=head_chunk_size),
        torch.no_grad(),
    ):
        out_mem_efficient = FF.scaled_dot_product_attention(q, k, v, **kwargs)

    # THEN the output has the shape of the query and matches the MATH kernel
    assert out_mem_efficient.shape == q.shape
    assert_max_abs_err(out_math, out_mem_efficient, atol=1e-6)


@pytest.mark.parametrize("head_chunk_size", [None, 1, 3])
@pytest.mark.parametrize("query_chunk_size", _QUERY_CHUNK_SIZES)
@pytest.mark.parametrize("use_attn_mask", ["none", "float", "causal"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_query_with_single_head_broadcast_over_key_and_value_heads(
    device: str, use_attn_mask: str, query_chunk_size: int, head_chunk_size: int | None
) -> None:
    # GIVEN a query with a single head and key and value with several heads, so
    #       that the query broadcasts over the heads and the output has 4 heads
    N, H, L, S, E = 2, 4, 8, 6, 4
    q = torch.randn(N, 1, L, E, device=device)
    k = torch.randn(N, H, S, E, device=device)
    v = torch.randn(N, H, S, E, device=device)
    kwargs: dict[str, Any] = {"strict_quantization": False}
    if use_attn_mask == "float":
        kwargs["attn_mask"] = torch.rand(N, H, L, S, device=device) - 0.5
    elif use_attn_mask == "causal":
        kwargs["is_causal"] = True

    with torch.no_grad():
        out_math = scaled_dot_product_attention_math(query=q, key=k, value=v, **kwargs)

    # WHEN the memory-efficient kernel processes the query in chunks
    with (
        sdpa_mem_efficient(query_chunk_size=query_chunk_size, head_chunk_size=head_chunk_size),
        torch.no_grad(),
    ):
        out_mem_efficient = FF.scaled_dot_product_attention(q, k, v, **kwargs)

    # THEN the output has the broadcast heads and matches the MATH kernel
    assert out_mem_efficient.shape == (N, H, L, E)
    assert_max_abs_err(out_math, out_mem_efficient, atol=1e-6)


@pytest.mark.parametrize("kv_shape", [(6, 4), (1, 6, 4), (4, 6, 4)], ids=str)
@pytest.mark.parametrize("head_chunk_size", [1, 3, 100])
@pytest.mark.parametrize("query_chunk_size", _QUERY_CHUNK_SIZES)
@pytest.mark.parametrize("use_attn_mask", ["none", "float", "causal"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_query_with_heads_and_key_value_without_heads(
    device: str,
    use_attn_mask: str,
    query_chunk_size: int,
    head_chunk_size: int,
    kv_shape: tuple[int, ...],
) -> None:
    # GIVEN a (N, H, L, E) query and key and value without a batch and head
    #       dimension, (S, E), or with fewer dimensions than the query, (H, S, E)
    #       or (1, S, E), which broadcast over the missing leading dimensions
    N, H, L, E = 2, 4, 8, 4
    q = torch.randn(N, H, L, E, device=device)
    k = torch.randn(*kv_shape, device=device)
    v = torch.randn(*kv_shape, device=device)
    kwargs: dict[str, Any] = {"strict_quantization": False}
    if use_attn_mask == "float":
        kwargs["attn_mask"] = torch.rand(N, H, L, 6, device=device) - 0.5
    elif use_attn_mask == "causal":
        kwargs["is_causal"] = True

    with torch.no_grad():
        out_math = scaled_dot_product_attention_math(query=q, key=k, value=v, **kwargs)

    # WHEN the memory-efficient kernel processes the heads and query in chunks
    with (
        sdpa_mem_efficient(query_chunk_size=query_chunk_size, head_chunk_size=head_chunk_size),
        torch.no_grad(),
    ):
        out_mem_efficient = FF.scaled_dot_product_attention(q, k, v, **kwargs)

    # THEN the output has the shape of the query and matches the MATH kernel
    assert out_mem_efficient.shape == q.shape
    assert_max_abs_err(out_math, out_mem_efficient, atol=1e-6)


@pytest.mark.parametrize("head_chunk_size", [None, 1])
@pytest.mark.parametrize("query_chunk_size", _QUERY_CHUNK_SIZES)
@pytest.mark.parametrize("use_attn_mask", ["none", "float", "bool", "causal"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_2d_query_broadcast_over_batched_key_and_value(
    device: str, use_attn_mask: str, query_chunk_size: int, head_chunk_size: int | None
) -> None:
    # GIVEN a 2-d (L, E) query and batched (B, S, E) key and value, so that the
    #       query broadcasts over the batch and the output is (B, L, E)
    B, L, S, E = 2, 8, 6, 4
    q = torch.randn(L, E, device=device)
    k = torch.randn(B, S, E, device=device)
    v = torch.randn(B, S, E, device=device)
    kwargs: dict[str, Any] = {"strict_quantization": False}
    if use_attn_mask == "float":
        kwargs["attn_mask"] = torch.rand(L, S, device=device) - 0.5
    elif use_attn_mask == "bool":
        kwargs["attn_mask"] = torch.rand(L, S, device=device) > 0.3
        kwargs["attn_mask"][:, 0] = True  # Keep at least one visible position per row
    elif use_attn_mask == "causal":
        kwargs["is_causal"] = True

    with torch.no_grad():
        out_math = scaled_dot_product_attention_math(query=q, key=k, value=v, **kwargs)

    # WHEN the memory-efficient kernel processes the query in chunks
    with (
        sdpa_mem_efficient(query_chunk_size=query_chunk_size, head_chunk_size=head_chunk_size),
        torch.no_grad(),
    ):
        out_mem_efficient = FF.scaled_dot_product_attention(q, k, v, **kwargs)

    # THEN the output has the broadcast batch dimension and matches the MATH kernel
    assert out_mem_efficient.shape == (B, L, E)
    assert_max_abs_err(out_math, out_mem_efficient, atol=1e-6)


@pytest.mark.parametrize("head_chunk_size", [1, 3, 100])
@pytest.mark.parametrize("use_attn_mask", _ATTN_MASK_OPTS, ids=lambda mask: f"attn_mask={mask}")
@pytest.mark.parametrize(
    "key_heads, value_heads", [(1, 1), (1, 4), (4, 1)], ids=lambda h: f"heads={h}"
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_key_or_value_with_single_head_broadcast_over_query_heads(
    device: str,
    use_attn_mask: bool | str,
    head_chunk_size: int,
    key_heads: int,
    value_heads: int,
) -> None:
    # GIVEN key and/or value with a head dimension of 1, which broadcasts over the
    #       query heads and therefore must not be sliced per head group
    q, _, _, _ = _make_sdpa_input_tensors(N=2, H=4, S=6, L=8, E=4, device=device)
    k = torch.randn(2, key_heads, 6, 4, device=device)
    v = torch.randn(2, value_heads, 6, 4, device=device)
    kwargs: dict[str, Any] = {"strict_quantization": False}
    if use_attn_mask == "float":
        kwargs["attn_mask"] = torch.rand(2, 4, 8, 6, device=device) - 0.5
    elif use_attn_mask == "bool":
        kwargs["attn_mask"] = torch.rand(2, 4, 8, 6, device=device) > 0.3
        kwargs["attn_mask"][..., 0] = True  # Keep at least one visible position per row
    elif use_attn_mask == "causal":
        kwargs["is_causal"] = True

    with torch.no_grad():
        out_math = scaled_dot_product_attention_math(query=q, key=k, value=v, **kwargs)

    # WHEN the memory-efficient kernel processes the heads in groups
    with sdpa_mem_efficient(head_chunk_size=head_chunk_size), torch.no_grad():
        out_mem_efficient = FF.scaled_dot_product_attention(q, k, v, **kwargs)

    # THEN every head group attends to the single key/value head, matching the MATH kernel
    assert out_mem_efficient.shape == q.shape
    assert_max_abs_err(out_math, out_mem_efficient, atol=1e-6)


@pytest.mark.parametrize("ndim", [2, 4])
@pytest.mark.parametrize("head_chunk_size", [None, 1])
@pytest.mark.parametrize("query_chunk_size", [1, 3])
@pytest.mark.parametrize("use_attn_mask", ["none", "float", "causal"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_single_query_position_matches_math_kernel(
    device: str,
    use_attn_mask: str,
    query_chunk_size: int,
    head_chunk_size: int | None,
    ndim: int,
) -> None:
    # GIVEN a query with a single target position (e.g. one decoding step)
    lead = (2, 4) if ndim == 4 else ()
    q = torch.randn(*lead, 1, 4, device=device)
    k = torch.randn(*lead, 6, 4, device=device)
    v = torch.randn(*lead, 6, 4, device=device)
    kwargs: dict[str, Any] = {"strict_quantization": False}
    if use_attn_mask == "float":
        kwargs["attn_mask"] = torch.rand(*lead, 1, 6, device=device) - 0.5
    elif use_attn_mask == "causal":
        kwargs["is_causal"] = True

    with torch.no_grad():
        out_math = scaled_dot_product_attention_math(query=q, key=k, value=v, **kwargs)

    # WHEN the memory-efficient kernel is used, also with several head groups
    with (
        sdpa_mem_efficient(query_chunk_size=query_chunk_size, head_chunk_size=head_chunk_size),
        torch.no_grad(),
    ):
        out_mem_efficient = FF.scaled_dot_product_attention(q, k, v, **kwargs)

    # THEN the output matches the MATH kernel
    assert out_mem_efficient.shape == q.shape
    assert_max_abs_err(out_math, out_mem_efficient, atol=1e-6)


@pytest.mark.parametrize("query_chunk_size", _QUERY_CHUNK_SIZES)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("L, S", [(8, 8), (8, 16), (16, 8)], ids=lambda n: f"len={n}")
def test_causal_mask_is_offset_per_chunk(
    L: int, S: int, device: torch.device, query_chunk_size: int
) -> None:
    # GIVEN a causal attention over a query that spans several chunks
    q, k, v, _ = _make_sdpa_input_tensors(N=2, H=2, S=S, L=L, E=4, device=device)

    # WHEN the memory-efficient kernel processes the query in chunks
    with sdpa_mem_efficient(query_chunk_size=query_chunk_size), torch.no_grad():
        out_mem_efficient = FF.scaled_dot_product_attention(
            q, k, v, is_causal=True, strict_quantization=False
        )

    # THEN each chunk masks against its position in the full target sequence, so
    #       the result matches torch's causal attention
    with sdpa_kernel(backends=[SDPBackend.MATH]), torch.no_grad():
        out_torch = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    _print_abs_max_err(out_mem_efficient, out_torch, 1e-6)
    assert_max_abs_err(out_torch, out_mem_efficient, atol=1e-6)


@pytest.mark.parametrize("query_chunk_size", _QUERY_CHUNK_SIZES)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_attn_mask_broadcast_over_target_sequence(
    device: torch.device, query_chunk_size: int
) -> None:
    # GIVEN a mask with a singleton target-sequence dimension, which broadcasts
    #        over the query and therefore must not be chunked
    q, k, v, _ = _make_sdpa_input_tensors(N=2, H=2, S=6, L=8, E=4, device=device)
    attn_mask = torch.rand(2, 2, 1, 6, device=device) - 0.5
    kwargs: dict[str, Any] = {"attn_mask": attn_mask, "strict_quantization": False}

    with torch.no_grad():
        out_math = FF.scaled_dot_product_attention(q, k, v, **kwargs)

    # WHEN the memory-efficient kernel processes the query in chunks
    with sdpa_mem_efficient(query_chunk_size=query_chunk_size), torch.no_grad():
        out_mem_efficient = FF.scaled_dot_product_attention(q, k, v, **kwargs)

    # THEN both kernels agree
    assert_max_abs_err(out_math, out_mem_efficient, atol=1e-6)


@pytest.mark.parametrize("head_chunk_size", _HEAD_CHUNK_SIZES)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_attn_mask_broadcast_over_heads(device: torch.device, head_chunk_size: int) -> None:
    # GIVEN a mask with a singleton head dimension, which broadcasts over every
    #       head group and therefore must not be chunked
    q, k, v, _ = _make_sdpa_input_tensors(N=2, H=4, S=6, L=8, E=4, device=device)
    attn_mask = torch.rand(2, 1, 8, 6, device=device) - 0.5
    kwargs: dict[str, Any] = {"attn_mask": attn_mask, "strict_quantization": False}

    with torch.no_grad():
        out_math = FF.scaled_dot_product_attention(q, k, v, **kwargs)

    # WHEN the memory-efficient kernel processes the heads in groups
    with sdpa_mem_efficient(head_chunk_size=head_chunk_size), torch.no_grad():
        out_mem_efficient = FF.scaled_dot_product_attention(q, k, v, **kwargs)

    # THEN both kernels agree
    assert_max_abs_err(out_math, out_mem_efficient, atol=1e-6)


@pytest.mark.parametrize("query_chunk_size", _QUERY_CHUNK_SIZES)
@pytest.mark.parametrize("bits", [16, 8], ids=lambda b: f"{b}b")
@pytest.mark.parametrize("strict_quant", [True, False], ids=lambda sq: "strict" if sq else "")
@pytest.mark.parametrize("use_attn_mask", _ATTN_MASK_OPTS, ids=lambda mask: f"attn_mask={mask}")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("input_type", ["cross_attn", "self_attn"])
def test_quantized_attn_matches_math_kernel(
    input_type: str,
    bits: int,
    device: torch.device,
    use_attn_mask: bool | str,
    strict_quant: bool,
    query_chunk_size: int,
) -> None:
    # GIVEN a quantized SDPA whose ranges were estimated with the MATH kernel
    q, k, v, attn_mask, is_causal = _make_attn_inputs(input_type, 1, use_attn_mask, device)
    qsdpa = _QuantizedSDPA(bits).to(device)
    kwargs: dict[str, Any] = {
        "is_causal": is_causal,
        "strict_quantization": strict_quant,
        "neg_inf": -1000.0,
    }
    with torch.no_grad():
        with estimate_ranges(qsdpa, range_setting.running_minmax):
            qsdpa(q, k, v, attn_mask, **kwargs)
        out_math = qsdpa(q, k, v, attn_mask, **kwargs).dequantize()

    # WHEN the memory-efficient kernel runs with the very same quantizers
    with sdpa_mem_efficient(query_chunk_size=query_chunk_size), torch.no_grad():
        out_mem_efficient = qsdpa(q, k, v, attn_mask, **kwargs).dequantize()

    # THEN switching kernels does not change the result, so no re-estimation of
    #       the quantization ranges is needed. Chunking changes the accumulation
    #       order of the matmuls, which can move a value across a quantization
    #       threshold, so the outputs are allowed to differ by a quantization step.
    tol = 2 * qsdpa.output_quantization_step()
    _print_abs_max_err(out_mem_efficient, out_math, tol)
    assert_max_abs_err(out_math, out_mem_efficient, atol=tol)


@pytest.mark.parametrize(
    "quantize_key",
    [
        "q",
        "k",
        "v",
        "attn_scores",
        "attn_mask",
        "masked_scores",
        "attn_weights",
        "scaled_query",
        "scaled_key",
        "dropout",
        "output",
    ],
)
@pytest.mark.parametrize("use_attn_mask", ["float", "causal"], ids=lambda mask: f"attn_mask={mask}")
def test_partially_quantized_attn_matches_math_kernel(
    quantize_key: str, use_attn_mask: bool | str
) -> None:
    # GIVEN a single quantizer, estimated with the MATH kernel
    q, k, v, attn_mask, is_causal = _make_attn_inputs("cross_attn", 1, use_attn_mask, "cpu")
    qsdpa = _QuantizedSDPA(8, quantizer_key=quantize_key)
    kwargs: dict[str, Any] = {
        "is_causal": is_causal,
        "strict_quantization": False,
        "neg_inf": -1000.0,
    }
    with torch.no_grad():
        with estimate_ranges(qsdpa, range_setting.running_minmax):
            qsdpa(q, k, v, attn_mask, **kwargs)
        out_math = qsdpa(q, k, v, attn_mask, **kwargs).dequantize()

    # WHEN the memory-efficient kernel runs with the same quantizer
    with sdpa_mem_efficient(query_chunk_size=3), torch.no_grad():
        out_mem_efficient = qsdpa(q, k, v, attn_mask, **kwargs).dequantize()

    # THEN the quantizer observes the same values in both kernels, up to a
    #       quantization step of the single active quantizer
    assert len(list(qsdpa.named_quantizers())) == 1
    assert_max_abs_err(out_math, out_mem_efficient, atol=1e-3)


@pytest.mark.parametrize("granularity", ["per_tensor", "per_channel"])
def test_chunking_a_quantized_query(granularity: str) -> None:
    # GIVEN a query quantized with a granularity that supports slicing
    q, k, v, _ = _make_sdpa_input_tensors(N=2, H=2, S=6, L=8, E=4)
    quantizer = fastforward.nn.LinearQuantizer(
        num_bits=8,
        granularity=(
            fastforward.PerTensor() if granularity == "per_tensor" else fastforward.PerChannel(-1)
        ),
    )
    with estimate_ranges(quantizer, range_setting.running_minmax), torch.no_grad():
        quantizer(q)
    with torch.no_grad():
        quantized_query = quantizer(q)
        out_math = FF.scaled_dot_product_attention(quantized_query, k, v, strict_quantization=False)

    # WHEN the memory-efficient kernel splits the quantized query into chunks
    with sdpa_mem_efficient(query_chunk_size=3), torch.no_grad():
        out_mem_efficient = FF.scaled_dot_product_attention(
            quantized_query, k, v, strict_quantization=False
        )

    # THEN the chunks carry the quantization parameters of the full query
    assert_max_abs_err(out_math, out_mem_efficient, atol=1e-6)


@pytest.mark.parametrize("query_chunk_size", _QUERY_CHUNK_SIZES)
@pytest.mark.parametrize("granularity", ["per_tensor", "per_channel"])
def test_attn_mask_quantizer_without_mask_matches_math_kernel(
    granularity: str, query_chunk_size: int
) -> None:
    # GIVEN an attention mask quantizer and no attention mask, so that the zero
    #       attention bias is itself quantized, with ranges estimated by the MATH kernel
    q, k, v, _ = _make_sdpa_input_tensors(N=2, H=2, S=6, L=8, E=4)
    attn_mask_quantizer = _linear_quantizer(granularity)
    kwargs: dict[str, Any] = {
        "attn_mask_quantizer": attn_mask_quantizer,
        "strict_quantization": False,
    }
    with estimate_ranges(attn_mask_quantizer, range_setting.running_minmax), torch.no_grad():
        FF.scaled_dot_product_attention(q, k, v, **kwargs)
    with torch.no_grad():
        out_math = FF.scaled_dot_product_attention(q, k, v, **kwargs)

    # WHEN the memory-efficient kernel runs with the very same quantizer
    with sdpa_mem_efficient(query_chunk_size=query_chunk_size), torch.no_grad():
        out_mem_efficient = FF.scaled_dot_product_attention(q, k, v, **kwargs)

    # THEN both kernels agree, so the quantizer accepts the bias of every chunk
    assert_max_abs_err(out_math, out_mem_efficient, atol=1e-6)


# ------------------------------------------------------------------------------
# GRADIENT TESTS
# ------------------------------------------------------------------------------


def _attention_gradients(
    query_chunk_size: int | None, is_causal: bool = False, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, ...]:
    """Return the output and the query, key and value gradients of one backward pass."""
    q, k, v, _ = _make_sdpa_input_tensors(N=2, H=2, S=6, L=8, E=4, dtype=dtype)
    q, k, v = (tensor.requires_grad_(True) for tensor in (q, k, v))

    def forward() -> torch.Tensor:
        return FF.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal, strict_quantization=False
        )

    if query_chunk_size is None:
        out = forward()
    else:
        with sdpa_mem_efficient(query_chunk_size=query_chunk_size):
            out = forward()
    # A non-linear loss so that the gradient depends on the output values.
    out.square().sum().backward()
    assert q.grad is not None and k.grad is not None and v.grad is not None
    return out.detach(), q.grad, k.grad, v.grad


@pytest.mark.parametrize("query_chunk_size", _QUERY_CHUNK_SIZES)
@pytest.mark.parametrize("is_causal", [False, True], ids=lambda c: "causal" if c else "")
def test_gradients_match_math_kernel(query_chunk_size: int, is_causal: bool) -> None:
    # GIVEN the gradients of the MATH kernel
    torch.manual_seed(0)
    expected = _attention_gradients(None, is_causal)

    # WHEN the memory-efficient kernel is differentiated
    torch.manual_seed(0)
    actual = _attention_gradients(query_chunk_size, is_causal)

    # THEN recomputing each chunk in the backward pass yields the same gradients
    for expected_tensor, actual_tensor in zip(expected, actual):
        assert_max_abs_err(expected_tensor, actual_tensor, atol=1e-5)


def test_gradients_match_torch_autograd() -> None:
    # GIVEN double-precision inputs, so that the comparison is not limited by
    #        floating-point accumulation
    torch.manual_seed(0)
    q, k, v, _ = _make_sdpa_input_tensors(N=2, H=2, S=6, L=8, E=4, dtype=torch.float64)
    torch_q, torch_k, torch_v = (tensor.clone().requires_grad_(True) for tensor in (q, k, v))
    q, k, v = (tensor.requires_grad_(True) for tensor in (q, k, v))

    # WHEN the memory-efficient kernel and torch are differentiated
    with sdpa_mem_efficient(query_chunk_size=3):
        FF.scaled_dot_product_attention(
            q, k, v, strict_quantization=False
        ).square().sum().backward()
    F.scaled_dot_product_attention(torch_q, torch_k, torch_v).square().sum().backward()

    # THEN the gradients agree with torch's own attention gradients
    for tensor, torch_tensor in ((q, torch_q), (k, torch_k), (v, torch_v)):
        assert tensor.grad is not None and torch_tensor.grad is not None
        assert_max_abs_err(tensor.grad, torch_tensor.grad, atol=1e-12)


@pytest.fixture
def spy_checkpoint(monkeypatch: pytest.MonkeyPatch) -> Iterator[Mock]:
    spy = Mock(wraps=torch.utils.checkpoint.checkpoint)
    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", spy)
    yield spy


def test_query_chunks_are_checkpointed_when_a_backward_pass_can_happen(
    spy_checkpoint: Mock,
) -> None:
    # GIVEN a query spanning four chunks, of which only the query requires grad,
    #       with head-chunking disabled to isolate the query axis
    q, k, v, _ = _make_sdpa_input_tensors(N=2, H=2, S=6, L=8, E=4)
    q.requires_grad_(True)

    # WHEN the memory-efficient kernel runs with gradients enabled
    with sdpa_mem_efficient(query_chunk_size=2, head_chunk_size=2):
        FF.scaled_dot_product_attention(q, k, v, strict_quantization=False).sum().backward()

    # THEN every chunk is wrapped in a gradient checkpoint
    assert spy_checkpoint.call_count == 4


def test_head_and_query_chunks_are_checkpointed_when_a_backward_pass_can_happen(
    spy_checkpoint: Mock,
) -> None:
    # GIVEN a query spanning four chunks and two heads spanning two head groups,
    #       of which only the query requires grad
    q, k, v, _ = _make_sdpa_input_tensors(N=2, H=2, S=6, L=8, E=4)
    q.requires_grad_(True)

    # WHEN the memory-efficient kernel runs with gradients enabled
    with sdpa_mem_efficient(query_chunk_size=2, head_chunk_size=1):
        FF.scaled_dot_product_attention(q, k, v, strict_quantization=False).sum().backward()

    # THEN every (head group, query chunk) pair is wrapped in its own checkpoint
    assert spy_checkpoint.call_count == 2 * 4


@pytest.mark.parametrize(
    "run",
    [
        pytest.param(lambda fn: _run_without_grad(fn), id="no_grad"),
        pytest.param(lambda fn: _run_without_checkpointing(fn), id="checkpoint=False"),
    ],
)
def test_query_chunks_are_not_checkpointed_when_no_backward_pass_follows(
    run: Callable[[Callable[[], torch.Tensor]], torch.Tensor], spy_checkpoint: Mock
) -> None:
    # GIVEN a query that requires a gradient
    q, k, v, _ = _make_sdpa_input_tensors(N=2, H=2, S=6, L=8, E=4)
    q.requires_grad_(True)

    def forward() -> torch.Tensor:
        with sdpa_mem_efficient(query_chunk_size=2):
            return FF.scaled_dot_product_attention(q, k, v, strict_quantization=False)

    # WHEN the memory-efficient kernel runs with gradients disabled, or with
    #       checkpointing turned off
    run(forward)

    # THEN no gradient checkpoint is created
    assert spy_checkpoint.call_count == 0


def test_query_chunks_are_not_checkpointed_when_nothing_requires_grad(
    spy_checkpoint: Mock,
) -> None:
    # GIVEN inputs that do not require a gradient
    q, k, v, _ = _make_sdpa_input_tensors(N=2, H=2, S=6, L=8, E=4)

    # WHEN the memory-efficient kernel runs
    with sdpa_mem_efficient(query_chunk_size=2):
        FF.scaled_dot_product_attention(q, k, v, strict_quantization=False)

    # THEN no gradient checkpoint is created, so that `torch.utils.checkpoint`
    #       does not warn about inputs without a gradient
    assert spy_checkpoint.call_count == 0


def _run_without_grad(forward: Callable[[], torch.Tensor]) -> torch.Tensor:
    with torch.no_grad():
        return forward()


def _run_without_checkpointing(forward: Callable[[], torch.Tensor]) -> torch.Tensor:
    with sdpa_mem_efficient(checkpoint=False):
        return forward()


@pytest.mark.parametrize(
    "checkpoint",
    [
        True,
        pytest.param(
            False,
            marks=pytest.mark.xfail(
                strict=True,
                raises=RuntimeError,
                reason="Chunks without checkpointing modify tensors saved for backward",
            ),
        ),
    ],
    ids=lambda c: f"checkpoint={c}",
)
def test_ranges_estimated_with_backward_match_math_kernel(checkpoint: bool) -> None:
    # GIVEN identical inputs and quantizers, of which the query requires a gradient
    q, k, v, _ = _make_sdpa_input_tensors(N=2, H=2, S=6, L=8, E=4)
    q.requires_grad_(True)

    # WHEN the ranges of the attention score quantizer are estimated with a backward
    #       pass inside the estimation context, once with the MATH kernel and once with
    #       the memory-efficient kernel
    expected_min, expected_max = _estimate_scores_range(q, k, v, chunked=False)
    actual_min, actual_max = _estimate_scores_range(q, k, v, chunked=True, checkpoint=checkpoint)

    # THEN the estimated ranges are identical
    torch.testing.assert_close(actual_min, expected_min)
    torch.testing.assert_close(actual_max, expected_max)


def _estimate_scores_range(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, chunked: bool, checkpoint: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    scores_quantizer = _linear_quantizer()
    with (
        estimate_ranges(scores_quantizer, range_setting.running_minmax),
        sdpa_mem_efficient(
            enabled=chunked, query_chunk_size=2, head_chunk_size=2, checkpoint=checkpoint
        ),
    ):
        out = FF.scaled_dot_product_attention(
            q, k, v, attn_scores_quantizer=scores_quantizer, strict_quantization=False
        )
        out.sum().backward()
    min_, max_ = scores_quantizer.quantization_range
    assert isinstance(min_, torch.Tensor)
    assert isinstance(max_, torch.Tensor)
    return min_, max_


def _linear_quantizer(granularity: str = "per_tensor") -> fastforward.nn.LinearQuantizer:
    return fastforward.nn.LinearQuantizer(
        num_bits=8,
        granularity=(
            fastforward.PerTensor() if granularity == "per_tensor" else fastforward.PerChannel(-1)
        ),
    )


# ------------------------------------------------------------------------------
# MEMORY TESTS
# ------------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Peak memory is measured on CUDA")
@pytest.mark.parametrize("train", [False, True], ids=lambda t: "train" if t else "inference")
def test_chunking_reduces_peak_memory(train: bool) -> None:
    # GIVEN a self attention whose attention matrix dominates memory use, with
    #       head-chunking disabled to isolate the query axis
    L = S = 1024
    H = 8
    q, k, v, attn_mask = _make_sdpa_input_tensors(N=1, H=H, S=S, L=L, E=32, device="cuda")
    # The builder also returns a mask of the size of the attention matrix, which
    # this test does not use and which must not count towards peak memory.
    del attn_mask
    q.requires_grad_(train)

    def run(query_chunk_size: int | None) -> int:
        def forward() -> None:
            out = FF.scaled_dot_product_attention(q, k, v, strict_quantization=False)
            if train:
                out.sum().backward()
                q.grad = None

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.enable_grad() if train else torch.no_grad():
            if query_chunk_size is None:
                forward()
            else:
                with sdpa_mem_efficient(query_chunk_size=query_chunk_size, head_chunk_size=H):
                    forward()
        torch.cuda.synchronize()
        return int(torch.cuda.max_memory_allocated())

    # WHEN the query is processed in chunks of decreasing size
    peak_math = run(None)
    peak_large_chunk = run(L // 4)
    peak_small_chunk = run(L // 16)

    # THEN peak memory is governed by the chunk size rather than by the target
    #       sequence length, and drops by a large factor
    assert peak_small_chunk < peak_large_chunk < peak_math
    assert peak_small_chunk * 3 < peak_math


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Peak memory is measured on CUDA")
@pytest.mark.parametrize("train", [False, True], ids=lambda t: "train" if t else "inference")
def test_head_chunking_reduces_peak_memory(train: bool) -> None:
    # GIVEN a self attention with many heads whose attention matrix dominates
    #       memory use, and a query_chunk_size that never chunks the query, so
    #       any peak-memory change below is caused by head-chunking alone
    L = S = 1024
    H = 16
    q, k, v, attn_mask = _make_sdpa_input_tensors(N=1, H=H, S=S, L=L, E=32, device="cuda")
    del attn_mask
    q.requires_grad_(train)

    def run(head_chunk_size: int | None) -> int:
        def forward() -> None:
            out = FF.scaled_dot_product_attention(q, k, v, strict_quantization=False)
            if train:
                out.sum().backward()
                q.grad = None

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.enable_grad() if train else torch.no_grad():
            if head_chunk_size is None:
                forward()
            else:
                with sdpa_mem_efficient(query_chunk_size=L, head_chunk_size=head_chunk_size):
                    forward()
        torch.cuda.synchronize()
        return int(torch.cuda.max_memory_allocated())

    # WHEN the heads are processed in groups of decreasing size
    peak_math = run(None)
    peak_large_group = run(H // 4)
    peak_small_group = run(H // 16)

    # THEN peak memory is governed by the head group size rather than by the
    #       number of heads, and drops by a large factor
    assert peak_small_group < peak_large_group < peak_math
    assert peak_small_group * 3 < peak_math


# ------------------------------------------------------------------------------
# ERROR TESTS
# ------------------------------------------------------------------------------


def test_explicit_attn_mask_with_is_causal_raises() -> None:
    # GIVEN both an explicit mask and is_causal
    q, k, v, attn_mask, _ = _make_attn_inputs("self_attn", 1, "float", "cpu")

    # WHEN the memory-efficient kernel is used
    # THEN the same error as torch is raised, before any chunk is processed
    with sdpa_mem_efficient(query_chunk_size=1), torch.no_grad():
        with pytest.raises(ValueError, match="attn_mask should not be set when is_causal"):
            FF.scaled_dot_product_attention(
                q, k, v, attn_mask, is_causal=True, strict_quantization=False
            )


def test_grouped_query_attention_under_strict_quantization_raises() -> None:
    # GIVEN grouped-query attention inputs
    q, k, v, _, _ = _make_attn_inputs("self_attn", 4, False, "cpu")

    # WHEN the memory-efficient kernel is used under strict quantization
    # THEN the same error as the MATH kernel is raised
    with sdpa_mem_efficient(query_chunk_size=1), torch.no_grad():
        with pytest.raises(QuantizationError, match="enable_gqa"):
            FF.scaled_dot_product_attention(q, k, v, enable_gqa=True, strict_quantization=True)


@pytest.mark.parametrize("head_chunk_size", [1, 3, 100])
@pytest.mark.parametrize("query_chunk_size", _QUERY_CHUNK_SIZES)
@pytest.mark.parametrize("kv_heads", [1, 2, 4], ids=lambda h: f"kv_heads={h}")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_grouped_query_attention_matches_math_kernel(
    device: str, kv_heads: int, query_chunk_size: int, head_chunk_size: int
) -> None:
    # GIVEN 8 query heads and fewer key and value heads, each shared by a group of
    #       query heads (kv_heads=1 is multi-query attention)
    q, _, _, _ = _make_sdpa_input_tensors(N=2, H=8, S=6, L=8, E=4, device=device)
    k = torch.randn(2, kv_heads, 6, 4, device=device)
    v = torch.randn(2, kv_heads, 6, 4, device=device)
    kwargs: dict[str, Any] = {"enable_gqa": True, "is_causal": True, "strict_quantization": False}

    with torch.no_grad():
        out_math = scaled_dot_product_attention_math(query=q, key=k, value=v, **kwargs)

    # WHEN the memory-efficient kernel processes the heads and query in chunks
    with (
        sdpa_mem_efficient(query_chunk_size=query_chunk_size, head_chunk_size=head_chunk_size),
        torch.no_grad(),
    ):
        out_mem_efficient = FF.scaled_dot_product_attention(q, k, v, **kwargs)

    # THEN the output matches the MATH kernel
    assert out_mem_efficient.shape == q.shape
    assert_max_abs_err(out_math, out_mem_efficient, atol=1e-6)


@pytest.mark.parametrize("head_chunk_size", [None, 1, 2, 100])
def test_head_counts_that_do_not_broadcast_raise(head_chunk_size: int | None) -> None:
    # GIVEN a query with 4 heads and key and value with 2 heads, without enable_gqa
    q = torch.randn(2, 4, 8, 4)
    k = torch.randn(2, 2, 6, 4)
    v = torch.randn(2, 2, 6, 4)

    # WHEN the memory-efficient kernel chunks the query
    # THEN an error is raised, like in the MATH kernel, instead of broadcasting silently
    with pytest.raises(RuntimeError, match="number of heads"):
        with sdpa_mem_efficient(query_chunk_size=3, head_chunk_size=head_chunk_size):
            FF.scaled_dot_product_attention(q, k, v, strict_quantization=False)
    with pytest.raises(RuntimeError):
        scaled_dot_product_attention_math(query=q, key=k, value=v, strict_quantization=False)


# ------------------------------------------------------------------------------
# AGREEMENT WITH TORCH FOR ALL COMBINATIONS OF LEADING DIMENSIONS
# ------------------------------------------------------------------------------

# Dimensions in front of the trailing (L, E) of the query and (S, E) of key and
# value. They cover no leading dims, batch-only, head-only and batch + head
# layouts, size-1 (broadcast) dims, and sizes that are not compatible with each other.
_QUERY_LEADING_DIMS = [(), (1,), (4,), (8,), (2, 1), (2, 4), (1, 8), (2, 8)]
_KEY_VALUE_LEADING_DIMS = [(), (1,), (2,), (3,), (4,), (8,), (2, 1), (2, 2), (2, 4), (1, 4), (2, 8)]


@pytest.mark.parametrize("use_causal", [False, True], ids=["causal", ""])
@pytest.mark.parametrize("enable_gqa", [False, True], ids=["gqa", ""])
@pytest.mark.parametrize("head_chunk_size", [1, 3], ids=lambda hc: f"hchunk={hc}")
@pytest.mark.parametrize("kv_dims", _KEY_VALUE_LEADING_DIMS, ids=lambda x: f"kv={x}")
@pytest.mark.parametrize("q_dims", _QUERY_LEADING_DIMS, ids=lambda x: f"q={x}")
def test_leading_dimension_combinations_agree_with_torch(
    q_dims: tuple[int, ...],
    kv_dims: tuple[int, ...],
    enable_gqa: bool,
    use_causal: bool,
    head_chunk_size: int,
) -> None:
    # GIVEN query, key and value whose leading dimensions are any combination of
    #       the above, which torch's MATH kernel either accepts or rejects
    L, S, E = 8, 6, 4
    q = torch.randn(*q_dims, L, E)
    k = torch.randn(*kv_dims, S, E)
    v = torch.randn(*kv_dims, S, E)
    kwargs: dict[str, Any] = {"is_causal": use_causal, "enable_gqa": enable_gqa}

    torch_error: Exception | None = None
    out_torch = torch.empty(0)
    try:
        with sdpa_kernel(backends=[SDPBackend.MATH]), torch.no_grad():
            out_torch = F.scaled_dot_product_attention(q, k, v, **kwargs)
    except (RuntimeError, ValueError, IndexError) as error:
        torch_error = error

    # WHEN the memory-efficient kernel chunks the query and the heads
    ff_error: Exception | None = None
    out_ff = torch.empty(0)
    try:
        with (
            sdpa_mem_efficient(query_chunk_size=3, head_chunk_size=head_chunk_size),
            torch.no_grad(),
        ):
            out_ff = FF.scaled_dot_product_attention(q, k, v, strict_quantization=False, **kwargs)
    except (RuntimeError, ValueError, IndexError) as error:
        ff_error = error

    # THEN it rejects exactly what torch rejects, and otherwise returns the same output
    if torch_error is not None:
        assert ff_error is not None, f"torch raised {torch_error!r} but the kernel returned"
    else:
        assert ff_error is None, f"torch returned but the kernel raised {ff_error!r}"
        assert out_ff.shape == out_torch.shape
        assert_max_abs_err(out_torch, out_ff, atol=1e-5)
