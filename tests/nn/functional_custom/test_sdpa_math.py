# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Any, Callable

import fastforward
import pytest
import torch

from fastforward import estimate_ranges, range_setting
from fastforward.nn import functional as FF
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from ._utils import (
    _make_attn_inputs,
    _make_diffusers_cross_attn_inputs,
    _print_abs_max_err,
    _QuantizedSDPA,
    assert_max_abs_err,
)

torch.backends.cudnn.deterministic = True

_ATTN_MASK_OPTS = [False, "float", "bool", "causal"]

GROUP_ATTN_VALUES = [1, 4] if torch.__version__ >= "2.5" else [1]


# ------------------------------------------------------------------------------
# BIT-EXACT TESTS
@pytest.mark.skipif(torch.__version__ < "2.5", reason="Bit-exact tests require torch 2.5 or higher")
@pytest.mark.parametrize("use_attn_mask", _ATTN_MASK_OPTS, ids=lambda mask: f"attn_mask={mask}")
@pytest.mark.parametrize("groups", GROUP_ATTN_VALUES, ids=lambda g: f"gqa={g}" if g > 1 else "")
@pytest.mark.parametrize("scale", [None, 0.1], ids=lambda s: f"scale={s}" if s else "")
@pytest.mark.parametrize("dropout_p", [0.0, 0.5], ids=lambda p: f"dropout={p}")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("input_type", ["cross_attn", "self_attn"])
def test_unquantized_attn_zero_error(
    input_type: str,
    dropout_p: float,
    groups: int,
    device: torch.device,
    scale: float | None,
    use_attn_mask: bool | str,
) -> None:
    # GIVEN output of torch scaled-dot-product-attention MATH implementation as
    #        used as in attention layers
    q, k, v, attn_mask, is_causal = _make_attn_inputs(input_type, groups, use_attn_mask, device)
    kwargs = {"enable_gqa": groups > 1} if torch.__version__ >= "2.5" else {}
    with sdpa_kernel(backends=[SDPBackend.MATH]), torch.no_grad():
        torch.manual_seed(0)  # for dropout reproducibility
        out_torch = F.scaled_dot_product_attention(
            q, k, v, attn_mask, scale=scale, dropout_p=dropout_p, is_causal=is_causal, **kwargs
        )

    # WHEN fastforward version of scaled-dot-product-attention is
    #       executed over the same input tensors with no quantization
    qsdpa = _QuantizedSDPA(bits=None).to(device)
    with torch.no_grad():
        torch.manual_seed(0)  # for dropout reproducibility
        out_ff = qsdpa(
            q,
            k,
            v,
            attn_mask,
            scale=scale,
            dropout_p=dropout_p,
            is_causal=is_causal,
            enable_gqa=groups > 1,
            strict_quantization=False,
        )

    # THEN output error is exactly zero
    _print_abs_max_err(out_ff, out_torch, 0.0)
    assert torch.all(out_torch == out_ff)


# ------------------------------------------------------------------------------
# UPCAST DTYPE TESTS


@pytest.mark.skipif(torch.__version__ < "2.5", reason="Bit-exact tests require torch 2.5 or higher")
@pytest.mark.parametrize("groups", GROUP_ATTN_VALUES, ids=lambda g: f"gqa={g}" if g > 1 else "")
@pytest.mark.parametrize("use_attn_mask", _ATTN_MASK_OPTS, ids=lambda mask: f"attn_mask={mask}")
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16], ids=lambda dt: str(dt).split(".")[-1]
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("input_type", ["cross_attn", "self_attn"])
def test_unquantized_sdpa_implicitly_upcast_to_fp32(
    input_type: str,
    device: torch.device,
    dtype: torch.dtype,
    use_attn_mask: bool | str,
    groups: int,
) -> None:
    # GIVEN of torch scaled-dot-product-attention MATH implementation
    #        executed over lower-precision float tensors
    q, k, v, attn_mask, is_causal = _make_attn_inputs(
        input_type, groups, use_attn_mask, device, dtype
    )
    kwargs = {"enable_gqa": groups > 1} if torch.__version__ >= "2.5" else {}
    with sdpa_kernel(backends=[SDPBackend.MATH]), torch.no_grad():
        out_torch = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask,
            is_causal=is_causal,
            **kwargs,
        )

    # WHEN fastforward version of scaled-dot-product-attention is executed
    #       over the same input tensors with no quant and no explicit upcast
    qsdpa = _QuantizedSDPA(bits=None).to(device)
    with torch.no_grad():
        out_ff = qsdpa(
            q,
            k,
            v,
            attn_mask,
            is_causal=is_causal,
            enable_gqa=groups > 1,
            strict_quantization=False,
        )

    # THEN output error is exactly zero (ff and torch have same default upcast)
    _print_abs_max_err(out_ff, out_torch, 0.0)
    assert torch.all(out_torch == out_ff)


@pytest.mark.skipif(torch.__version__ < "2.5", reason="Bit-exact tests require torch 2.5 or higher")
@pytest.mark.parametrize("groups", GROUP_ATTN_VALUES, ids=lambda g: f"gqa={g}" if g > 1 else "")
@pytest.mark.parametrize("use_attn_mask", _ATTN_MASK_OPTS, ids=lambda mask: f"attn_mask={mask}")
@pytest.mark.parametrize(
    "upcast", [True, torch.float32], ids=lambda dt: f"upcast={str(dt).split('.')[-1]}"
)
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16], ids=lambda dt: str(dt).split(".")[-1]
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("input_type", ["cross_attn", "self_attn"])
def test_unquantized_sdpa_default_upcast_zero_error(
    input_type: str,
    device: torch.device,
    upcast: torch.dtype | bool | None,
    dtype: torch.dtype,
    use_attn_mask: bool | str,
    groups: int,
) -> None:
    # GIVEN output of torch scaled-dot-product-attention MATH implementation
    #        executed over lower-precision float tensors
    q, k, v, attn_mask, is_causal = _make_attn_inputs(
        input_type, groups, use_attn_mask, device, dtype
    )
    kwargs = {"enable_gqa": groups > 1} if torch.__version__ >= "2.5" else {}
    with sdpa_kernel(backends=[SDPBackend.MATH]), torch.no_grad():
        out_torch = F.scaled_dot_product_attention(
            q, k, v, attn_mask, is_causal=is_causal, **kwargs
        )

    # WHEN fastforward version of scaled-dot-product-attention is executed
    #       over the same input tensors with no quant and with explicit upcast
    qsdpa = _QuantizedSDPA(bits=None).to(device)
    with fastforward.sdpa_upcast(upcast), torch.no_grad():
        out_ff = qsdpa(
            q,
            k,
            v,
            attn_mask,
            is_causal=is_causal,
            enable_gqa=groups > 1,
            strict_quantization=False,
        )

    # THEN output error is exactly zero
    _print_abs_max_err(out_ff, out_torch, 0.0)
    assert torch.all(out_torch == out_ff)


@pytest.mark.skipif(torch.__version__ < "2.5", reason="Torch<2.5 do not upcast SDPA")
@pytest.mark.parametrize("groups", GROUP_ATTN_VALUES, ids=lambda g: f"gqa={g}" if g > 1 else "")
@pytest.mark.parametrize("use_attn_mask", _ATTN_MASK_OPTS, ids=lambda mask: f"attn_mask={mask}")
@pytest.mark.parametrize("upcast", [None, False], ids=lambda dt: f"upcast={str(dt).split('.')[-1]}")
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16], ids=lambda dt: str(dt).split(".")[-1]
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("input_type", ["cross_attn", "self_attn"])
def test_unquantized_sdpa_upcast_disabled_imply_small_error(
    input_type: str,
    device: torch.device,
    upcast: torch.dtype | bool | None,
    dtype: torch.dtype,
    use_attn_mask: bool | str,
    groups: int,
) -> None:
    # GIVEN output of torch scaled-dot-product-attention MATH implementation
    #        executed over lower-precision float tensors
    q, k, v, attn_mask, is_causal = _make_attn_inputs(
        input_type, groups, use_attn_mask, device, dtype
    )
    kwargs = {"enable_gqa": groups > 1} if torch.__version__ >= "2.5" else {}
    with sdpa_kernel(backends=[SDPBackend.MATH]), torch.no_grad():
        out_torch = F.scaled_dot_product_attention(
            q, k, v, attn_mask, is_causal=is_causal, **kwargs
        )

    # WHEN fastforward version of scaled-dot-product-attention is executed over
    #       over the same input tensors with no quant and disabling upcast
    qsdpa = _QuantizedSDPA(bits=None).to(device)
    with fastforward.sdpa_upcast(upcast), torch.no_grad():
        out_ff = qsdpa(
            q,
            k,
            v,
            attn_mask,
            is_causal=is_causal,
            enable_gqa=groups > 1,
            strict_quantization=False,
        )

    # THEN output error is always greater than 0 but below a tolerance
    _print_abs_max_err(out_ff, out_torch, 0.0)
    assert (out_torch - out_ff).abs().mean() > 0.0
    assert torch.allclose(out_torch, out_ff, atol=1e-2)


# ------------------------------------------------------------------------------
# STRICTLY QUANTIZED TESTS


@pytest.mark.parametrize(
    "bits, tol",
    [(16, 0.01), (8, 0.2)],
    ids=lambda bt: f"{bt}b" if isinstance(bt, int) else "",  # f"tol={bt}"
)
@pytest.mark.parametrize("strict_quant", [True, False], ids=lambda sq: "strict" if sq else "")
@pytest.mark.parametrize("use_attn_mask", _ATTN_MASK_OPTS, ids=lambda mask: f"attn_mask={mask}")
@pytest.mark.parametrize("groups", GROUP_ATTN_VALUES, ids=lambda g: f"gqa={g}" if g > 1 else "")
@pytest.mark.parametrize("scale", [None, 0.01], ids=lambda s: f"scale={s}" if s else "")
@pytest.mark.parametrize("dropout_p", [0.0, 0.5], ids=lambda p: f"dropout={p}")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("input_type", ["cross_attn", "self_attn"])
def test_quantized_attn(
    input_type: str,
    bits: int,
    tol: float,
    dropout_p: float,
    groups: int,
    device: torch.device,
    scale: float | None,
    use_attn_mask: bool | str,
    strict_quant: bool,
) -> None:
    # GIVEN The output of torch scaled-dot-product-attention MATH implementation
    #        with self-attention-like inputs
    q, k, v, attn_mask, is_causal = _make_attn_inputs(input_type, groups, use_attn_mask, device)
    kwargs = {"enable_gqa": groups > 1} if torch.__version__ >= "2.5" else {}
    with sdpa_kernel(backends=[SDPBackend.MATH]), torch.no_grad():
        torch.manual_seed(0)  # for dropout reproducibility
        out_torch = F.scaled_dot_product_attention(
            q, k, v, attn_mask, scale=scale, dropout_p=dropout_p, is_causal=is_causal, **kwargs
        )

    if groups > 1 and strict_quant:
        pytest.skip("GQA does not support strict-quantization")

    # WHEN quantized version of fastforward SDPA is executed on the same inputs
    qsdpa = _QuantizedSDPA(bits).to(device)
    with torch.no_grad():
        with estimate_ranges(qsdpa, range_setting.running_minmax):
            torch.manual_seed(0)  # for dropout reproducibility
            qsdpa(
                q,
                k,
                v,
                attn_mask,
                scale=scale,
                dropout_p=dropout_p,
                is_causal=is_causal,
                strict_quantization=False,
                neg_inf=-1000.0,
                **kwargs,
            )
        torch.manual_seed(0)  # for dropout reproducibility
        out_ff = qsdpa(
            q,
            k,
            v,
            attn_mask,
            scale=scale,
            dropout_p=dropout_p,
            is_causal=is_causal,
            strict_quantization=False,
            neg_inf=-1000.0,
            **kwargs,
        ).dequantize()

    # THEN output error is in tolerance and proportionate with bits
    _print_abs_max_err(out_ff, out_torch, tol)
    assert torch.allclose(out_torch, out_ff, atol=tol)


@pytest.mark.parametrize(
    "bits, tol",
    [(16, 1e-2), (8, 5e-2)],
    ids=lambda bt: f"{bt}b" if isinstance(bt, int) else f"tol={bt}",
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("strict_quant", [True, False], ids=lambda sq: "strict" if sq else "")
def test_diffusers_like_quantized_cross_attention(
    bits: int,
    tol: float,
    device: torch.device,
    strict_quant: bool,
) -> None:

    # GIVEN The output of torch scaled-dot-product-attention MATH implementation
    #         with diffusers-like cross-attention-like inputs
    q, k, v, attn_mask, is_causal = _make_diffusers_cross_attn_inputs(device=device)
    with sdpa_kernel(backends=[SDPBackend.MATH]), torch.no_grad():
        torch.manual_seed(0)  # for dropout reproducibility
        out_torch = F.scaled_dot_product_attention(q, k, v, attn_mask, is_causal=is_causal)

    # WHEN quantized version of fastforward SDPA is executed on the same inputs
    qsdpa = _QuantizedSDPA(bits).to(device)
    with torch.no_grad():
        with estimate_ranges(qsdpa, range_setting.running_minmax):
            torch.manual_seed(0)  # for dropout reproducibility
            qsdpa(q, k, v, attn_mask, is_causal=is_causal, strict_quantization=strict_quant)
        torch.manual_seed(0)  # for dropout reproducibility
        out_ff = qsdpa(
            q, k, v, attn_mask, is_causal=is_causal, strict_quantization=strict_quant
        ).dequantize()

    # THEN output error is in tolerance and proportionate with bits
    _print_abs_max_err(out_ff, out_torch, tol)
    assert_max_abs_err(out_torch, out_ff, atol=tol)


# ------------------------------------------------------------------------------
# PARTIALLY QUANTIZED TESTS

QUANTIZER_KEYS_BITS_TOL = (
    ("q", 8, 0.01),
    ("k", 8, 0.01),
    ("v", 8, 0.01),
    ("attn_scores", 8, 0.01),
    ("attn_mask", 8, 0.01),
    # masked_scores with causal mask is harder to quantize, we need more tolerance
    ("masked_scores", 8, 0.1),
    ("attn_weights", 8, 0.01),
    ("scaled_query", 8, 0.01),
    ("scaled_key", 8, 0.01),
    ("dropout", 8, 0.01),
    ("output", 8, 0.01),
)


@pytest.mark.parametrize("use_attn_mask", _ATTN_MASK_OPTS, ids=lambda mask: f"attn_mask={mask}")
@pytest.mark.parametrize("groups", GROUP_ATTN_VALUES, ids=lambda g: f"gqa={g}" if g > 1 else "")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("input_type", ["cross_attn", "self_attn"])
@pytest.mark.parametrize("quantize_key, bits, tol", QUANTIZER_KEYS_BITS_TOL)
def test_partially_quantized_attn(
    input_type: str,
    quantize_key: str,
    bits: int,
    tol: float,
    groups: int,
    device: torch.device,
    use_attn_mask: bool | str,
    scale: float | None = None,
    dropout_p: float = 0.0,
) -> None:
    # GIVEN The output of torch scaled-dot-product-attention MATH implementation
    #        with self-attention-like inputs
    q, k, v, attn_mask, is_causal = _make_attn_inputs(input_type, groups, use_attn_mask, device)
    kwargs = {"enable_gqa": groups > 1} if torch.__version__ >= "2.5" else {}
    with sdpa_kernel(backends=[SDPBackend.MATH]), torch.no_grad():
        torch.manual_seed(0)  # for dropout reproducibility
        out_torch = F.scaled_dot_product_attention(
            q, k, v, attn_mask, scale=scale, dropout_p=dropout_p, is_causal=is_causal, **kwargs
        )

    # WHEN quantized version of fastforward SDPA is executed on the same inputs,
    #       with only a single quantizer effectively instantiated
    qsdpa = _QuantizedSDPA(bits, quantizer_key=quantize_key).to(device)
    with torch.no_grad():
        with estimate_ranges(qsdpa, range_setting.running_minmax):
            torch.manual_seed(0)  # for dropout reproducibility
            qsdpa(
                q,
                k,
                v,
                attn_mask,
                scale=scale,
                dropout_p=dropout_p,
                is_causal=is_causal,
                enable_gqa=groups > 1,
                strict_quantization=False,
                neg_inf=-1000.0,
            )
        torch.manual_seed(0)  # for dropout reproducibility
        out_ff = qsdpa(
            q,
            k,
            v,
            attn_mask,
            scale=scale,
            dropout_p=dropout_p,
            is_causal=is_causal,
            enable_gqa=groups > 1,
            strict_quantization=False,
            neg_inf=-1000.0,
        ).dequantize()

    # THEN we have exactly 1 quantizer, and the output error is in tolerance
    #       and proportionate with the number of bits
    assert len(list(qsdpa.named_quantizers())) == 1
    _print_abs_max_err(out_ff, out_torch, tol)
    assert_max_abs_err(out_torch, out_ff, atol=tol)


# ------------------------------------------------------------------------------
# TORCH.COMPILE TESTS

# Importing torch._inductor on torch 2.11+ emits the torch.jit.script_method DeprecationWarning.
_IGNORE_SCRIPT_METHOD_DEPRECATION = pytest.mark.filterwarnings(
    "ignore:.*torch.jit.script_method. is deprecated:DeprecationWarning"
)


def _attention_results(
    attention: Callable[..., torch.Tensor],
    groups: int,
    use_attn_mask: bool | str,
    train: bool,
) -> tuple[torch.Tensor, ...]:
    """Return the output of `attention` and, in training, the query, key and value gradients."""
    torch.manual_seed(0)
    q, k, v, attn_mask, is_causal = _make_attn_inputs("self_attn", groups, use_attn_mask, "cpu")
    kwargs: dict[str, Any] = {"is_causal": is_causal, "enable_gqa": groups > 1}
    if not train:
        with torch.no_grad():
            return (attention(q, k, v, attn_mask, **kwargs),)

    q, k, v = (tensor.requires_grad_(True) for tensor in (q, k, v))
    out = attention(q, k, v, attn_mask, **kwargs)
    out.square().sum().backward()
    assert q.grad is not None and k.grad is not None and v.grad is not None
    return out.detach(), q.grad, k.grad, v.grad


def _attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None,
    **kwargs: Any,
) -> torch.Tensor:
    return FF.scaled_dot_product_attention(q, k, v, attn_mask, strict_quantization=False, **kwargs)


@pytest.mark.slow
@_IGNORE_SCRIPT_METHOD_DEPRECATION
@pytest.mark.parametrize("train", [False, True], ids=lambda t: "train" if t else "inference")
@pytest.mark.parametrize("use_attn_mask", _ATTN_MASK_OPTS, ids=lambda mask: f"attn_mask={mask}")
@pytest.mark.parametrize("groups", GROUP_ATTN_VALUES, ids=lambda g: f"gqa={g}" if g > 1 else "")
def test_compiled_attn_matches_eager(groups: int, use_attn_mask: bool | str, train: bool) -> None:
    # GIVEN the eager output (and gradients) of the MATH kernel, which is the one
    #       selected by default
    torch._dynamo.reset()
    expected = _attention_results(_attention, groups, use_attn_mask, train)

    # WHEN the same attention is compiled into a single graph, forward and backward
    compiled = torch.compile(_attention, backend="aot_eager", fullgraph=True)
    actual = _attention_results(compiled, groups, use_attn_mask, train)

    # THEN compilation succeeds without graph breaks and the results match eager
    for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
        assert_max_abs_err(expected_tensor, actual_tensor, atol=1e-6)


@pytest.mark.slow
@_IGNORE_SCRIPT_METHOD_DEPRECATION
# Torch 2.8 - 2.13 instantiate autograd.Function classes while dynamo traces them,
# emitting a DeprecationWarning.
@pytest.mark.filterwarnings("ignore:.*should not be instantiated:DeprecationWarning")
@pytest.mark.parametrize("train", [False, True], ids=lambda t: "train" if t else "inference")
@pytest.mark.parametrize("use_attn_mask", ["float", "causal"], ids=lambda m: f"attn_mask={m}")
@pytest.mark.parametrize("groups", GROUP_ATTN_VALUES, ids=lambda g: f"gqa={g}" if g > 1 else "")
def test_compiled_quantized_attn_matches_eager_in_qdq_mode(
    groups: int, use_attn_mask: bool | str, train: bool
) -> None:
    # GIVEN a quantized SDPA whose ranges were estimated eagerly, run in qdq mode so
    #       that quantizers return plain tensors, as `QuantizedTensor` cannot be compiled
    torch._dynamo.reset()
    qsdpa = _QuantizedSDPA(8)

    def attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
        **kwargs: Any,
    ) -> torch.Tensor:
        out: torch.Tensor = qsdpa(
            q, k, v, attn_mask, strict_quantization=False, neg_inf=-1000.0, **kwargs
        )
        return out

    with estimate_ranges(qsdpa, range_setting.running_minmax):
        _attention_results(attention, groups, use_attn_mask, train=False)

    with fastforward.qdq_mode(True):
        expected = _attention_results(attention, groups, use_attn_mask, train)

        # WHEN the same attention is compiled into a single graph, forward and backward
        compiled = torch.compile(attention, backend="aot_eager", fullgraph=True)
        actual = _attention_results(compiled, groups, use_attn_mask, train)

    # THEN compilation succeeds without graph breaks and the results match eager
    for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
        assert_max_abs_err(expected_tensor, actual_tensor, atol=1e-6)
