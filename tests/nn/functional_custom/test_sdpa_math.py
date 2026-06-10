# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Any, Iterable

import fastforward
import pytest
import torch

from fastforward import estimate_ranges, range_setting
from fastforward.nn import LinearQuantizer, QuantizedModule, QuantizerStub
from fastforward.nn import functional as FF
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

torch.backends.cudnn.deterministic = True

_ATTN_MASK_OPTS = [False, "float", "bool", "causal"]


# ------------------------------------------------------------------------------
# BIT-EXACT TESTS


@pytest.mark.parametrize("use_attn_mask", _ATTN_MASK_OPTS, ids=lambda mask: f"attn_mask={mask}")
@pytest.mark.parametrize("groups", [1, 4], ids=lambda g: f"gqa={g}" if g > 1 else "")
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
    # GIVEN: output of torch scaled-dot-product-attention MATH implementation as
    #        used as in attention layers
    q, k, v, attn_mask, is_causal = _make_attn_inputs(input_type, groups, use_attn_mask, device)
    with sdpa_kernel(backends=[SDPBackend.MATH]), torch.no_grad():
        torch.manual_seed(0)  # for dropout reproducibility
        out_torch = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask,
            scale=scale,
            dropout_p=dropout_p,
            is_causal=is_causal,
            enable_gqa=groups > 1,
        )

    # WHEN: fastforward version of scaled-dot-product-attention is
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

    # THEN: output error is exactly zero
    _print_abs_max_err(out_ff, out_torch, 0.0)
    assert torch.all(out_torch == out_ff)


# ------------------------------------------------------------------------------
# UPCAST DTYPE TESTS


@pytest.mark.parametrize("groups", [1, 4], ids=lambda g: f"gqa={g}" if g > 1 else "")
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
    # GIVEN: of torch scaled-dot-product-attention MATH implementation
    #        executed over lower-precision float tensors
    q, k, v, attn_mask, is_causal = _make_attn_inputs(
        input_type, groups, use_attn_mask, device, dtype
    )
    with sdpa_kernel(backends=[SDPBackend.MATH]), torch.no_grad():
        out_torch = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask,
            is_causal=is_causal,
            enable_gqa=groups > 1,
        )

    # WHEN: fastforward version of scaled-dot-product-attention is executed
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

    # THEN: output error is exactly zero (ff and torch have same default upcast)
    _print_abs_max_err(out_ff, out_torch, 0.0)
    assert torch.all(out_torch == out_ff)


@pytest.mark.parametrize("groups", [1, 4], ids=lambda g: f"gqa={g}" if g > 1 else "")
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
    # GIVEN: output of torch scaled-dot-product-attention MATH implementation
    #        executed over lower-precision float tensors
    q, k, v, attn_mask, is_causal = _make_attn_inputs(
        input_type, groups, use_attn_mask, device, dtype
    )
    with sdpa_kernel(backends=[SDPBackend.MATH]), torch.no_grad():
        out_torch = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask,
            is_causal=is_causal,
            enable_gqa=groups > 1,
        )

    # WHEN: fastforward version of scaled-dot-product-attention is executed
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

    # THEN: output error is exactly zero
    _print_abs_max_err(out_ff, out_torch, 0.0)
    assert torch.all(out_torch == out_ff)


@pytest.mark.parametrize("groups", [1, 4], ids=lambda g: f"gqa={g}" if g > 1 else "")
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
    # GIVEN: output of torch scaled-dot-product-attention MATH implementation
    #        executed over lower-precision float tensors
    q, k, v, attn_mask, is_causal = _make_attn_inputs(
        input_type, groups, use_attn_mask, device, dtype
    )
    with sdpa_kernel(backends=[SDPBackend.MATH]), torch.no_grad():
        out_torch = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask,
            is_causal=is_causal,
            enable_gqa=groups > 1,
        )

    # WHEN: fastforward version of scaled-dot-product-attention is executed over
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

    # THEN: output error is always greater than 0 but below a tolerance
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
@pytest.mark.parametrize("groups", [1, 4], ids=lambda g: f"gqa={g}" if g > 1 else "")
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

    # GIVEN: The output of torch scaled-dot-product-attention MATH implementation
    #        with self-attention-like inputs
    q, k, v, attn_mask, is_causal = _make_attn_inputs(input_type, groups, use_attn_mask, device)
    with sdpa_kernel(backends=[SDPBackend.MATH]), torch.no_grad():
        torch.manual_seed(0)  # for dropout reproducibility
        out_torch = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask,
            scale=scale,
            dropout_p=dropout_p,
            is_causal=is_causal,
            enable_gqa=groups > 1,
        )

    if groups > 1 and strict_quant:
        pytest.skip("GQA does not support strict-quantization")

    # WHEN: quantized version of fastforward SDPA is executed on the same inputs
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

    # THEN: output error is in tolerance and proportionate with bits
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

    # GIVEN: The output of torch scaled-dot-product-attention MATH implementation
    #         with diffusers-like cross-attention-like inputs
    q, k, v, attn_mask, is_causal = _make_diffusers_cross_attn_inputs(device=device)
    with sdpa_kernel(backends=[SDPBackend.MATH]), torch.no_grad():
        torch.manual_seed(0)  # for dropout reproducibility
        out_torch = F.scaled_dot_product_attention(q, k, v, attn_mask, is_causal=is_causal)

    # WHEN: quantized version of fastforward SDPA is executed on the same inputs
    qsdpa = _QuantizedSDPA(bits).to(device)
    with torch.no_grad():
        with estimate_ranges(qsdpa, range_setting.running_minmax):
            torch.manual_seed(0)  # for dropout reproducibility
            qsdpa(q, k, v, attn_mask, is_causal=is_causal, strict_quantization=strict_quant)
        torch.manual_seed(0)  # for dropout reproducibility
        out_ff = qsdpa(
            q, k, v, attn_mask, is_causal=is_causal, strict_quantization=strict_quant
        ).dequantize()

    # THEN: output error is in tolerance and proportionate with bits
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
@pytest.mark.parametrize("groups", [1, 4], ids=lambda g: f"gqa={g}" if g > 1 else "")
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

    # GIVEN: The output of torch scaled-dot-product-attention MATH implementation
    #        with self-attention-like inputs
    q, k, v, attn_mask, is_causal = _make_attn_inputs(input_type, groups, use_attn_mask, device)
    with sdpa_kernel(backends=[SDPBackend.MATH]), torch.no_grad():
        torch.manual_seed(0)  # for dropout reproducibility
        out_torch = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask,
            scale=scale,
            dropout_p=dropout_p,
            is_causal=is_causal,
            enable_gqa=groups > 1,
        )

    # WHEN: quantized version of fastforward SDPA is executed on the same inputs,
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

    # THEN: we have exactly 1 quantizer, and the output error is in tolerance
    #       and proportionate with the number of bits
    assert len(list(qsdpa.named_quantizers())) == 1
    _print_abs_max_err(out_ff, out_torch, tol)
    assert_max_abs_err(out_torch, out_ff, atol=tol)


# ------------------------------------------------------------------------------
# TEST UTILS


class _QuantizedSDPA(QuantizedModule):
    def __init__(self, bits: int | None, quantizer_key: str | None = None) -> None:
        super().__init__()
        self._bits = bits
        self._quantizer_key = quantizer_key

    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.q = QuantizerStub()
        self.k = QuantizerStub()
        self.v = QuantizerStub()
        self.attn_scores = QuantizerStub()
        self.attn_mask = QuantizerStub()
        self.masked_scores = QuantizerStub()
        self.attn_weights = QuantizerStub()
        self.scaled_query = QuantizerStub()
        self.scaled_key = QuantizerStub()
        self.dropout = QuantizerStub()
        self.output = QuantizerStub()

        if self._bits is not None:
            if self._quantizer_key is None:
                self._quantizer_key = "*"
            fastforward.find_quantizers(self, self._quantizer_key).initialize(
                LinearQuantizer, num_bits=self._bits, symmetric=False
            )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return FF.scaled_dot_product_attention(
            self.q(q),
            self.k(k),
            self.v(v),
            attn_mask=attn_mask,
            attn_scores_quantizer=self.attn_scores,
            attn_mask_quantizer=self.attn_mask,
            masked_scores_quantizer=self.masked_scores,
            attn_weights_quantizer=self.attn_weights,
            scaled_query_quantizer=self.scaled_query,
            scaled_key_quantizer=self.scaled_key,
            dropout_quantizer=self.dropout,
            output_quantizer=self.output,
            **kwargs,
        )


def _make_attn_inputs(
    attn_type: str,
    groups: int,
    use_attn_mask: bool | str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, bool]:
    if attn_type.lower() in ["cross_attn", "cross-attn", "cross"]:
        return _make_cross_attn_inputs(groups, use_attn_mask, device, dtype)
    elif attn_type.lower() in ["self_attn", "self-attn", "self"]:
        return _make_self_attn_inputs(groups, use_attn_mask, device, dtype)
    else:
        raise ValueError("attn_type should be `cross_attn` or `self_attn`.")


def _make_self_attn_inputs(
    groups: int,
    use_attn_mask: bool | str,
    device: torch.device | str = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, bool]:
    attn_mask: torch.Tensor | None
    is_causal = False
    if use_attn_mask == "causal":
        is_causal = True
        use_attn_mask = False
    q, k, v, attn_mask = _make_sdpa_input_tensors(
        N=2,
        H=4,
        S=4,
        L=4,
        E=3,
        gqa_groups=groups,
        bool_attn_mask=(use_attn_mask == "bool"),
        device=device,
        dtype=dtype,
    )
    attn_mask = attn_mask if use_attn_mask else None
    return q, k, v, attn_mask, is_causal


def _make_cross_attn_inputs(
    groups: int,
    use_attn_mask: bool | str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, bool]:
    attn_mask: torch.Tensor | None
    is_causal = False
    if use_attn_mask == "causal":
        is_causal = True
        use_attn_mask = False
    q, k, v, attn_mask = _make_sdpa_input_tensors(
        N=2,
        H=4,
        S=16,
        L=8,
        E=5,
        E_v=10,
        gqa_groups=groups,
        bool_attn_mask=(use_attn_mask == "bool"),
        device=device,
        dtype=dtype,
    )
    attn_mask = attn_mask if use_attn_mask else None
    return q, k, v, attn_mask, is_causal


def _make_diffusers_cross_attn_inputs(
    bs: int = 4,
    heads: int = 16,
    q_seqlen: int = 4096,
    kv_seqlen: int = 300,
    emb_dims: int = 72,
    groups: int = 1,
    device: torch.device | str = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, bool]:
    assert (bs >= 2) and (bs % 2 == 0)
    q, k, v, attn_mask = _make_sdpa_input_tensors(
        N=bs,
        H=heads,
        S=kv_seqlen,
        L=q_seqlen,
        E=emb_dims,
        gqa_groups=groups,
        bool_attn_mask=False,
        device=device,
        dtype=dtype,
    )

    attn_mask[:] = 0.0

    # Empty sentence for the first half of the batch (BoS token at position 0):
    attn_mask[: bs // 2, :, :, 1:] = -10000.0

    # End-of-sentence after kv_seqlen//2 tokens for the rest of the batch:
    attn_mask[bs // 2 :, :, :, kv_seqlen // 2 :] = -10000.0
    is_causal = False

    return q, k, v, attn_mask, is_causal


def _make_sdpa_input_tensors(
    N: int,
    H: int,
    S: int,
    L: int,
    E: int,
    E_v: int | None = None,
    gqa_groups: int = 1,
    *,
    extra_dims: Iterable[int] | None = None,
    bool_attn_mask: bool = False,
    device: torch.device | str = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate correct inputs for scaled dot product attention.

    Ref: https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.scaled_dot_product_attention.html
    Scaled dot product attention input tensors are:
        - query:        tensor with shape (N, ..., H_q, L, E)
        - key:          tensor with shape (N, ..., H,   S, E)
        - value:        tensor with shape (N, ..., H,   S, E_v)
        - attn_mask:    tensor with shape (N, ..., H_q, L, S)
                        (optional, boolean or query's dtype)

    NB:
        H_q: Number of heads of query is equal to H * gqa_groups

    Args:
        N: Batch size...:Any number of other batch dimensions (optional)
        H: Number of heads of key and value
        S: Source sequence length
        L: Target sequence length
        E: Embedding dimension of the query and key
        E_v: Embedding dimension of the value
        gqa_groups: number of grouped-query-attention groups
        extra_dims: extra dimension for all the tensors that will be placd after the batch size.
        bool_attn_mask: if True, the attention mask will be a boolean tensor instead of float.
        device: move the generated tensors to target device.
        dtype: specify the dtype for query/key/value and (non-bolean) attn_mask tensors

    """
    H_q = H * gqa_groups
    if E_v is None:
        E_v = E
    # Number of heads are all zero or all non-zero
    assert bool(H_q > 0) == bool(H > 0)
    extra_dims = [] if extra_dims is None else list(extra_dims)
    q_shape = [N, H_q] + extra_dims + [L, E]
    k_shape = [N, H] + extra_dims + [S, E]
    v_shape = [N, H] + extra_dims + [S, E_v]
    mask_shape = [N, H_q] + extra_dims + [L, S]

    # Remove empty dims
    nb_dims = len(q_shape)
    for dim in range(nb_dims - 1, -1, -1):
        for sh in (q_shape, k_shape, v_shape, mask_shape):
            if sh[dim] <= 0:
                sh.pop(dim)

    # Create tensors
    q = (torch.rand(*q_shape) - 0.5).to(device=device, dtype=dtype)
    k = (torch.rand(*k_shape) - 0.5).to(device=device, dtype=dtype)
    v = (torch.rand(*v_shape) - 0.5).to(device=device, dtype=dtype)
    attn_mask = (torch.rand(*mask_shape)).to(device=device, dtype=dtype)
    if bool_attn_mask:
        # transform to boolean: [-1, 0) -> False, [0, +1] -> True
        attn_mask = attn_mask >= 0.0

    return q, k, v, attn_mask


def _print_abs_max_err(out: torch.Tensor, target: torch.Tensor, tol: float) -> None:
    print()
    print(f"Abs-Max-Error: {(out - target).abs().max()}")
    print(f"TOL: {tol}")


def assert_max_abs_err(
    a: torch.Tensor,
    b: torch.Tensor,
    atol: float = 1e-08,
) -> None:
    max_abs_err = (a - b).abs().max()
    assert float(max_abs_err) < atol
