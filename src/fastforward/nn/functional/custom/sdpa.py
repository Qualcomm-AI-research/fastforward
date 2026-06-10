# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import math

from types import TracebackType

import torch

from fastforward._utils import classproperty
from fastforward.dispatcher import dispatch, register
from fastforward.exceptions import QuantizationError
from fastforward.flags import get_strict_quantization, set_strict_quantization
from fastforward.nn import functional as functional
from fastforward.nn.quantizer import Quantizer


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    *,
    neg_inf: float = float("-inf"),
    attn_scores_quantizer: Quantizer | None = None,
    attn_mask_quantizer: Quantizer | None = None,
    masked_scores_quantizer: Quantizer | None = None,
    attn_weights_quantizer: Quantizer | None = None,
    scaled_query_quantizer: Quantizer | None = None,
    scaled_key_quantizer: Quantizer | None = None,
    dropout_quantizer: Quantizer | None = None,
    output_quantizer: Quantizer | None = None,
    strict_quantization: bool | None = None,
    **kwargs: Quantizer | None,
) -> torch.Tensor:
    """Quantized version of torch.nn.functional.scaled_dot_product_attention."""
    if strict_quantization is None:
        strict_quantization = get_strict_quantization()

    dispatch_op = dispatch(
        "scaled_dot_product_attention",
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
        neg_inf=neg_inf,
        attn_scores_quantizer=attn_scores_quantizer,
        attn_mask_quantizer=attn_mask_quantizer,
        masked_scores_quantizer=masked_scores_quantizer,
        attn_weights_quantizer=attn_weights_quantizer,
        scaled_query_quantizer=scaled_query_quantizer,
        scaled_key_quantizer=scaled_key_quantizer,
        dropout_quantizer=dropout_quantizer,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
        **kwargs,
    )
    selected_op = dispatch_op
    assert selected_op is not None

    return selected_op(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
        neg_inf=neg_inf,
        attn_scores_quantizer=attn_scores_quantizer,
        attn_mask_quantizer=attn_mask_quantizer,
        masked_scores_quantizer=masked_scores_quantizer,
        attn_weights_quantizer=attn_weights_quantizer,
        scaled_query_quantizer=scaled_query_quantizer,
        scaled_key_quantizer=scaled_key_quantizer,
        dropout_quantizer=dropout_quantizer,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
        **kwargs,
    )


@register("scaled_dot_product_attention", None)  # type: ignore[arg-type]
def scaled_dot_product_attention_math(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    *,
    neg_inf: float = float("-inf"),
    attn_scores_quantizer: Quantizer | None = None,
    attn_mask_quantizer: Quantizer | None = None,
    masked_scores_quantizer: Quantizer | None = None,
    attn_weights_quantizer: Quantizer | None = None,
    scaled_query_quantizer: Quantizer | None = None,
    scaled_key_quantizer: Quantizer | None = None,
    dropout_quantizer: Quantizer | None = None,
    output_quantizer: Quantizer | None = None,
    strict_quantization: bool | None = None,
) -> torch.Tensor:
    """Quantized version of scaled_dot_product_attention ATEN MATH implementation.

    References:
      - https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
      - _scaled_dot_product_attention_math function defined in:
        https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/transformers/attention.cpp
        https://github.com/pytorch/pytorch/blob/cbf5bf0c4812c31268d1297700c342c6382c3670/aten/src/ATen/native/transformers/attention.cpp#L826

    Shapes:
        N: Batch size (any number of other batch dimensions is allowed).
        H: Number of heads of key and value.
        H_q: Number of heads of query (equal to H unless enable_gqa is True).
        S: Source sequence length.
        L: Target sequence length.
        E: Embedding dimension of the query and key.
        E_v: Embedding dimension of the value.

    Args:
        query: Query tensor with shape (N, ..., H_q, L, E).
        key: Key tensor with shape (N, ..., H, S, E).
        value: Value tensor with shape (N, ..., H, S, E_v).
        attn_mask: Attention mask with shape broadcastable to (N, ..., H_q, L, S).
            A boolean mask is converted to an additive bias where `False` positions
            become `-inf`; a float mask is added directly to the attention scores.
            Must be `None` when `is_causal` is True.
        dropout_p: Dropout probability applied to the attention weights.
        is_causal: If True, apply an upper-triangular causal mask internally. Mutually
            exclusive with `attn_mask`.
        scale: Scaling factor applied to the QK product. If None, defaults to
            `1 / sqrt(E)`.
        enable_gqa: If `True`, enable grouped-query attention by repeating key and
            value along the head dimension to match `H_q`. Not supported under
            strict quantization.
        neg_inf: the float value representing "negative infinity" in the attention mask.
            If the masked_scores_quantizer is instantiated, tuning `neg_inf` could
            be essential to reach a decent accuracy.
        attn_scores_quantizer: Quantizer applied to the raw attention scores
            (`Q @ K^T`) before the softmax.
        attn_mask_quantizer: Quantizer applied to `attn_mask` (if `attn_mask` is
            a float tensor) or to a mask tensor with values in {-inf, 0} generated
            from the `attn_mask` (if `attn_mask` is a bool tensor, or
            `attn_mask=None` and `causal_mask=True`).
        masked_scores_quantizer: Quantizer applied to the output of the addition
            between the attention bias (computed from `attn_mask` or causal mask)
            and attention scores.
        attn_weights_quantizer: Quantizer applied to the attention weights after
            the safe softmax.
        scaled_query_quantizer: Quantizer applied to the query after multiplying
            by `sqrt(scale)`.
        scaled_key_quantizer: Quantizer applied to the transposed key after
            multiplying by `sqrt(scale)`.
        dropout_quantizer: Quantizer applied to the attention weights after dropout.
        output_quantizer: Quantizer applied to the final attention output
            (`attn_weight @ V`).
        strict_quantization: If None, the value is read from the global
            `get_strict_quantization()` flag. When True, every operation must
            produce quantized tensors; this is incompatible with `enable_gqa=True`.

    Returns:
        Attention output tensor with shape (N, ..., H_q, L, E_v) and the same
        dtype as the input `query`.

    Raises:
        ValueError: If `attn_mask` is provided together with `is_causal=True`.
        QuantizationError: If `enable_gqa=True` is used under strict quantization.
    """
    L, S = query.size(-2), key.size(-2)

    if strict_quantization is None:
        strict_quantization = get_strict_quantization()

    # Due to activation outliers, when float16 is used Q @ K matmul is
    # very sensitive to overflow, leading to inf values (NaNs later on).
    # So when float16 is used we move to float32 for this operation.
    orig_dtype = query.dtype
    if sdpa_upcast.dtype is not None:
        if query.dtype == torch.float16 or query.dtype == torch.bfloat16:
            query = query.to(sdpa_upcast.dtype)
            key = key.to(sdpa_upcast.dtype)
            value = value.to(sdpa_upcast.dtype)

    if enable_gqa:
        if strict_quantization:
            msg = "Strict quantization currently not supported when enable_gqa=True"
            raise QuantizationError(msg)
        with set_strict_quantization(False):
            key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    # Scale q, k before matmul for stability see https://tinyurl.com/sudb9s96 for math
    scale_factor = 1.0 / math.sqrt(query.size(-1)) if scale is None else scale
    scale_factor_sqrt = math.sqrt(scale_factor)
    query = functional.mul(
        query,
        scale_factor_sqrt,
        output_quantizer=scaled_query_quantizer,
        strict_quantization=strict_quantization,
    )

    key = functional.mul(
        key.transpose(-2, -1),
        scale_factor_sqrt,
        output_quantizer=scaled_key_quantizer,
        strict_quantization=strict_quantization,
    )

    attn_scores = functional.matmul(
        query, key, output_quantizer=attn_scores_quantizer, strict_quantization=strict_quantization
    )

    attn_mask_bias = _get_quantized_attn_bias(
        attn_mask,
        is_causal,
        L,
        S,
        query.device,
        query.dtype,
        neg_inf=neg_inf,
        output_quantizer=attn_mask_quantizer,
    )

    masked_attn_scores = functional.add(
        attn_scores,
        attn_mask_bias,
        output_quantizer=masked_scores_quantizer,
        strict_quantization=strict_quantization,
    )

    attn_weight = _quantized_safe_softmax(
        masked_attn_scores, dim=-1, output_quantizer=attn_weights_quantizer, neg_inf=neg_inf
    )

    attn_weight = functional.dropout(
        attn_weight,
        dropout_p,
        training=True,
        output_quantizer=dropout_quantizer,
        strict_quantization=strict_quantization,
    )

    attn_out = functional.matmul(
        attn_weight,
        value,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )

    attn_out = attn_out.to(orig_dtype)
    return attn_out


def _get_quantized_attn_bias(
    attn_mask: torch.Tensor | None,
    is_causal: bool,
    L: int,
    S: int,
    device: torch.device | str,
    dtype: torch.dtype,
    *,
    neg_inf: float = float("-inf"),
    output_quantizer: Quantizer | None = None,
) -> torch.Tensor:

    if attn_mask is not None and is_causal:
        # Raise the same error raised by torch in the same situation
        msg = "Explicit attn_mask should not be set when is_causal=True"
        raise ValueError(msg)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias = torch.zeros_like(attn_mask, dtype=dtype, device=device)
            attn_bias.masked_fill_(attn_mask.logical_not(), neg_inf)
        else:
            attn_bias = attn_mask

    elif is_causal:
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=device).tril(diagonal=0)
        attn_bias = torch.zeros(L, S, dtype=dtype, device=device)
        attn_bias.masked_fill_(temp_mask.logical_not(), neg_inf)
        del temp_mask

    else:
        attn_bias = torch.zeros(L, S, dtype=dtype, device=device)

    if output_quantizer is not None:
        attn_bias = output_quantizer(attn_bias)
    return attn_bias


def _quantized_safe_softmax(
    t: torch.Tensor,
    dim: int,
    dtype: torch.dtype | None = None,
    output_quantizer: Quantizer | None = None,
    neg_inf: float = float("-inf"),
) -> torch.Tensor:
    """Quantized version of safe softmax for SDPA converted from ATEN.

    Reference:
        https://github.com/pytorch/pytorch/blob/a177d1852953f460cc0f0d412f311875f77dd4de/aten/src/ATen/native/transformers/attention.cpp#L672
    """
    with set_strict_quantization(False):
        out = functional.softmax(t, dim, dtype)
        if neg_inf == float("-inf"):
            masked = t.isneginf()
        else:
            masked = t <= neg_inf
        masked_rows = torch.all(masked, dim=dim, keepdim=True)
        zero = torch.tensor(0.0, dtype=out.dtype, device=out.device)
        torch.where(condition=masked_rows, input=zero, other=out, out=out)
        if output_quantizer:
            out = output_quantizer(out)
    return out


class _SDPAUpcast:
    """Context Manager that can be used to select the upcast dtype for fastforward SDPA Math kernel.

    By default, torch scaled-dot-product-attention Math implementations upcast
    to torch.float32 to avoid accumulation of errors.

    Fastforward Math implementation does the same, but offer this context manager
    to change this behavior.
    """

    __DEFAULT_UPCAST_DTYPE: torch.dtype = torch.float32
    _DTYPE: torch.dtype | None = __DEFAULT_UPCAST_DTYPE

    def __init__(self, dtype: torch.dtype | bool | None):
        self._orig_dtype = _SDPAUpcast._DTYPE
        self._dtype: torch.dtype | None

        if isinstance(dtype, torch.dtype):
            # Set custom upcast dtype
            self._dtype = dtype
        elif dtype:
            # when dtype==True, set default upcast dtype
            self._dtype = _SDPAUpcast.__DEFAULT_UPCAST_DTYPE
        else:
            # when dtype==False or None, disble upcasting
            self._dtype = None

    def __enter__(self) -> None:
        _SDPAUpcast._DTYPE = self._dtype

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        _SDPAUpcast._DTYPE = self._orig_dtype

    @classproperty
    def dtype(cls) -> torch.dtype | None:
        return cls._DTYPE

    @classmethod
    def upcast(cls, t: torch.Tensor) -> torch.Tensor:
        if cls.dtype:
            return t.to(cls.dtype)
        return t


sdpa_upcast = _SDPAUpcast
