# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import logging
import math

from types import TracebackType
from typing import Any, TypedDict

import torch

from fastforward._utils import classproperty
from fastforward.dispatcher import Predicate, dispatch, register
from fastforward.exceptions import QuantizationError
from fastforward.flags import get_sdpa_torch_fallback_allowed, get_strict_quantization
from fastforward.flags import strict_quantization as strict_quantization_ctx
from fastforward.nn import functional as functional
from fastforward.nn.quantizer import Quantizer, QuantizerStub

logger = logging.getLogger(__name__)

_EXPLICIT_MASK_WITH_CAUSAL_MSG = "Explicit attn_mask should not be set when is_causal=True"
_GQA_STRICT_QUANTIZATION_MSG = "Strict quantization currently not supported when enable_gqa=True"
_HEAD_MISMATCH_MSG = (
    "The number of heads of query, key, value and attn_mask must be equal or 1 "
    "(or key and value must be repeated with enable_gqa=True), got {head_sizes}"
)


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
    sdpa_torch_fallback: bool | None = None,
    **kwargs: Quantizer | None,
) -> torch.Tensor:
    """Quantized version of torch.nn.functional.scaled_dot_product_attention."""
    if strict_quantization is None:
        strict_quantization = get_strict_quantization()
    if sdpa_torch_fallback is None:
        sdpa_torch_fallback = get_sdpa_torch_fallback_allowed()

    if not strict_quantization and sdpa_torch_fallback:
        # If the sdpa_torch_fallback_allowed flag is True and no quantizers are active
        # we will just run torch scaled_dot_product_attention function (faster).
        _all_quantizers = [
            attn_scores_quantizer,
            attn_mask_quantizer,
            masked_scores_quantizer,
            attn_weights_quantizer,
            scaled_query_quantizer,
            scaled_key_quantizer,
            dropout_quantizer,
            output_quantizer,
        ] + [q for q in kwargs.values() if isinstance(q, Quantizer)]

        if not any([_is_quantizer_active(q) for q in _all_quantizers]):
            return _torch_sdpa_fallback(
                query=query,
                key=key,
                value=value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )

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

    assert dispatch_op is not None
    return dispatch_op(
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


def _math_sdpa_predicate(*_args: Any, **_kwargs: Any) -> bool:
    """Select the MATH kernel unless the memory-efficient kernel is enabled."""
    return not sdpa_mem_efficient.enabled


@register("scaled_dot_product_attention", Predicate(_math_sdpa_predicate))
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
    S = key.size(-2)

    if strict_quantization is None:
        strict_quantization = get_strict_quantization()

    query, key, value, orig_dtype = _prepare_sdpa_operands(
        query,
        key,
        value,
        scale=scale,
        enable_gqa=enable_gqa,
        strict_quantization=strict_quantization,
        scaled_query_quantizer=scaled_query_quantizer,
        scaled_key_quantizer=scaled_key_quantizer,
    )

    return _attention_for_query_chunk(
        query,
        key,
        value,
        attn_mask,
        S=S,
        is_causal=is_causal,
        causal_diagonal=0,
        dropout_p=dropout_p,
        neg_inf=neg_inf,
        attn_scores_quantizer=attn_scores_quantizer,
        attn_mask_quantizer=attn_mask_quantizer,
        masked_scores_quantizer=masked_scores_quantizer,
        attn_weights_quantizer=attn_weights_quantizer,
        dropout_quantizer=dropout_quantizer,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
        orig_dtype=orig_dtype,
    )


def _mem_efficient_sdpa_predicate(*_args: Any, **_kwargs: Any) -> bool:
    """Select the memory-efficient kernel when it is enabled."""
    return bool(sdpa_mem_efficient.enabled)


@register("scaled_dot_product_attention", Predicate(_mem_efficient_sdpa_predicate))
def scaled_dot_product_attention_mem_efficient(
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
    """Memory-efficient quantized version of scaled_dot_product_attention.

    Computes the same result as `scaled_dot_product_attention_math` without
    materializing the full (N, ..., H_q, L, S) attention matrix: attention is
    computed per (head group, query chunk) pair, with sizes taken from
    `sdpa_mem_efficient`. Chunks hold complete attention rows, so quantizers
    observe the same values as in the math kernel, with two exceptions: range
    estimators that fit parameters to a full tensor (e.g. MSE) observe one chunk
    at a time, and dropout masks are drawn per chunk.
    Under `enable_gqa=True`, key and value are repeated once before chunking.
    Chunks are gradient checkpointed by default (see `sdpa_mem_efficient`).

    Arguments and return value are as in `scaled_dot_product_attention_math`.
    Enable this kernel with the `sdpa_mem_efficient` context manager:

    ```python
    with fastforward.sdpa_mem_efficient(query_chunk_size=512):
        output = model(data)
    ```

    Raises:
        ValueError: If `attn_mask` is provided together with `is_causal=True`.
        QuantizationError: If `enable_gqa=True` is used under strict quantization.
        NotImplementedError: If a quantized `query` cannot be split into chunks
            because its granularity is not supported by `QuantizedTensor.__getitem__`.
    """
    L, S = query.size(-2), key.size(-2)
    # Dim -3 is the head dimension.
    # It broadcasts across query, key, value and mask: the number of heads is the largest of them.
    # If none has one (all 2-d), there are no heads to chunk.
    head_sizes = [
        t.size(-3) for t in (query, key, value, attn_mask) if t is not None and t.ndim >= 3
    ]
    has_heads = bool(head_sizes)
    H_q = max(head_sizes, default=1)

    query_chunk_size = sdpa_mem_efficient.query_chunk_size or L
    head_chunk_size = sdpa_mem_efficient.head_chunk_size or H_q

    if L == 0 or (L <= query_chunk_size and H_q <= head_chunk_size):
        # A single chunk, or no query positions to chunk: delegate the math kernel.
        return scaled_dot_product_attention_math(
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
        )

    if strict_quantization is None:
        strict_quantization = get_strict_quantization()

    query, key, value, orig_dtype = _prepare_sdpa_operands(
        query,
        key,
        value,
        scale=scale,
        enable_gqa=enable_gqa,
        strict_quantization=strict_quantization,
        scaled_query_quantizer=scaled_query_quantizer,
        scaled_key_quantizer=scaled_key_quantizer,
    )

    # Chunking would silently broadcast mismatched head counts that the math kernel rejects.
    # Checked after `_prepare_sdpa_operands`, which repeats key and value for `enable_gqa`.
    operand_head_sizes = [
        t.size(-3) for t in (query, key, value, attn_mask) if t is not None and t.ndim >= 3
    ]
    if any(size not in (1, H_q) for size in operand_head_sizes):
        msg = _HEAD_MISMATCH_MSG.format(head_sizes=operand_head_sizes)
        raise RuntimeError(msg)

    quantizers = (
        attn_scores_quantizer,
        attn_mask_quantizer,
        masked_scores_quantizer,
        attn_weights_quantizer,
        dropout_quantizer,
        output_quantizer,
    )
    use_checkpoint = _use_gradient_checkpointing((query, key, value, attn_mask), quantizers)

    shared_kwargs = _QueryChunkKwargs(
        S=S,
        is_causal=is_causal,
        dropout_p=dropout_p,
        neg_inf=neg_inf,
        attn_scores_quantizer=attn_scores_quantizer,
        attn_mask_quantizer=attn_mask_quantizer,
        masked_scores_quantizer=masked_scores_quantizer,
        attn_weights_quantizer=attn_weights_quantizer,
        dropout_quantizer=dropout_quantizer,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
        orig_dtype=orig_dtype,
    )

    head_outputs: list[torch.Tensor] = []
    for h_start in range(0, H_q, head_chunk_size):
        h_stop = min(h_start + head_chunk_size, H_q)
        query_heads = _slice_heads(query, h_start, h_stop)
        key_heads = _slice_heads(key, h_start, h_stop)
        value_heads = _slice_heads(value, h_start, h_stop)
        mask_heads = _slice_attn_mask(attn_mask, -3, h_start, h_stop)

        chunk_outputs: list[torch.Tensor] = []
        for c_start in range(0, L, query_chunk_size):
            c_stop = min(c_start + query_chunk_size, L)
            query_chunk = query_heads[..., c_start:c_stop, :]
            mask_chunk = _slice_attn_mask(mask_heads, -2, c_start, c_stop)

            attn_args = (query_chunk, key_heads, value_heads, mask_chunk)
            if use_checkpoint:
                attn_out = torch.utils.checkpoint.checkpoint(
                    _attention_for_query_chunk,
                    *attn_args,
                    use_reentrant=False,
                    causal_diagonal=c_start,
                    **shared_kwargs,
                )
            else:
                attn_out = _attention_for_query_chunk(
                    *attn_args,
                    causal_diagonal=c_start,
                    **shared_kwargs,
                )
            chunk_outputs.append(attn_out)
        head_outputs.append(torch.cat(chunk_outputs, dim=-2))

    return torch.cat(head_outputs, dim=-3) if has_heads else head_outputs[0]


class _QueryChunkKwargs(TypedDict):
    """Keyword arguments of `_attention_for_query_chunk` shared by all query chunks."""

    S: int
    is_causal: bool
    dropout_p: float
    neg_inf: float
    attn_scores_quantizer: Quantizer | None
    attn_mask_quantizer: Quantizer | None
    masked_scores_quantizer: Quantizer | None
    attn_weights_quantizer: Quantizer | None
    dropout_quantizer: Quantizer | None
    output_quantizer: Quantizer | None
    strict_quantization: bool
    orig_dtype: torch.dtype


def _attention_for_query_chunk(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
    *,
    S: int,
    is_causal: bool,
    causal_diagonal: int,
    dropout_p: float,
    neg_inf: float,
    attn_scores_quantizer: Quantizer | None,
    attn_mask_quantizer: Quantizer | None,
    masked_scores_quantizer: Quantizer | None,
    attn_weights_quantizer: Quantizer | None,
    dropout_quantizer: Quantizer | None,
    output_quantizer: Quantizer | None,
    strict_quantization: bool,
    orig_dtype: torch.dtype,
) -> torch.Tensor:
    """Attend a chunk of query positions to the full key and value sequences.

    Performs the quantized operations of the MATH kernel for the target positions in
    `query`. `query`, `key` and `value` are as returned by `_prepare_sdpa_operands`
    (`key` is scaled and transposed); `causal_diagonal` is the offset of the chunk
    within the full target sequence.
    """
    attn_scores = functional.matmul(
        query, key, output_quantizer=attn_scores_quantizer, strict_quantization=strict_quantization
    )

    attn_mask_bias = _get_quantized_attn_bias(
        attn_mask,
        is_causal,
        query.size(-2),
        S,
        query.device,
        query.dtype,
        neg_inf=neg_inf,
        output_quantizer=attn_mask_quantizer,
        causal_diagonal=causal_diagonal,
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

    return attn_out.to(orig_dtype)


def _slice_heads(tensor: torch.Tensor, start: int, stop: int) -> torch.Tensor:
    """Restrict a query, key or value to heads `[start, stop)`, unless it broadcasts over heads.

    A tensor without a head dimension (2-d) or with a single head is shared by all
    heads.
    """
    if tensor.ndim < 3 or tensor.size(-3) == 1:
        return tensor
    return tensor[..., start:stop, :, :]


def _slice_attn_mask(
    attn_mask: torch.Tensor | None, dim: int, start: int, stop: int
) -> torch.Tensor | None:
    """Restrict `attn_mask` to `[start, stop)` along `dim`, unless it broadcasts there."""
    if attn_mask is None or attn_mask.ndim < -dim or attn_mask.size(dim) == 1:
        return attn_mask
    return attn_mask.narrow(dim, start, stop - start)


def _use_gradient_checkpointing(
    tensors: tuple[torch.Tensor | None, ...], quantizers: tuple[Quantizer | None, ...]
) -> bool:
    """Return True if chunks should be gradient checkpointed.

    Only when enabled and a backward pass can reach `tensors` or the parameters of
    `quantizers` (`torch.utils.checkpoint` warns when no input requires grad).

    Note that a checkpointed chunk runs its quantizers again when the backward pass
    recomputes it. If a range estimation context is still active at that point, the
    range estimator observes the chunk a second time. Running min/max estimation is
    unaffected, since it is idempotent, but estimators that accumulate state may not be.
    """
    if not sdpa_mem_efficient.checkpoint or not torch.is_grad_enabled():
        return False
    if any(tensor is not None and tensor.requires_grad for tensor in tensors):
        return True
    return any(
        parameter.requires_grad
        for quantizer in quantizers
        if quantizer is not None
        for parameter in quantizer.parameters()
    )


def _prepare_sdpa_operands(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float | None,
    enable_gqa: bool,
    strict_quantization: bool,
    scaled_query_quantizer: Quantizer | None,
    scaled_key_quantizer: Quantizer | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.dtype]:
    """Upcast, expand for grouped-query attention and fold `scale` into query and key.

    These steps do not depend on the target sequence dimension, so chunked kernels
    perform them once.

    Returns:
        The scaled query, the scaled and transposed key, the value and the original
        dtype of `query`.
    """
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
            raise QuantizationError(_GQA_STRICT_QUANTIZATION_MSG)
        with strict_quantization_ctx(False):
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

    return query, key, value, orig_dtype


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
    causal_diagonal: int = 0,
) -> torch.Tensor:
    """Build the additive attention bias for `L` target and `S` source positions.

    `causal_diagonal` is the offset of the `L` target positions within the full
    target sequence (non-zero only when the query is processed in chunks).
    """
    if attn_mask is not None and is_causal:
        # Raise the same error raised by torch in the same situation
        raise ValueError(_EXPLICIT_MASK_WITH_CAUSAL_MSG)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias = torch.zeros_like(attn_mask, dtype=dtype, device=device)
            attn_bias.masked_fill_(attn_mask.logical_not(), neg_inf)
        else:
            attn_bias = attn_mask

    elif is_causal:
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=device).tril(diagonal=causal_diagonal)
        attn_bias = torch.zeros(L, S, dtype=dtype, device=device)
        attn_bias.masked_fill_(temp_mask.logical_not(), neg_inf)
        del temp_mask

    elif output_quantizer is not None:
        # Quantizers may not accept a 0-dim tensor (e.g. per-channel granularity), so
        # the zero bias is materialized when it has to be quantized.
        attn_bias = torch.zeros(L, S, dtype=dtype, device=device)

    else:
        # A 0-dim zero broadcasts identically to an (L, S) zero in the `functional.add`
        # below, so this avoids materializing an (L, S) tensor whose value never
        # affects the result.
        attn_bias = torch.zeros((), dtype=dtype, device=device)

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
    with strict_quantization_ctx(False):
        out = functional.softmax(t, dim, dtype)
        if neg_inf == float("-inf"):
            masked = t.isneginf()
        else:
            masked = t <= neg_inf
        masked_rows = torch.all(masked, dim=dim, keepdim=True)
        zero = out.new_tensor(0.0)
        if out.requires_grad:
            # `torch.where` does not accept `out=` when it has to be differentiated.
            out = torch.where(condition=masked_rows, input=zero, other=out)
        else:
            # Zero the fully masked rows in place to avoid allocating a second
            # tensor of the size of the attention scores.
            torch.where(condition=masked_rows, input=zero, other=out, out=out)
        if output_quantizer:
            out = output_quantizer(out)
    return out


def _is_quantizer_active(quantizer: Quantizer | None) -> bool:
    """Return true only if the quantizer is not None and is not a Stub."""
    return quantizer is not None and not isinstance(quantizer, QuantizerStub)


def _torch_sdpa_fallback(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:

    # torch < 2.5 does not support enable_gqa argument
    if torch.__version__ < "2.5":
        if enable_gqa:
            msg = (
                "Cannot fallback to Torch scaled_dot_product_attention because the installed "
                "version of torch does not support argument `enable_gqa` that was set to true. "
                "Fastforward scaled_dot_product_attention will be used instead."
            )
            logger.warning(msg)
            return scaled_dot_product_attention_math(
                query=query,
                key=key,
                value=value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )
        return torch.nn.functional.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )

    return torch.nn.functional.scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )


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


class _SDPAMemEfficient:
    """Context manager that selects the memory-efficient FastForward SDPA kernel.

    Within the context, `scaled_dot_product_attention` dispatches to
    `scaled_dot_product_attention_mem_efficient`, which attends the query in chunks
    of `query_chunk_size` positions and attention heads in groups of
    `head_chunk_size`. Intermediate attention tensors are bounded by
    (N, ..., head_chunk_size, query_chunk_size, S) elements. By default only heads
    are chunked (one at a time), which keeps memory accesses contiguous.

    Example:
        ```python
        with fastforward.sdpa_mem_efficient(query_chunk_size=512):
            output = model(data)
        ```

    Args:
        enabled: Whether the memory-efficient kernel is used within the context.
        query_chunk_size: Query positions attended per chunk. If None, the current
            value is kept (default None: the whole query in one chunk).
        head_chunk_size: Attention heads processed per group. If None, the current
            value is kept (default 1: one head at a time).
        checkpoint: Whether each chunk is wrapped in a gradient checkpoint, trading
            recomputation in the backward pass for memory. If None, the current
            value is kept. Without checkpointing, a forward pass with more than one
            chunk inside an active `estimate_ranges` context is not supported: each
            chunk updates the shared quantization parameters, so the backward pass
            raises an error. Keep checkpointing enabled for that case.
    """

    __DEFAULT_QUERY_CHUNK_SIZE: int | None = None
    __DEFAULT_HEAD_CHUNK_SIZE: int | None = 1

    _ENABLED: bool = False
    _QUERY_CHUNK_SIZE: int | None = __DEFAULT_QUERY_CHUNK_SIZE
    _HEAD_CHUNK_SIZE: int | None = __DEFAULT_HEAD_CHUNK_SIZE
    _CHECKPOINT: bool = True

    def __init__(
        self,
        enabled: bool = True,
        *,
        query_chunk_size: int | None = None,
        head_chunk_size: int | None = None,
        checkpoint: bool | None = None,
    ) -> None:
        if query_chunk_size is not None and query_chunk_size < 1:
            msg = f"`query_chunk_size` must be a positive integer, but got {query_chunk_size}"
            raise ValueError(msg)
        if head_chunk_size is not None and head_chunk_size < 1:
            msg = f"`head_chunk_size` must be a positive integer, but got {head_chunk_size}"
            raise ValueError(msg)

        self._enabled = enabled
        self._query_chunk_size = query_chunk_size
        self._head_chunk_size = head_chunk_size
        self._checkpoint = checkpoint
        # A stack so the same instance can be re-entered, including nested.
        self._orig_settings: list[tuple[bool, int | None, int | None, bool]] = []

    def __enter__(self) -> None:
        cls = _SDPAMemEfficient
        self._orig_settings.append((
            cls._ENABLED,
            cls._QUERY_CHUNK_SIZE,
            cls._HEAD_CHUNK_SIZE,
            cls._CHECKPOINT,
        ))
        cls._ENABLED = self._enabled
        if self._query_chunk_size is not None:
            cls._QUERY_CHUNK_SIZE = self._query_chunk_size
        if self._head_chunk_size is not None:
            cls._HEAD_CHUNK_SIZE = self._head_chunk_size
        if self._checkpoint is not None:
            cls._CHECKPOINT = self._checkpoint

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        cls = _SDPAMemEfficient
        cls._ENABLED, cls._QUERY_CHUNK_SIZE, cls._HEAD_CHUNK_SIZE, cls._CHECKPOINT = (
            self._orig_settings.pop()
        )

    @classproperty
    def enabled(cls) -> bool:
        return cls._ENABLED

    @classproperty
    def query_chunk_size(cls) -> int | None:
        return cls._QUERY_CHUNK_SIZE

    @classproperty
    def head_chunk_size(cls) -> int | None:
        return cls._HEAD_CHUNK_SIZE

    @classproperty
    def checkpoint(cls) -> bool:
        return cls._CHECKPOINT


sdpa_mem_efficient = _SDPAMemEfficient
