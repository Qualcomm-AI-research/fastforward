# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Iterable

import torch


def _make_attn_inputs(
    attn_type: str,
    groups: int,
    use_attn_mask: bool | str,
    device: torch.device | str,
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
    device: torch.device | str,
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
