# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Any, Callable, Sequence, TypeVar, overload

import torch

T = TypeVar("T")
S = TypeVar("S")


@overload
def maybe_tensor_apply(maybe_tensor: torch.Tensor, fn: Callable[[torch.Tensor], S]) -> S: ...


@overload
def maybe_tensor_apply(maybe_tensor: T, fn: Callable[[torch.Tensor], Any]) -> T: ...


def maybe_tensor_apply(maybe_tensor: T, fn: Callable[[torch.Tensor], S]) -> T | S:
    """Apply function to tensor.

    Apply `fn` to `maybe_tensor` if `maybe_tensor` is an instance of `torch.Tensor`.
    Otherwise return `maybe_tensor.

    Args:
        maybe_tensor: Any object
        fn: Function that is applied to `maybe_tensor` if it is a tensor
    """
    if isinstance(maybe_tensor, torch.Tensor):
        return fn(maybe_tensor)
    return maybe_tensor


def ensure_tensor(
    maybe_tensor: torch.Tensor | float | Sequence[float], device: torch.device | None = None
) -> torch.Tensor:
    """Convert `maybe_tensor` to a tensor if it is not already a tensor.

    Args:
        maybe_tensor: `Object` to convert to a tensor
        device: Device to create the tensor on if `maybe_tensor` is not a
            tensor, otherwise device is ignored.
    """
    if isinstance(maybe_tensor, torch.Tensor):
        return maybe_tensor
    return torch.tensor(maybe_tensor, device=device)


@overload
def tensor_or_none(
    data: float | torch.Tensor, dtype: torch.dtype, device: torch.device
) -> torch.Tensor: ...
@overload
def tensor_or_none(data: None, dtype: torch.dtype, device: torch.device) -> None: ...


def tensor_or_none(
    data: float | torch.Tensor | None, dtype: torch.dtype, device: torch.device
) -> torch.Tensor | None:
    """Convert data to `Tensor` unless it already is or equals `None`.

    Args:
        data: data to convert to a tensor
        dtype: The dtype to use for the newly created tensor.
        device: The device to create the new tensor on.

    Returns:
        Tensor from `data` or `None` is `data` equals None.

    """
    if data is None or isinstance(data, torch.Tensor):
        return data
    return torch.tensor(data, dtype=dtype, device=device)
