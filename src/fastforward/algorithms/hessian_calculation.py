# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#
# This file includes code derived from https://github.com/IST-DASLab/gptq
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import math

from typing import Any, Iterable, cast

import torch

_Size = int | tuple[int, int]


def calculate_hessian(
    activations: Iterable[tuple[tuple[Any, ...], dict[str, Any]]],
    *,
    device: torch.device | None = None,
    kernel_size: _Size | None = None,
    dilation: _Size = 1,
    padding: _Size = 0,
    stride: _Size = 1,
) -> torch.Tensor:
    """Calculate approximate Hessian matrix from layer activations.

    This is the function of the `FlowOp` that GPTQ registers. The settings of the layer
    come from `hessian_settings`.

    Args:
        activations: Input activations flowing into the layer, an iterable. Each entry
            is a tuple of args (tuple of obj) and kwargs (dict). The first arg is a
            tensor of shape [..., in_features], or of shape [N, C, H, W] for a convolution.
        device: The device to accumulate on. Each batch is moved there. If `None`, the
            device of the first batch.
        kernel_size: The kernel size of the convolution, or `None` for a linear layer. The
            patches of a convolution are its samples.
        dilation: The dilation of the convolution.
        padding: The padding of the convolution.
        stride: The stride of the convolution.

    Returns:
        Hessian matrix of shape [in_features, in_features] in float32. For a convolution,
        in_features is C * kernel height * kernel width.

    Raises:
        ValueError: If `activations` has no entry.
    """
    hessian: torch.Tensor | None = None
    n_samples = 0

    for (activation,), _ in activations:
        x = cast(torch.Tensor, activation).to(device=device, dtype=torch.float32)
        if kernel_size is None:
            # (..., H) -> (H, N)
            x = x.reshape(-1, x.shape[-1]).transpose(0, 1)
        else:
            # (N, C, H, W) -> (C * kH * kW, N * L), with L the number of patches
            x = torch.nn.functional.unfold(
                x, kernel_size, dilation=dilation, padding=padding, stride=stride
            )
            x = x.permute(1, 0, 2).flatten(1)

        if hessian is None:
            hessian = torch.zeros((x.shape[0], x.shape[0]), device=x.device, dtype=torch.float64)

        hessian.mul_(n_samples / (n_samples + x.shape[1]))
        n_samples += x.shape[1]

        # Not in-place: `x` can be a view of an activation that other layers also read.
        x = x * math.sqrt(2.0 / n_samples)
        hessian.add_(x @ x.transpose(0, 1))

    if hessian is None:
        msg = "Cannot calculate a Hessian without activations."
        raise ValueError(msg)

    # Handle dead neurons (zero diagonal in Hessian).
    dead = torch.diag(hessian) == 0
    hessian[dead, dead] = 1
    return hessian.float()


def hessian_settings(module: torch.nn.Module) -> dict[str, Any]:
    """Return the arguments of `calculate_hessian` that depend on the layer.

    This is the `from_module` of the `FlowOp` that GPTQ registers.

    Args:
        module: The layer.

    Returns:
        The device of the weight, and the settings of the convolution for a `Conv2d`. The
        values are hashable.

    Raises:
        ValueError: If the padding of the convolution is a string.
        NotImplementedError: If the convolution has groups, or pads with other values
            than zeros.
    """
    device = {"device": module.weight.device} if isinstance(module.weight, torch.Tensor) else {}
    if not isinstance(module, torch.nn.Conv2d):
        return device
    if isinstance(module.padding, str):
        msg = f"A string padding is not supported, got {module.padding!r}."
        raise ValueError(msg)
    if module.groups != 1:
        msg = f"A convolution with groups is not supported, got groups={module.groups}."
        raise NotImplementedError(msg)
    if module.padding_mode != "zeros":
        # `unfold` pads with zeros, so the Hessian would not match the convolution.
        msg = f"Only zero padding is supported, got padding_mode={module.padding_mode!r}."
        raise NotImplementedError(msg)
    return {
        **device,
        "kernel_size": module.kernel_size,
        "dilation": module.dilation,
        "padding": module.padding,
        "stride": module.stride,
    }


def invert_hessian(hessian: torch.Tensor, perc_damp: float) -> torch.Tensor:
    """Return the inverse of a dampened copy of `hessian`, using Cholesky decomposition.

    Args:
        hessian: Square Hessian matrix of shape [n, n]
        perc_damp: Dampening factor as percentage of mean diagonal value

    Returns:
        Inverted Hessian matrix of shape [n, n]
    """
    dampening = perc_damp * torch.mean(torch.diag(hessian))
    hessian = hessian + dampening * torch.eye(
        hessian.shape[0], device=hessian.device, dtype=hessian.dtype
    )

    # invert via Cholesky
    hessian = torch.linalg.cholesky(hessian)
    hessian = torch.cholesky_inverse(hessian)
    hessian = torch.linalg.cholesky(hessian, upper=True)
    return hessian
