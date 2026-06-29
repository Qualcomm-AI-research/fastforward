# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#
# This file includes code derived from https://github.com/IST-DASLab/gptq
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import logging
import math

from typing import Any, Callable, Iterable, cast

import torch

import fastforward as ff
import fastforward.quantization.affine as affine_quant

from fastforward.quantization import granularity as granularities

logger = logging.getLogger(__name__)


def gptq(
    module: ff.nn.QuantizedLinear | ff.nn.QuantizedConv2d,
    dataset: Iterable[tuple[tuple[Any, ...], dict[str, Any]]],
    block_size: int = 128,
    perc_damp: float = 0.01,
    actorder: bool = False,
    layer_name: str = "",
) -> None:
    """Quantize a QuantizedLinear or QuantizedConv2d layer in-place using GPTQ.

    Only `PerTensor` and `PerChannel(channel_dim=0)` weight quantizer granularities
    are supported on Conv2d.

    Args:
        module: A QuantizedLinear or QuantizedConv2d layer whose weight_quantizer is a LinearQuantizer.
        dataset: Input activations flowing from the previous layer, an iterable.
            Each entry is a tuple of args (tuple of obj) and kwargs (dict).
        block_size: Number of columns to process per block.
        perc_damp: Dampening factor as percentage of mean diagonal value.
        actorder: Whether to reorder columns by activation magnitude.
        layer_name: Optional name for logging.
    """
    if not isinstance(module.weight_quantizer, ff.nn.LinearQuantizer):
        msg = f"weight_quantizer must be a LinearQuantizer, got {type(module.weight_quantizer).__name__}."
        raise ValueError(msg)

    original_weight_shape = module.weight.shape
    weights = module.weight.data.clone().float()

    if isinstance(module, ff.nn.QuantizedConv2d):
        match module.weight_quantizer.granularity:
            case granularities.PerTensor() | granularities.PerChannel(channel_dims=(0,)):
                pass
            case _ as g:
                msg = (
                    "GPTQ on QuantizedConv2d currently supports only PerTensor and "
                    f"PerChannel(channel_dim=0) granularities. Got: {type(g).__name__}."
                )
                raise NotImplementedError(msg)
        # (out_channels, in_channels/groups, kH, kW) -> (out_channels, in_channels/groups * kH * kW)
        weights = weights.flatten(1)

    columns = weights.shape[1]

    weight_quantizer = module.weight_quantizer

    with ff.estimate_ranges(weight_quantizer, ff.range_setting.smoothed_minmax):
        weight_quantizer(weights)

    hessian = calculate_hessian(module, dataset)

    column_order = (
        torch.argsort(torch.diag(hessian), descending=True) if actorder else torch.arange(columns)
    )
    weights = weights[:, column_order]
    hessian = hessian[column_order][:, column_order]

    quantized_weights = torch.zeros_like(weights)
    errors = torch.zeros_like(weights)
    hessian_inverse = invert_hessian(hessian, perc_damp)

    for i in range(0, columns, block_size):
        block_end = min(i + block_size, columns)

        weights_block = weights[:, i:block_end].clone()
        hessinv_block = hessian_inverse[i:block_end, i:block_end]

        for j in range(block_end - i):
            orig_col = int(column_order[i + j].item())
            quant_deq = column_quantizer(weight_quantizer, weights.shape, orig_col)
            quantized_weights[:, i + j] = quant_deq(weights_block[:, j])

            errors[:, i + j] = (weights_block[:, j] - quantized_weights[:, i + j]) / hessinv_block[
                j, j
            ]

            weights_block[:, j + 1 :] -= (
                errors[:, i + j].unsqueeze(1) @ hessinv_block[j : j + 1, j + 1 :]
            )

        weights[:, block_end:] -= errors[:, i:block_end] @ hessian_inverse[i:block_end, block_end:]

    restore_order = torch.argsort(column_order)
    quantized_weights = quantized_weights[:, restore_order]
    errors = errors[:, restore_order]

    module.weight.copy_(quantized_weights.view(original_weight_shape).to(module.weight.dtype))

    loss = torch.mean(torch.abs(errors)).item()
    num_bits = module.weight_quantizer.num_bits
    logger.info("[GPTQ][wbits=%d][%s] loss=%.6f", num_bits, layer_name, loss)


def column_quantizer(
    weight_quantizer: ff.nn.LinearQuantizer, weight_shape: torch.Size, col_index: int
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a quantize-dequantize operator for a single column.

    GPTQ optimizes consecutive (blocks of) columns. The quantizers in FastForward can operate
    on various levels of granularity, so here we transform the selected granularity explicitly
    to a PerChannel(channel_dim=0) setting (one scale/offset per row).

    Args:
        weight_quantizer: A calibrated LinearQuantizer.
        weight_shape: Shape of the full weight tensor `(out_features, in_features)`.
        col_index: Column index in the original weight matrix.

    Returns:
        A callable that takes a column and returns it quantized-dequantized.
    """
    out_features, in_features = weight_shape
    scale: torch.Tensor = weight_quantizer.scale
    offset: torch.Tensor | None = weight_quantizer.offset

    match weight_quantizer.granularity:
        case granularities.PerTensor():
            # Expand the single scale for the entire column.
            scale = scale.expand(out_features)
            offset = offset.expand(out_features) if offset is not None else None

        case granularities.PerChannel(channel_dims=(0,)):
            # Linear: scale.shape == (out_features,) — reshape is a no-op.
            # Conv2d: scale.shape == (out_channels, 1, 1, 1) — reshape flattens to (out_channels,).
            scale = scale.reshape(out_features)
            offset = offset.reshape(out_features) if offset is not None else None

        case granularities.PerChannel(channel_dims=(1,)):
            # One scale for the entire column.
            scale = scale[col_index].expand(out_features)
            offset = offset[col_index].expand(out_features) if offset is not None else None

        case granularities.PerChannel(channel_dims=(0, 1)):
            # Per-element scale. NB: this is a valid configuration but a strange one.
            scale = scale.view(out_features, in_features)[:, col_index]
            offset = (
                offset.view(out_features, in_features)[:, col_index] if offset is not None else None
            )

        case granularities.PerBlock(strict_blocks=False):
            msg = "GPTQ does not support PerBlock with strict_blocks=False."
            raise ValueError(msg)

        case granularities.PerBlock() | granularities.PerTile():
            tile_size = weight_quantizer.granularity.tile_size(weight_shape)
            row_block_size, col_block_size = tile_size
            num_row_blocks = out_features // row_block_size
            num_col_blocks = in_features // col_block_size
            col_block_idx = col_index // col_block_size

            # Pick the correct column for this block.
            scale = scale.view(num_row_blocks, num_col_blocks)[:, col_block_idx]

            # broadcast the column values to get (out_features,) shape.
            scale = scale.repeat_interleave(row_block_size)

            offset = (
                offset.view(num_row_blocks, num_col_blocks)[:, col_block_idx].repeat_interleave(
                    row_block_size
                )
                if offset is not None
                else None
            )

        case _:
            msg = f"Unsupported granularity: {type(weight_quantizer.granularity).__name__}"
            raise TypeError(msg)

    ctx = affine_quant.quantization_context(
        scale=scale,
        offset=offset,
        num_bits=weight_quantizer.num_bits,
        granularity=granularities.PerChannel(channel_dim=0),
        output_dtype=weight_quantizer.quantized_dtype,
    )

    def _quant_fn(col: torch.Tensor) -> torch.Tensor:
        q = ctx.quantization_fn.quantize(col.unsqueeze(1), ctx.quantization_params)
        return q.dequantize().flatten()

    return _quant_fn


def calculate_hessian(
    layer: ff.nn.QuantizedLinear | ff.nn.QuantizedConv2d,
    activations: Iterable[tuple[tuple[Any, ...], dict[str, Any]]],
) -> torch.Tensor:
    """Calculate approximate Hessian matrix from layer activations.

    Args:
        layer: QuantizedLinear or QuantizedConv2d layer to quantize.
        activations: Input activations flowing from the previous layer, an iterable.
            Each entry is a tuple of args (tuple of obj) and kwargs (dict).

    Returns:
        Hessian matrix of shape [in_features, in_features] in float32, where
        in_features = weight.shape[1] for Linear and in_channels/groups * kH * kW for Conv2d.
    """
    device = layer.weight.device
    in_features = _hessian_in_features(layer)
    flatten_input = _input_flattener(layer)

    hessian = torch.zeros((in_features, in_features), device=device, dtype=torch.float64)
    n_samples = 0

    for (activation,), _ in activations:
        activation = cast(torch.Tensor, activation)
        x = flatten_input(activation.to(device=device, dtype=torch.float32))

        hessian.mul_(n_samples / (n_samples + x.shape[1]))
        n_samples += x.shape[1]

        x.mul_(math.sqrt(2.0 / n_samples))
        hessian.add_(x @ x.transpose(0, 1))

    # Handle dead neurons (zero diagonal in Hessian).
    dead = torch.diag(hessian) == 0
    hessian[dead, dead] = 1
    return hessian.float()


def _hessian_in_features(layer: ff.nn.QuantizedLinear | ff.nn.QuantizedConv2d) -> int:
    """Number of input features the Hessian is built over.

    Matches the column count of the flattened weight used inside `gptq`:
    `in_features` for Linear, `in_channels/groups * kH * kW` for Conv2d.
    """
    if isinstance(layer, ff.nn.QuantizedConv2d):
        _, in_per_group, kh, kw = layer.weight.shape
        return in_per_group * kh * kw
    return layer.weight.shape[1]


def _input_flattener(
    layer: ff.nn.QuantizedLinear | ff.nn.QuantizedConv2d,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build the per-batch reshape that produces `(in_features, samples)` activations.

    For Linear: `(B, S, H) -> (H, B*S)`.
    For Conv2d: `(N, C, H, W) -> (C/groups * kH * kW, N*L)` via `nn.Unfold`,
    matching the flattened weight layout used inside `gptq`.
    """
    if isinstance(layer, ff.nn.QuantizedConv2d):
        assert not isinstance(layer.padding, str), "string padding not supported"
        unfold = torch.nn.Unfold(
            kernel_size=layer.kernel_size,
            dilation=layer.dilation,
            padding=layer.padding,
            stride=layer.stride,
        )

        def _conv2d(activation: torch.Tensor) -> torch.Tensor:
            # (N, C, H, W) -> (N, C*kH*kW, L) -> (C*kH*kW, N*L)
            unfolded = cast(torch.Tensor, unfold(activation))
            return unfolded.permute(1, 0, 2).flatten(1)

        return _conv2d

    def _linear(activation: torch.Tensor) -> torch.Tensor:
        # (B, S, H) -> (B*S, H) -> (H, B*S)
        bsz, seq_len, hidden = activation.shape
        return activation.reshape(bsz * seq_len, hidden).transpose(0, 1)

    return _linear


def invert_hessian(hessian: torch.Tensor, perc_damp: float) -> torch.Tensor:
    """Invert Hessian matrix using Cholesky decomposition with dampening.

    Args:
        hessian: Square Hessian matrix of shape [n, n]
        perc_damp: Dampening factor as percentage of mean diagonal value

    Returns:
        Inverted Hessian matrix of shape [n, n]
    """
    dampening = perc_damp * torch.mean(torch.diag(hessian))
    diag = torch.arange(hessian.shape[0], device=hessian.device)
    hessian[diag, diag] += dampening

    # invert via Cholesky
    hessian = torch.linalg.cholesky(hessian)
    hessian = torch.cholesky_inverse(hessian)
    hessian = torch.linalg.cholesky(hessian, upper=True)
    return hessian
