# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#
# This file includes code derived from https://github.com/IST-DASLab/gptq
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import logging
import math

from typing import cast

import torch

import fastforward as ff

from fastforward._orchestration.graph_module import GraphModule
from fastforward._orchestration.instruction_engine import ActivationDataset

logger = logging.getLogger(__name__)


def _find_quantized_linear(module: GraphModule) -> tuple[ff.nn.QuantizedLinear, str]:
    """Find and validate the single QuantizedLinear layer in a GraphModule.

    Args:
        module: GraphModule to search.

    Returns:
        Tuple of (QuantizedLinear layer, full layer name).

    Raises:
        ValueError: If the module does not contain exactly one QuantizedLinear, if it is
            not the first node in the subgraph, or if its weight_quantizer is not a
            LinearQuantizer.
    """
    layers = ff.mpath.search("**/[cls:ff.nn.QuantizedLinear]", module)
    if len(layers) != 1:
        msg = (
            f"Expected exactly 1 QuantizedLinear layer in module, found {len(layers)}. "
            f"GPTQ requires a subgraph with a single QuantizedLinear as its first layer."
        )
        raise ValueError(msg)

    quantized_linear = cast(ff.nn.QuantizedLinear, layers[0].module)
    layer_name = layers[0].full_name

    # If this Node has any (internal) Nodes as input, it's not the first node in the subgraph.
    node_ref = module.node_ref(quantized_linear)
    if any(True for _ in module.node_inputs(node_ref)):
        msg = (
            f"[{layer_name}] QuantizedLinear is not the first layer in the subgraph. "
            f"GPTQ requires the linear layer to receive subgraph inputs directly, "
            f"so that activations passed to the Hessian are its direct inputs."
        )
        raise ValueError(msg)

    if not isinstance(quantized_linear.weight_quantizer, ff.nn.LinearQuantizer):
        msg = (
            f"[{layer_name}] weight_quantizer must be a LinearQuantizer, "
            f"got {type(quantized_linear.weight_quantizer).__name__}."
        )
        raise ValueError(msg)

    return quantized_linear, layer_name


def gptq(
    module: GraphModule,
    dataset: ActivationDataset,
    block_size: int = 128,
    perc_damp: float = 0.01,
    actorder: bool = False,
) -> None:
    """Quantize a GraphModule containing a single QuantizedLinear layer using GPTQ.

    Args:
        module: GraphModule containing exactly one QuantizedLinear layer
        dataset: Input activation dataset of shape [batch_size, seq_len, in_features]
        block_size: Number of columns to process per block
        perc_damp: Dampening factor as percentage of mean diagonal value
        actorder: Whether to reorder columns by activation magnitude
    """
    quantized_linear, layer_name = _find_quantized_linear(module)
    loss = _gptq(
        quantized_linear, dataset, block_size=block_size, perc_damp=perc_damp, actorder=actorder
    )

    num_bits = quantized_linear.weight_quantizer.num_bits
    logger.info("[GPTQ][wbits=%d][%s] loss=%.6f", num_bits, layer_name, loss)


def _gptq(
    layer: ff.nn.QuantizedLinear,
    activations: ActivationDataset,
    block_size: int = 128,
    perc_damp: float = 0.01,
    actorder: bool = False,
) -> float:
    """Quantize a QuantizedLinear layer using GPTQ algorithm on a per-channel basis.

    Args:
        layer: QuantizedLinear layer to quantize
        activations: Input activation dataset of shape [batch_size, seq_len, in_features]
        block_size: Number of columns to process per block
        perc_damp: Dampening factor as percentage of mean diagonal value
        actorder: Whether to reorder columns by activation magnitude

    Returns:
        Average quantization loss in float precision
    """
    columns = layer.weight.shape[1]
    weights = layer.weight.data.clone().float()

    weight_quantizer = layer.weight_quantizer
    assert isinstance(weight_quantizer, ff.nn.LinearQuantizer)

    # [paper] the quantization grid for [W] is fixed before the process,
    # individual weights can move freely [...].
    with ff.estimate_ranges(weight_quantizer, ff.range_setting.smoothed_minmax):
        weight_quantizer(weights)

    # Approximate the hessian from the intermediate activations of the previous layer.
    hessian = calculate_hessian(layer, activations)

    if actorder:
        perm = torch.argsort(torch.diag(hessian), descending=True)
        weights = weights[:, perm]
        hessian = hessian[perm][:, perm]
        invperm = torch.argsort(perm)

    quantized_weights = torch.zeros_like(weights)  # [paper] // quantized output
    errors = torch.zeros_like(weights)  # [paper] // block quantization errors
    hessian_inverse = invert_hessian(hessian, perc_damp)  # [paper] Hessian inverse information

    # [paper] for i = 0, B, 2B, ...
    for i in range(0, columns, block_size):
        block_end = min(i + block_size, columns)

        weights_block = weights[:, i:block_end].clone()
        hessinv_block = hessian_inverse[i:block_end, i:block_end]

        # [paper] for j = i, ..., i + B - 1
        for j in range(block_end - i):
            # [paper] Q <- ... // quantize-dequantize columns
            quantized_weights[:, i + j] = (
                weight_quantizer(weights_block[:, j].unsqueeze(1)).dequantize().flatten()
            )

            # [paper] E <- ... // quantization error
            errors[:, i + j] = (weights_block[:, j] - quantized_weights[:, i + j]) / hessinv_block[
                j, j
            ]

            # [paper] W <- ... // update weights in block
            weights_block[:, j + 1 :] -= (
                errors[:, i + j].unsqueeze(1) @ hessinv_block[j : j + 1, j + 1 :]
            )

        # [paper] W <- // update all remaining weights
        weights[:, block_end:] -= errors[:, i:block_end] @ hessian_inverse[i:block_end, block_end:]

    if actorder:
        quantized_weights = quantized_weights[:, invperm]
        errors = errors[:, invperm]

    # Modify weights in-place.
    layer.weight.copy_(quantized_weights.to(layer.weight.dtype))

    return torch.mean(torch.abs(errors)).item()


def calculate_hessian(layer: ff.nn.QuantizedLinear, activations: ActivationDataset) -> torch.Tensor:
    """Calculate approximate Hessian matrix from layer activations.

    Args:
        layer: QuantizedLinear layer to quantize
        activations: Input activation dataset of shape [batch_size, seq_len, in_features]

    Returns:
        Hessian matrix of shape [in_features, in_features] in float32
    """
    in_features = layer.weight.shape[1]
    device = layer.weight.device

    hessian = torch.zeros((in_features, in_features), device=device, dtype=torch.float64)
    n_samples = 0

    for activation in activations.batches:
        activation = cast(torch.Tensor, activation)
        bsz, seq_len, hidden = activation.shape

        activation = activation.reshape(bsz * seq_len, hidden)
        x = activation.transpose(0, 1).to(device=device, dtype=torch.float32)

        hessian.mul_(n_samples / (n_samples + (bsz * seq_len)))
        n_samples += bsz * seq_len

        x.mul_(math.sqrt(2.0 / n_samples))
        hessian.add_(x @ x.transpose(0, 1))

    # Handle dead neurons (zero diagonal in Hessian).
    dead = torch.diag(hessian) == 0
    hessian[dead, dead] = 1
    return hessian.float()


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
