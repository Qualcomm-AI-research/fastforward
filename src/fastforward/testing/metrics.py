# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import torch

import fastforward as ff


def sqnr(
    non_quantized_tensor: torch.Tensor,
    quantized_tensor: torch.Tensor,
    in_db: bool = True,
    eps: float = 1e-15,
) -> torch.Tensor:
    """Calculate the signal-to-quantization-noise-ratio (SQNR) over the batch dimension.

    Args:
        non_quantized_tensor: Original tensor.
        quantized_tensor: Quantized tensor.
        in_db: Whether to return the output in dB scale.
        eps: Small value to avoid division by zero.

    Returns:
        SQNR value between the two input tensors.
    """
    if non_quantized_tensor.shape != quantized_tensor.shape:
        msg = f"The shapes of `org_out` and `quant_out` must match. Got {non_quantized_tensor.shape} and {quantized_tensor.shape}."
        raise ValueError(msg)

    with ff.strict_quantization(False):
        quant_error = non_quantized_tensor - quantized_tensor
        exp_noise = quant_error.pow(2).view(quant_error.shape[0], -1).mean(1) + eps
        exp_signal = non_quantized_tensor.pow(2).view(non_quantized_tensor.shape[0], -1).mean(1)
        sqnr = (exp_signal / exp_noise).mean()

    if in_db:
        return 10 * torch.log10(sqnr)
    return sqnr
