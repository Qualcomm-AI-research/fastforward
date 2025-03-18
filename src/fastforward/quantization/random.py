# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


import torch

from fastforward.quantization import granularity as granularities
from fastforward.quantized_tensor import QuantizedTensor

from .affine import quantize_per_granularity


def random_quantized(
    shape: tuple[int, ...],
    scale: float | torch.Tensor = 0.1,
    offset: int | torch.Tensor | None = None,
    num_bits: int = 3,
    requires_grad: bool = False,
    granularity: granularities.Granularity | None = None,
    device: torch.device | str | None = None,
    storage_dtype: torch.dtype = torch.float32,
) -> QuantizedTensor:
    """Generate a random quantized tensor.

    The tensor is sampled from a zero-centered unit normal distribution and
    subsequently quantized using the provided quantization parameters.

    Args:
        shape: The shape of the tensor.
        scale: The scale factor for quantization.
        offset: The offset for quantization.
        num_bits: The number of bits for quantization.
        requires_grad: If True, the tensor requires gradient.
        granularity: The granularity of quantization.
        device: The device for the tensor.
        storage_dtype: The storage data type for the tensor.

    Returns:
        QuantizedTensor: The generated random quantized tensor.

    Raises:
        ValueError: If the scale and offset tensors do not have the same number of elements.
    """
    granularity = granularity or granularities.PerTensor()
    if isinstance(scale, torch.Tensor) and scale.numel() > 1:
        if not isinstance(offset, torch.Tensor):
            offset = torch.ones(scale.shape) * (offset if offset else 0)
        elif offset.numel() != scale.numel():
            raise ValueError(
                "scale and offset must contain the same number of elements. "
                + f"Found {scale.numel()} and {offset.numel()}"
            )

    random_tensor = quantize_per_granularity(
        torch.randn(shape, device=device),
        scale=scale,
        offset=offset,
        granularity=granularity,
        num_bits=num_bits,
        output_dtype=storage_dtype,
    )
    if requires_grad:
        _ = random_tensor.requires_grad_()
    return random_tensor
