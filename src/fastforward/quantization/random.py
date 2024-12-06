# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import Optional

import torch

from fastforward.quantization import granularity as granularities
from fastforward.quantized_tensor import QuantizedTensor

from .affine import quantize_by_tile


def random_quantized(
    shape: tuple[int, ...],
    scale: float | torch.Tensor = 0.1,
    offset: Optional[int | torch.Tensor] = None,
    num_bits: int = 3,
    requires_grad: bool = False,
    granularity: granularities.Granularity = granularities.PerTensor(),
    device: Optional[torch.device | str] = None,
    storage_dtype: torch.dtype = torch.float32,
) -> QuantizedTensor:
    """
    Generate a random quantized tensor.

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
    if isinstance(scale, torch.Tensor) and scale.numel() > 1:
        if not isinstance(offset, torch.Tensor):
            offset = torch.ones(scale.shape) * (offset if offset else 0)
        elif offset.numel() != scale.numel():
            raise ValueError(
                "scale and offset must contain the same number of elements. "
                f"Found {scale.numel()} and {offset.numel()}"
            )

    tile_size = granularity.tile_size(torch.Size(shape))
    random_tensor = quantize_by_tile(
        torch.randn(*shape, device=device),
        scale=scale,
        offset=offset,
        num_bits=num_bits,
        tile_size=tile_size,
        output_dtype=storage_dtype,
    )
    if requires_grad:
        random_tensor.requires_grad_()
    return random_tensor
