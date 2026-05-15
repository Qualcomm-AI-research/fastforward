# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import dataclasses

from collections.abc import Sequence
from typing import Any, cast

import torch


@dataclasses.dataclass
class FFQuantizerSpec:
    """FF specific quantization spec used during ONNX metadata export."""

    num_bits: int
    scale: torch.Tensor
    offset: torch.Tensor
    symmetric: bool
    tile_size: tuple[int, ...] | None = None
    data_shape: tuple[int, ...] | None = None


def _ff_quantizer_spec(
    node: torch.fx.Node, quant_params: dict[str, torch.Tensor]
) -> FFQuantizerSpec:
    """Get ff specific quantization spec for `node`."""
    if node.target is not torch.ops.fastforward.quantize_by_tile.default:  # type: ignore[misc,unused-ignore]
        msg = f"Unsupported target: {node.target}"
        raise NotImplementedError(msg)

    data, scale, tile_size, num_bits, _output_dtype, offset = node.args

    assert isinstance(scale, torch.fx.Node) and isinstance(scale.target, str)
    assert isinstance(offset, torch.fx.Node) and isinstance(offset.target, str)
    scale_value = quant_params[scale.target]
    offset_value = quant_params[offset.target]
    symmetric = _is_symmetric_offset(offset_value)

    if not isinstance(num_bits, (int, float)):
        msg = f"Expected number num_bits, got {type(num_bits)}"
        raise TypeError(msg)
    num_bits = float(num_bits)
    if not num_bits.is_integer():
        msg = f"Cannot export non-integer bitwidths, found {num_bits}"
        raise ValueError(msg)

    if not symmetric:
        offset_value = 2 ** (num_bits - 1) - quant_params[offset.target]

    tile_size_tuple: tuple[int, ...] | None = None
    if isinstance(tile_size, torch.Size):
        tile_size_tuple = tuple(int(dim) for dim in tile_size)
    elif isinstance(tile_size, (tuple, list)):
        if not all(isinstance(dim, int) for dim in tile_size):
            msg = f"Expected integer tile_size values, found {tile_size}"
            raise TypeError(msg)
        int_tile_size = cast(Sequence[int], tile_size)
        tile_size_tuple = tuple(int_tile_size)

    data_shape_tuple: tuple[int, ...] | None = None
    if isinstance(data, torch.fx.Node):
        meta = data.meta
        if isinstance(meta, dict):
            tensor_meta: Any = meta.get("tensor_meta")
            shape = getattr(tensor_meta, "shape", None)
            if shape is not None:
                data_shape_tuple = tuple(int(dim) for dim in shape)

    return FFQuantizerSpec(
        num_bits=int(num_bits),
        scale=scale_value,
        offset=offset_value,
        symmetric=symmetric,
        tile_size=tile_size_tuple,
        data_shape=data_shape_tuple,
    )


def _is_symmetric_offset(offset: torch.Tensor | float) -> bool:
    match offset:
        case float():
            return offset == 0.0
        case torch.Tensor():
            return bool((offset == 0.0).all())
