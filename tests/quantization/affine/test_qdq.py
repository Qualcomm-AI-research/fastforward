# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Tests for the quant-dequant kernel."""

from typing import Any, TypeAlias

import fastforward as ff
import pytest
import torch

from fastforward.nn.linear_quantizer import LinearQuantizer
from fastforward.quantization.affine import _ops
from fastforward.quantization.affine._autograd import (
    affine_dequantize_fn,
    affine_static_qdq_fn,
    affine_static_quantize_fn,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

IntTuple: TypeAlias = tuple[int, ...]

# `(data_shape, tile_size)` pairs covering per-tensor, per-channel over each
# axis, and multi-dimensional tiling.
SHAPES_AND_TILES: list[tuple[IntTuple, IntTuple]] = [
    ((8,), (8,)),
    ((4, 6), (4, 6)),
    ((4, 6), (1, 6)),
    ((4, 6), (4, 1)),
    ((2, 4, 6), (1, 4, 6)),
    ((2, 4, 6), (2, 1, 6)),
    ((2, 4, 6), (2, 4, 1)),
    ((2, 4, 6), (1, 1, 6)),
    ((8, 16), (4, 8)),
    ((6, 10, 4), (3, 5, 2)),
]


def _quant_params(
    data_shape: IntTuple,
    tile_size: IntTuple,
    device: str,
    with_offset: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Build data, scale and (optionally) offset for a shape/tile combination."""
    generator = torch.Generator(device="cpu").manual_seed(1234)
    num_tiles = torch.Size(data_shape).numel() // torch.Size(tile_size).numel()

    data = torch.randn(*data_shape, generator=generator) * 2.0
    scale = torch.rand(num_tiles, generator=generator) * 0.05 + 0.01
    offset = torch.round(torch.randn(num_tiles, generator=generator) * 5.0) if with_offset else None

    data = data.to(device)
    scale = scale.to(device)
    offset = offset.to(device) if offset is not None else None
    return data, scale, offset


# ------------------------------------------------------------------------------
# Op level: fused QdQ vs. the reference two-step compositions
# ------------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("data_shape", "tile_size"), SHAPES_AND_TILES)
@pytest.mark.parametrize("with_offset", [False, True], ids=["symmetric", "asymmetric"])
def test_qdq_op_matches_reference_quantize_then_dequantize(
    device: str, data_shape: IntTuple, tile_size: IntTuple, with_offset: bool
) -> None:
    # GIVEN data and quantization parameters for a tile configuration
    data, scale, offset = _quant_params(data_shape, tile_size, device, with_offset)

    # WHEN quantizing then dequantizing with the reference two-step path, and
    # quantizing-and-dequantizing in one call with the fused broadcast op
    quantized = _ops.affine_static_quantize_op(data, scale, tile_size, 8, None, offset)
    expected = _ops.affine_dequantize_op(quantized, scale, tile_size, offset, None)
    actual = _ops.affine_static_qdq_op(data, scale, tile_size, 8, None, offset)

    # THEN the results are bit-identical
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("num_bits", [2, 4, 8, 16])
def test_qdq_op_respects_bitwidth(device: str, num_bits: int) -> None:
    # GIVEN data that exceeds the quantization range on both sides
    data, scale, offset = _quant_params((4, 16), (4, 16), device, with_offset=False)

    # WHEN quantize-dequantizing at a given bitwidth with the fused op, and
    # with the reference two-step path
    quantized = _ops.affine_static_quantize_op(data, scale, (4, 16), num_bits, None, offset)
    expected = _ops.affine_dequantize_op(quantized, scale, (4, 16), offset, None)
    actual = _ops.affine_static_qdq_op(data, scale, (4, 16), num_bits, None, offset)

    # THEN the fused op clamps to the same integer grid as the reference path
    assert torch.equal(actual, expected)


# --------------------------------------------------------------------------------------
# Autograd wrapper level: the full quantize-dequantize matrix, forward and backward
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("data_shape", "tile_size"), SHAPES_AND_TILES)
@pytest.mark.parametrize("with_offset", [False, True], ids=["symmetric", "asymmetric"])
def test_mode_matches_reference_end_to_end(
    device: str, data_shape: IntTuple, tile_size: IntTuple, with_offset: bool
) -> None:
    """Every implementation must reproduce the reference output and gradients."""
    # GIVEN the same data and parameters, quantized and dequantized through the
    # reference implementation and qdq mode
    data, scale, offset = _quant_params(data_shape, tile_size, device, with_offset)
    num_bits = 8

    ref_dequant, ref_data_grad, ref_scale_grad, ref_offset_grad = _run_quant_dequant(
        data, scale, offset, tile_size, num_bits, use_qdq_fn=False
    )
    actual_dequant, actual_data_grad, actual_scale_grad, actual_offset_grad = _run_quant_dequant(
        data, scale, offset, tile_size, num_bits, use_qdq_fn=True
    )

    # THEN the dequantized output and data gradient are bit-identical, and the
    # scale/offset gradients agree up to floating point reassociation
    assert torch.equal(actual_dequant, ref_dequant)
    assert torch.equal(actual_data_grad, ref_data_grad)
    torch.testing.assert_close(actual_scale_grad, ref_scale_grad, rtol=1e-4, atol=1e-4)
    if with_offset:
        assert ref_offset_grad is not None
        assert actual_offset_grad is not None
        torch.testing.assert_close(actual_offset_grad, ref_offset_grad, rtol=1e-4, atol=1e-4)


def _run_quant_dequant(
    data: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor | None,
    tile_size: IntTuple,
    num_bits: int,
    use_qdq_fn: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run a full quantize-then-dequantize computation.

    Returns:
        A `(dequantized, data_grad, scale_grad, offset_grad)` tuple, where
        `offset_grad` is `None` if `offset` is `None`.
    """
    size = torch.Size(tile_size)
    data = data.detach().clone().requires_grad_(True)
    scale = scale.detach().clone().requires_grad_(True)
    offset = offset.detach().clone().requires_grad_(True) if offset is not None else None

    with ff.qdq_mode(use_qdq_fn):
        if use_qdq_fn:
            dequantized = affine_static_qdq_fn(data, scale, offset, size, num_bits, None)
        else:
            quantized = affine_static_quantize_fn(data, scale, offset, size, num_bits, None)
            dequantized = affine_dequantize_fn(quantized, scale, offset, size, None)

    # A non-uniform objective makes the scale/offset gradients non-trivial.
    dequantized.pow(2).sum().backward()
    offset_grad = offset.grad if offset is not None else None
    assert data.grad is not None
    assert scale.grad is not None
    return dequantized.detach(), data.grad, scale.grad, offset_grad


# --------------------------------------------------------------------------------------
# End to end through the quantizer
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "granularity_factory",
    [ff.PerTensor, lambda: ff.PerChannel(0), lambda: ff.PerChannel(1)],
    ids=["per-tensor", "per-channel-0", "per-channel-1"],
)
@pytest.mark.parametrize("symmetric", [True, False], ids=["symmetric", "asymmetric"])
def test_qdq_mode_matches_reference_end_to_end(
    device: str, granularity_factory: Any, symmetric: bool
) -> None:
    """`qdq_mode` alone must not change quantizer results."""
    data = torch.randn(8, 16, device=device, requires_grad=True)
    granularity = granularity_factory()
    dimensionality = granularity.parameter_dimensionality(torch.Size((8, 16)))
    lower = torch.full((dimensionality,), -3.0, device=device)
    quantizer = LinearQuantizer(
        num_bits=8, granularity=granularity, symmetric=symmetric, device=torch.device(device)
    )
    quantizer.quantization_range = (lower, -lower)

    # GIVEN the same quantizer and data, run with `qdq_mode` off and on
    ref_out, ref_data_grad, ref_scale_grad = _run_fwd_bck(data, quantizer, qdq=False)
    qdq_out, qdq_data_grad, qdq_scale_grad = _run_fwd_bck(data, quantizer, qdq=True)

    # THEN outputs and data gradients are bit-identical and scale gradients match
    assert torch.equal(qdq_out, ref_out)
    assert torch.equal(qdq_data_grad, ref_data_grad)
    torch.testing.assert_close(qdq_scale_grad, ref_scale_grad, rtol=1e-4, atol=1e-4)


def _run_fwd_bck(
    data: torch.Tensor, quantizer: LinearQuantizer, qdq: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with ff.qdq_mode(qdq):
        output = quantizer(data)
    if not qdq:
        output = output.dequantize()

    # A non-uniform objective makes the scale gradient non-trivial.
    output.pow(2).sum().backward()

    assert data.grad is not None
    assert quantizer.scale.grad is not None
    return output, data.grad, quantizer.scale.grad
