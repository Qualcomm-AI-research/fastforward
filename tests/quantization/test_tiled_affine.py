# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from collections.abc import Sequence
from typing import Any

import pytest
import torch

from fastforward.quantization import affine, tiled_tensor
from fastforward.quantization.ste import round_ste
from fastforward.type_common import SizeT


def _tiled_quant_reference(
    data: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor, num_bits: float, tile_size: SizeT
) -> torch.Tensor:
    min_threshold = -(2 ** (num_bits - 1))
    max_threshold = -min_threshold - 1
    row_representation = tiled_tensor.tiles_to_rows(data, tile_size)
    quantized = round_ste(row_representation / scale[:, None]) - round_ste(offset[:, None])
    quantized = torch.clamp(quantized, min_threshold, max_threshold)
    return tiled_tensor.rows_to_tiles(quantized, data.shape, tile_size)


def _tiled_dequant_reference(
    data: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor, tile_size: SizeT
) -> torch.Tensor:
    row_representation = tiled_tensor.tiles_to_rows(data, tile_size)
    dequantized = (row_representation + round_ste(offset[:, None])) * scale[:, None]
    return tiled_tensor.rows_to_tiles(dequantized, data.shape, tile_size)


DEVICES = ["cpu", "cuda"]
NUM_BITS = [2, 3, 4]
OUTPUT_DTYPES = [
    torch.int32,
    torch.int16,
    torch.float32,
]


def assert_close(actual: Any, expected: Any, extra_dtype: torch.dtype | None = None) -> None:
    dtypes = (torch.float16, torch.bfloat16)
    if (
        expected.dtype in dtypes
        or actual in dtypes
        or (extra_dtype is not None and extra_dtype in dtypes)
    ):
        atol = 1e-1
        rtol = 1.3e-2
    else:
        atol = 1e-4
        rtol = 1.3e-5
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.slow
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("num_bits", NUM_BITS)
@pytest.mark.parametrize("output_dtype", OUTPUT_DTYPES)
def test_quantize_per_tensor(
    device: torch.device, num_bits: int, output_dtype: torch.dtype, _seed_prngs: int
) -> None:
    data = torch.randn(8, 4, 2, device=device, requires_grad=True)
    scale = torch.tensor([0.38], device=device, requires_grad=True)
    offset = torch.tensor([0.35], device=device, requires_grad=True)

    quantized = affine.quantize_per_tensor(data, scale, offset, num_bits, output_dtype)
    dequantized = quantized.dequantize()

    quantized.dequantize().sum().backward()
    data_grad, scale_grad, offset_grad = data.grad, scale.grad, offset.grad
    data.grad, scale.grad, offset.grad = None, None, None

    expected_raw = _tiled_quant_reference(data, scale, offset, num_bits, data.shape)
    expected_dequant = _tiled_dequant_reference(expected_raw, scale, offset, data.shape)

    expected_dequant.sum().backward()

    assert_close(quantized.raw_data, expected_raw.to(output_dtype))
    assert_close(dequantized, expected_dequant)

    if output_dtype.is_floating_point:
        assert_close(data_grad, data.grad, output_dtype)
        assert_close(scale_grad, scale.grad, output_dtype)
        assert_close(offset_grad, offset.grad, output_dtype)


@pytest.mark.slow
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("num_bits", NUM_BITS)
@pytest.mark.parametrize("output_dtype", OUTPUT_DTYPES)
@pytest.mark.parametrize("channel", [0, 1, 2])
def test_quantize_per_channel(
    device: torch.device, num_bits: int, output_dtype: torch.dtype, channel: int, _seed_prngs: int
) -> None:
    data = torch.randn(32, 16, 8, device=device, requires_grad=True)
    scale = torch.randn(data.shape[channel], device=device) * 0.5 + 0.3
    offset = torch.randn(data.shape[channel], device=device) * 0.3 + 0.1

    scale.requires_grad_()
    offset.requires_grad_()

    quantized = affine.quantize_per_channel(data, scale, offset, channel, num_bits, output_dtype)
    dequantized = quantized.dequantize()

    quantized.dequantize().sum().backward()
    data_grad, scale_grad, offset_grad = data.grad, scale.grad, offset.grad
    data.grad, scale.grad, offset.grad = None, None, None

    tile_size_ = list(data.shape)
    tile_size_[channel] = 1
    tile_size: tuple[int, ...] = tuple(tile_size_)
    del tile_size_
    expected_raw = _tiled_quant_reference(data, scale, offset, num_bits, tile_size)
    expected_dequant = _tiled_dequant_reference(expected_raw, scale, offset, tile_size)

    expected_dequant.sum().backward()

    torch.testing.assert_close(quantized.raw_data, expected_raw.to(output_dtype))
    torch.testing.assert_close(dequantized, expected_dequant)

    if output_dtype.is_floating_point:
        assert_close(data_grad, data.grad, output_dtype)
        assert_close(scale_grad, scale.grad, output_dtype)
        assert_close(offset_grad, offset.grad, output_dtype)


@pytest.mark.slow
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("num_bits", NUM_BITS)
@pytest.mark.parametrize("output_dtype", OUTPUT_DTYPES)
@pytest.mark.parametrize(
    "channel_idx,block_axis,block_size", [(1, 0, 16), (2, 0, 8), (0, 1, 8), (0, 2, 4)]
)
def test_quantize_per_block(
    device: torch.device,
    num_bits: int,
    output_dtype: torch.dtype,
    channel_idx: int,
    block_axis: int,
    block_size: int,
    _seed_prngs: int,
) -> None:
    data = torch.randn(32, 16, 8, device=device, requires_grad=True)

    tile_size_ = list(data.shape)
    tile_size_[channel_idx] = 1
    tile_size_[block_axis] = block_size
    tile_size = torch.Size(tile_size_)
    del tile_size_
    num_params = data.numel() // tile_size.numel()

    scale = torch.randn(num_params, device=device) * 0.5 + 0.3
    offset = torch.randn(num_params, device=device) * 0.3 + 0.1

    scale.requires_grad_()
    offset.requires_grad_()

    quantized = affine.quantize_per_block(
        data, scale, offset, channel_idx, block_axis, block_size, num_bits, output_dtype
    )
    dequantized = quantized.dequantize()

    quantized.dequantize().sum().backward()
    data_grad, scale_grad, offset_grad = data.grad, scale.grad, offset.grad
    data.grad, scale.grad, offset.grad = None, None, None

    expected_raw = _tiled_quant_reference(data, scale, offset, num_bits, tile_size)
    expected_dequant = _tiled_dequant_reference(expected_raw, scale, offset, tile_size)

    expected_dequant.sum().backward()

    torch.testing.assert_close(quantized.raw_data, expected_raw.to(output_dtype))
    torch.testing.assert_close(dequantized, expected_dequant)

    if output_dtype.is_floating_point:
        assert_close(data_grad, data.grad, output_dtype)
        assert_close(scale_grad, scale.grad, output_dtype)
        assert_close(offset_grad, offset.grad, output_dtype)


@pytest.mark.slow
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("num_bits", NUM_BITS)
@pytest.mark.parametrize("output_dtype", OUTPUT_DTYPES)
@pytest.mark.parametrize("tile_size", [(16, 8, 4), (8, 16, 2)])
def test_quantize_by_tile(
    device: torch.device,
    num_bits: int,
    output_dtype: torch.dtype,
    tile_size: Sequence[int],
    _seed_prngs: int,
) -> None:
    data = torch.randn(32, 16, 8, device=device, requires_grad=True)

    tile_size = torch.Size(tile_size)
    num_params = data.numel() // tile_size.numel()

    scale = torch.randn(num_params, device=device) * 0.5 + 0.3
    offset = torch.randn(num_params, device=device) * 0.3 + 0.1

    scale.requires_grad_()
    offset.requires_grad_()

    quantized = affine.quantize_by_tile(data, scale, offset, tile_size, num_bits, output_dtype)
    dequantized = quantized.dequantize()

    quantized.dequantize().sum().backward()
    data_grad, scale_grad, offset_grad = data.grad, scale.grad, offset.grad
    data.grad, scale.grad, offset.grad = None, None, None

    expected_raw = _tiled_quant_reference(data, scale, offset, num_bits, tile_size)
    expected_dequant = _tiled_dequant_reference(expected_raw, scale, offset, tile_size)

    expected_dequant.sum().backward()

    assert_close(quantized.raw_data, expected_raw.to(output_dtype))
    assert_close(dequantized, expected_dequant)

    if output_dtype.is_floating_point:
        assert_close(data_grad, data.grad, output_dtype)
        assert_close(scale_grad, scale.grad, output_dtype)
        assert_close(offset_grad, offset.grad, output_dtype)


@pytest.mark.slow
@pytest.mark.parametrize("num_bits", NUM_BITS)
@pytest.mark.parametrize("output_dtype", OUTPUT_DTYPES)
@pytest.mark.parametrize(
    "input_dtype",
    [
        torch.float16,
        torch.float32,
        torch.int32,
        torch.int16,
        torch.int8,
    ],
)
@pytest.mark.parametrize("scale_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "offset_dtype",
    [
        torch.float16,
        torch.float32,
        torch.int32,
        torch.int16,
        torch.int8,
    ],
)
def test_quantize_per_element_cuda(
    num_bits: int,
    output_dtype: torch.dtype,
    input_dtype: torch.dtype,
    scale_dtype: torch.dtype,
    offset_dtype: torch.dtype,
) -> None:
    _quantize_per_element_impl(
        "cuda", num_bits, output_dtype, input_dtype, scale_dtype, offset_dtype
    )


@pytest.mark.parametrize("num_bits", NUM_BITS)
@pytest.mark.parametrize("output_dtype", OUTPUT_DTYPES)
@pytest.mark.parametrize(
    "input_dtype",
    [
        torch.float16,
        torch.float32,
        torch.int32,
        torch.int16,
        torch.int8,
    ],
)
@pytest.mark.parametrize("scale_dtype", [torch.float32])
def test_quantize_per_element_cpu(
    num_bits: int, output_dtype: torch.dtype, input_dtype: torch.dtype, scale_dtype: torch.dtype
) -> None:
    _quantize_per_element_impl("cpu", num_bits, output_dtype, input_dtype, scale_dtype)


def gen_data(
    scale: float,
    offset: float,
    shape: tuple[int, ...] | torch.Size,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Generate test data.

    Generate data that is sufficiently far from the discontinuities of the
    quantization step function using scale. This ensures that rounding errors
    are less likely to occur during testing.
    """
    margin = 0.05
    precision = 1e-3

    int_precision = int(precision**-1)
    int_margin = int(margin * int_precision)
    lo, hi = -(int_precision // 2) + int_margin, (int_precision // 2) + 1 - int_margin

    data = torch.round(torch.randn(*shape) * 3 / scale - offset)
    noise = torch.randint(lo, hi, data.shape) / int_precision
    data = (data + noise + offset) * scale
    data = data.to(device)
    return data


def _quantize_per_element_impl(
    device: torch.device | str,
    num_bits: int,
    output_dtype: torch.dtype,
    input_dtype: torch.dtype,
    scale_dtype: torch.dtype,
    offset_dtype: torch.dtype | None = None,
) -> None:
    offset_dtype = offset_dtype if offset_dtype is not None else scale_dtype
    scale_ = 0.46
    offset_ = 2.0  # offset are rounded to integer values
    data_ = gen_data(scale_, offset_, (32, 16, 8), device)
    data = data_.to(input_dtype)
    if not input_dtype.is_floating_point:
        data *= 10

    if data.dtype.is_floating_point:
        data.requires_grad_()
    scale = torch.ones(
        data.numel(), device=device, dtype=scale_dtype, requires_grad=scale_dtype.is_floating_point
    )
    offset = torch.ones(
        data.numel(),
        device=device,
        dtype=offset_dtype,
        requires_grad=offset_dtype.is_floating_point,
    )
    with torch.no_grad():
        scale.fill_(scale_)
        offset.fill_(offset_)

    tile_size = torch.Size((1, 1, 1))
    quantized = affine.quantize_by_tile(data, scale, offset, tile_size, num_bits, output_dtype)
    dequantized = quantized.dequantize()

    if dequantized.grad_fn is not None:
        dequantized.sum().backward()
        data_grad, scale_grad, offset_grad = data.grad, scale.grad, offset.grad
        data.grad, scale.grad, offset.grad = None, None, None

        expected_raw = _tiled_quant_reference(data, scale, offset, num_bits, tile_size).to(
            output_dtype
        )
        expected_dequant = _tiled_dequant_reference(expected_raw, scale, offset, tile_size)
        expected_dequant = expected_dequant.to(data.dtype)

        torch.testing.assert_close(quantized.raw_data, expected_raw, atol=0.005, rtol=0.01)
        torch.testing.assert_close(dequantized, expected_dequant, atol=0.005, rtol=0.01)

        expected_dequant.sum().backward()

        if data_grad is not None:
            torch.testing.assert_close(data_grad, data.grad, atol=0.005, rtol=0.01)
        if offset_grad is not None:
            torch.testing.assert_close(offset_grad, offset.grad, atol=0.05, rtol=0.01)
        if scale_grad is not None:
            torch.testing.assert_close(scale_grad, scale.grad, atol=0.05, rtol=0.01)


def test_quantized_value_precision_loss() -> None:
    """Test that exception is raised when output_dtype is too small to keep the quantized value perprecisely."""
    # GIVEN a tensor
    data = torch.tensor([257.0], dtype=torch.float32)

    # WHEN the tensor is quantized to a bitwidth that cannot be represented in the provided output_dtype
    # THEN a RuntimeError is raised
    with pytest.raises(RuntimeError):
        affine.quantize_by_tile(
            data,
            torch.tensor([1.0]),
            torch.tensor([0.0]),
            torch.Size((1,)),
            16,
            output_dtype=torch.bfloat16,
        )
