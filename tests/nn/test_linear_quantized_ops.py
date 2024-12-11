# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

# pylint: disable=missing-function-docstring
import pytest
import torch

import fastforward as ff

from fastforward.quantization.random import random_quantized


@pytest.mark.parametrize("data_shape", [(3, 3)])
def test_getitem_slice_rows(data_shape: tuple[int, int]):
    # Arrange
    rows, cols = data_shape
    qx = random_quantized(data_shape)
    x = qx.dequantize()
    sliced_rows_x = [x[row] for row in range(rows)]

    # Act:
    sliced_rows_qx = [qx[row] for row in range(rows)]

    # Assert:
    for sliced_qx, sliced_x in zip(sliced_rows_qx, sliced_rows_x):
        assert torch.all(sliced_qx.dequantize() == sliced_x)


@pytest.mark.parametrize("data_shape", [(3, 3)])
def test_getitem_slice_cols(data_shape: tuple[int, int]):
    # Arrange
    rows, cols = data_shape
    qx = random_quantized(data_shape)
    x = qx.dequantize()
    sliced_cols_x = [x[:, col] for col in range(cols)]

    # Act:
    sliced_cols_qx = [qx[:, col] for col in range(cols)]

    # Assert:
    for sliced_qx, sliced_x in zip(sliced_cols_qx, sliced_cols_x):
        assert torch.all(sliced_qx.dequantize() == sliced_x)


@pytest.mark.parametrize(
    "data_shape", [(3,), (3, 3), (3, 3, 3), (3, 3, 3, 3)], ids=["1D", "2D", "3D", "4D"]
)
def test_getitem_slice_multi_dims(data_shape: tuple[int]):
    # Arrange
    qx = random_quantized(data_shape)
    x = qx.dequantize()

    for dim in range(len(data_shape)):
        for index in range(data_shape[dim]):
            selector = tuple([slice(None, None, None) for _ in range(dim)] + [index])
            sliced_x = x[selector]

            # Act
            sliced_qx = qx[selector]

            # Assert
            assert torch.all(sliced_qx.dequantize() == sliced_x)


def test_scalar_multiply():
    qx = random_quantized((3, 3), requires_grad=True)
    scale_factor = 0.25
    with ff.set_strict_quantization(False):
        observed = ff.nn.functional.mul(qx, scale_factor)

    expected = qx.dequantize() * scale_factor
    torch.testing.assert_close(expected, observed.dequantize())


@pytest.mark.parametrize(
    "granularity",
    [
        ff.PerTensor(),
        ff.PerChannel(0),
        ff.PerChannel(1),
        ff.PerChannel(-1),
        ff.PerTile((4, 4)),
        ff.PerTile((8, 4)),
        ff.PerTile((16, 1)),
        ff.PerTile((2, 16)),
    ],
)
def test_unsqueeze(granularity):
    num_bits = 4

    qx = random_quantized(
        (16, 16),
        num_bits=num_bits,
        granularity=granularity,
    )

    for dim in [0, 1, 2, -1]:
        expected = qx.dequantize().unsqueeze(dim)
        actual = qx.unsqueeze(dim).dequantize()
        torch.testing.assert_close(actual, expected)


def test_expand():
    # Given: A quantized and non-quantized tensor that represent the same values
    quantized_data = ff.quantization.random.random_quantized((3, 3))
    data = quantized_data.dequantize()

    # When: Expand the second dimension of both tensors
    quantized_expanded = quantized_data.expand(3, 3, 3)
    expanded = data.expand(3, 3, 3)

    # Then: The dequantized and non-quantized tensor should match exactly.
    torch.testing.assert_close(quantized_expanded.dequantize(), expanded)
