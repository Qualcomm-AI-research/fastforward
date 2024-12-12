# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

# pylint: disable=missing-function-docstring
from typing import Callable

import pytest
import torch

import fastforward as ff

# Functions that produce tensors that should be treated as linearly quantized.
LINEAR_QUANT_GENERATORS = (
    ff.random.random_quantized,
    ff.random.random_quantized_dynamic,
)


@pytest.mark.parametrize("data_shape", [(3, 3)])
@pytest.mark.parametrize("quantfunc", LINEAR_QUANT_GENERATORS)
def test_getitem_slice_rows(
    data_shape: tuple[int, int], quantfunc: Callable[..., ff.QuantizedTensor]
):
    # Arrange
    rows, cols = data_shape
    qx = quantfunc(data_shape)
    x = qx.dequantize()
    sliced_rows_x = [x[row] for row in range(rows)]

    # Act:
    sliced_rows_qx = [qx[row] for row in range(rows)]

    # Assert:
    for sliced_qx, sliced_x in zip(sliced_rows_qx, sliced_rows_x):
        assert torch.all(sliced_qx.dequantize() == sliced_x)


@pytest.mark.parametrize("data_shape", [(3, 3)])
@pytest.mark.parametrize("quantfunc", LINEAR_QUANT_GENERATORS)
def test_getitem_slice_cols(
    data_shape: tuple[int, int], quantfunc: Callable[..., ff.QuantizedTensor]
):
    # Arrange
    rows, cols = data_shape
    qx = quantfunc(data_shape)
    x = qx.dequantize()
    sliced_cols_x = [x[:, col] for col in range(cols)]

    # Act:
    sliced_cols_qx = [qx[:, col] for col in range(cols)]

    # Assert:
    for sliced_qx, sliced_x in zip(sliced_cols_qx, sliced_cols_x):
        assert torch.all(sliced_qx.dequantize() == sliced_x)


@pytest.mark.parametrize(
    "data_shape",
    [(3,), (3, 3), (3, 3, 3), (3, 3, 3, 3)],
    ids=["1D", "2D", "3D", "4D"],
)
@pytest.mark.parametrize("quantfunc", LINEAR_QUANT_GENERATORS)
def test_getitem_slice_multi_dims(
    data_shape: tuple[int], quantfunc: Callable[..., ff.QuantizedTensor]
):
    # Arrange
    qx = quantfunc(data_shape)
    x = qx.dequantize()

    for dim in range(len(data_shape)):
        for index in range(data_shape[dim]):
            selector = tuple([slice(None, None, None) for _ in range(dim)] + [index])
            sliced_x = x[selector]

            # Act
            sliced_qx = qx[selector]

            # Assert
            assert torch.all(sliced_qx.dequantize() == sliced_x)


@pytest.mark.parametrize("quantfunc", LINEAR_QUANT_GENERATORS)
def test_scalar_multiply(quantfunc: Callable[..., ff.QuantizedTensor]):
    # Given: a per-tensor quantized tensor and a scalar scale_factor
    qx = quantfunc((3, 3), requires_grad=True)
    scale = qx.quant_args().scale
    scale_factor = 0.25

    # When: a quantized tensor is multiplied by a scalar scale factor
    observed = ff.nn.functional.mul(qx, scale_factor)

    # Then: the result is a quantized tensor
    assert isinstance(observed, ff.QuantizedTensor)

    # Then: only the scale parameter is affected
    assert observed.quant_args().scale == scale * scale_factor

    # Then: the scale_factor has effect on the dequantized tensor
    expected = qx.dequantize() * scale_factor
    torch.testing.assert_close(expected, observed.dequantize())

    # When: computing the backward pass
    with ff.strict_quantization(False):
        expected.sum().backward()

    # Then: the scale correctly affects the gradient
    assert (qx.grad == scale_factor).all()  # type: ignore[union-attr]


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
@pytest.mark.parametrize("quantfunc", LINEAR_QUANT_GENERATORS)
def test_unsqueeze(granularity, quantfunc: Callable[..., ff.QuantizedTensor]):
    num_bits = 4

    qx = quantfunc(
        (16, 16),
        num_bits=num_bits,
        granularity=granularity,
    )

    for dim in [0, 1, 2, -1]:
        expected = qx.dequantize().unsqueeze(dim)
        actual = qx.unsqueeze(dim).dequantize()
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("quantfunc", LINEAR_QUANT_GENERATORS)
def test_expand(quantfunc: Callable[..., ff.QuantizedTensor]):
    """
    Test `torch.expand/torch.Tensor.expand` implementation for quantized
    tensors for which the following holds:

        - Per-Tensor quantized
        - Either dynamically or staticly quantized
            (i.e., fastforward.affine or fastforward.dynamic)
    """
    # Given: A quantized and non-quantized tensor that represent the same values
    quantized_data = quantfunc((3, 3))
    data = quantized_data.dequantize()

    # When: Expand the second dimension of both tensors
    quantized_expanded = quantized_data.expand(3, 3, 3)
    expanded = data.expand(3, 3, 3)

    # Then: The dequantized and non-quantized tensor should match exactly.
    torch.testing.assert_close(quantized_expanded.dequantize(), expanded)
