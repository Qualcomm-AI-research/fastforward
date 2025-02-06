# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

# pylint: disable=missing-function-docstring
import itertools

from typing import Callable, Iterator

import pytest
import torch

import fastforward as ff

# Functions that produce tensors that should be treated as linearly quantized.
LINEAR_QUANT_GENERATORS = (ff.random.random_quantized,)


def _dim_slices_and_indices(dimlen: int) -> Iterator[slice | int]:
    """
    Generate various legal indexers (i.e., slice or integers) for a
    tensor dimension of size `dimlen`.
    """
    yield slice(None, None, None)  # case ':', e.g, x[:]

    # Positive indices
    for start in range(dimlen):
        yield start  # case x[start]
        for end in range(start, dimlen + 1):
            yield slice(start, end, None)  # case x[start:end]
            for step in range(1, end - start + 1):
                yield slice(start, end, step)  # case x[start:end:step]


@pytest.mark.parametrize("data_shape", [(3, 3), (), (1, 3), (3, 1), (1, 0, 2), (3,)])
@pytest.mark.parametrize("quantfunc", LINEAR_QUANT_GENERATORS)
def test_getitem(data_shape: tuple[int, int], quantfunc: Callable[..., ff.QuantizedTensor]):
    """
    Test QuantizedTensor.__getitem__ implementation of Per-Tensor quantized
    tensors for many different ways of indexing.
    """

    # Given: a quantized and dequantized tensor
    torch.randn(data_shape)
    qx = quantfunc(data_shape)
    x = qx.dequantize()

    # Iterate over a large set of possible ways of indexing a tensor
    for slices_and_indices in itertools.product(
        *[_dim_slices_and_indices(dim) for dim in data_shape]
    ):
        for i in range(1, len(slices_and_indices) + 1):
            # When: one or more dimensions are indexed
            qx_slice = qx[slices_and_indices[:i]]
            x_slice = x[slices_and_indices[:i]]

            # Then: indexing before or after dequantization should result in the same tensor
            torch.testing.assert_close(qx_slice.dequantize(), x_slice, rtol=0, atol=0)

        if len(slices_and_indices) == 0:
            continue

        # When: the first dimension is indexed
        qx_slice = qx[slices_and_indices[0]]
        x_slice = x[slices_and_indices[0]]

        # Then: indexing before or after dequantization should result in the same tensor
        torch.testing.assert_close(qx_slice.dequantize(), x_slice, rtol=0, atol=0)


@pytest.mark.parametrize("quantfunc", LINEAR_QUANT_GENERATORS)
def test_scalar_multiply(quantfunc: Callable[..., ff.QuantizedTensor]):
    # Given: a per-tensor quantized tensor and a scalar scale_factor
    qx = quantfunc((3, 3), requires_grad=True)
    scale = qx.quant_args().scale  # type: ignore[attr-defined]
    scale_factor = 0.25

    # When: a quantized tensor is multiplied by a scalar scale factor
    observed = ff.nn.functional.mul(qx, scale_factor)

    # Then: the result is a quantized tensor
    assert isinstance(observed, ff.QuantizedTensor)

    # Then: only the scale parameter is affected
    assert observed.quant_args().scale == scale * scale_factor  # type: ignore[attr-defined]

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
    torch.testing.assert_close(quantized_expanded.dequantize(), expanded, rtol=0, atol=0)


@pytest.mark.parametrize(
    "granularity",
    [
        ff.PerChannel(0),
        ff.PerChannel(1),
        ff.PerChannel(-1),
        ff.PerChannel((0, 1, 2)),
        ff.PerChannel((1, 2)),
        ff.PerChannel((0, 2)),
    ],
)
@pytest.mark.parametrize("quantfunc", LINEAR_QUANT_GENERATORS)
def test_getitem_perchannel(granularity, quantfunc: Callable[..., ff.QuantizedTensor]):
    """
    Test QuantizedTensor.__getitem__ implementation of Per-Channel quantized
    tensors for many different ways of indexing.
    """

    # Given: a quantized and dequantized tensor
    data_shape = (3, 2, 2)
    x_in = torch.randn(data_shape)
    qx = ff.nn.DynamicLinearQuantizer(4, granularity=granularity)(x_in)
    x = qx.dequantize()

    # Iterate over a large set of possible ways of indexing a tensor
    for slices_and_indices in itertools.product(
        *[_dim_slices_and_indices(dim) for dim in data_shape]
    ):
        for i in range(1, len(slices_and_indices) + 1):
            # When: one or more dimensions are indexed
            slicer = slices_and_indices[:i]
            qx_slice = qx[slices_and_indices[:i]]
            x_slice = x[slices_and_indices[:i]]

            # Then: indexing before or after dequantization should result in the same tensor
            torch.testing.assert_close(qx_slice.dequantize(), x_slice, rtol=0, atol=0)

        if len(slices_and_indices) == 0:
            continue

        # When: the first dimension is indexed
        qx_slice = qx[slices_and_indices[0]]
        x_slice = x[slices_and_indices[0]]

        # Then: indexing before or after dequantization should result in the same tensor
        torch.testing.assert_close(qx_slice.dequantize(), x_slice, rtol=0, atol=0)
