# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import fastforward as ff
import pytest
import torch


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
def test_dynamic_linear_quantizer(granularity: ff.granularity.Granularity):
    num_bits = 4
    static_quantizer = ff.nn.LinearQuantizer(
        num_bits=num_bits, granularity=granularity, symmetric=False
    )
    dynamic_quantizer = ff.nn.DynamicLinearQuantizer(num_bits=num_bits, granularity=granularity)

    input = torch.randn((16, 16))
    with ff.estimate_ranges(static_quantizer, ff.range_setting.running_minmax):
        static_quantizer(input)

    expected = static_quantizer(input).dequantize()
    actual = dynamic_quantizer(input).dequantize()

    torch.testing.assert_close(actual, expected)

    # Test backwards pass
    input_with_backwards = torch.nn.Parameter(input)
    expected = static_quantizer(input_with_backwards).dequantize()
    (expected**2).sum().backward()
    assert input_with_backwards.grad is not None
    expected_grad = input_with_backwards.grad.clone()

    input_with_backwards.grad = None
    actual = dynamic_quantizer(input_with_backwards).dequantize()
    (actual**2).sum().backward()
    torch.testing.assert_close(input_with_backwards.grad, expected_grad)
