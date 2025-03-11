# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import fastforward as ff
import torch


@ff.flags.context(ff.strict_quantization, False)
def test_quantized_conv1d_module(_seed_prngs: int) -> None:
    input_data = torch.randn(10, 2, 4)
    conv1d_module = ff.nn.QuantizedConv1d(2, 2, 2, 1)
    weight = conv1d_module.weight
    bias = conv1d_module.bias

    output_data = conv1d_module(input_data)
    output_expected = torch.nn.functional.conv1d(input_data, weight, bias)

    assert (output_data == output_expected).all()


@ff.flags.context(ff.strict_quantization, False)
def test_quantized_conv2d_module(_seed_prngs: int) -> None:
    input_data = torch.randn(10, 2, 4, 4)
    conv2d_module = ff.nn.QuantizedConv2d(2, 2, 2, 1)
    weight = conv2d_module.weight
    bias = conv2d_module.bias

    output_data = conv2d_module(input_data)
    output_expected = torch.nn.functional.conv2d(input_data, weight, bias)

    assert (output_data == output_expected).all()
