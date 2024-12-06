# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import torch

import fastforward as ff


@ff.flags.context(ff.strict_quantization, False)
def test_quantized_relu_module() -> None:
    input_data = torch.randn(10, 10)
    relu_module = ff.nn.QuantizedRelu()

    output_data = relu_module(input_data)
    output_expected = torch.nn.functional.relu(input_data)

    assert (output_data == output_expected).all()


@ff.flags.context(ff.strict_quantization, False)
def test_quantized_silu_module() -> None:
    input_data = torch.randn(10, 10)
    silu_module = ff.nn.QuantizedSilu()

    output_data = silu_module(input_data)
    output_expected = torch.nn.functional.silu(input_data)

    assert (output_data == output_expected).all()
