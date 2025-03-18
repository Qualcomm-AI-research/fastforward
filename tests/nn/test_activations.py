# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import fastforward as ff
import torch


@ff.flags.context(ff.strict_quantization, False)
def test_quantized_relu_module(_seed_prngs: int) -> None:
    input_data = torch.randn(10, 10)
    relu_module = ff.nn.QuantizedRelu()

    output_data = relu_module(input_data)
    output_expected = torch.nn.functional.relu(input_data)

    assert (output_data == output_expected).all()


@ff.flags.context(ff.strict_quantization, False)
def test_quantized_silu_module(_seed_prngs: int) -> None:
    input_data = torch.randn(10, 10)
    silu_module = ff.nn.QuantizedSilu()

    output_data = silu_module(input_data)
    output_expected = torch.nn.functional.silu(input_data)

    assert (output_data == output_expected).all()
