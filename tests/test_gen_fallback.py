# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import fastforward as ff
import pytest
import torch

from fastforward._gen import fallback


@pytest.fixture
def quantizer() -> ff.nn.LinearQuantizer:
    quantizer = ff.nn.LinearQuantizer(num_bits=4)
    quantizer.quantization_range = (-1, 1)
    return quantizer


def test_div(quantizer: ff.nn.LinearQuantizer) -> None:
    input = ff.random.random_quantized((3, 3))
    result = fallback.div(input, 3.0, output_quantizer=quantizer)

    assert (result.dequantize() == quantizer(input.dequantize() / 3).dequantize()).all()


def test_cat(quantizer: ff.nn.LinearQuantizer) -> None:
    input_1 = ff.random.random_quantized((3, 3))
    input_2 = ff.random.random_quantized((3, 3))
    input_3 = ff.random.random_quantized((3, 3))
    inputs = (input_1, input_2, input_3)

    result = fallback.cat(inputs, 0, output_quantizer=quantizer)
    expected = quantizer(torch.cat([inp.dequantize() for inp in inputs], 0)).dequantize()

    assert (result.dequantize() == expected.dequantize()).all()
