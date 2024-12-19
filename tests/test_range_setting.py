# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import cast

import torch

from torch import nn

import fastforward

from fastforward.nn.activations import QuantizedRelu
from fastforward.nn.linear import QuantizedLinear
from fastforward.nn.linear_quantizer import LinearQuantizer
from fastforward.nn.quantized_module import named_quantizers
from fastforward.quantization import granularity
from fastforward.range_setting import estimate_ranges, smoothed_minmax


@fastforward.flags.context(fastforward.strict_quantization, False)
def test_fit_quantizers():
    def make_fake_data():
        return torch.rand(3, 16)

    model = nn.Sequential(
        QuantizedLinear(16, 32, bias=False),
        QuantizedRelu(),
        QuantizedLinear(32, 64, bias=False),
        QuantizedRelu(),
    )
    model[0].input_quantizer = LinearQuantizer(8, symmetric=False)
    model[0].weight_quantizer = LinearQuantizer(8, granularity=granularity.PerChannel(-1))
    model[0].output_quantizer = LinearQuantizer(8, symmetric=False)
    model[1].output_quantizer = LinearQuantizer(8, symmetric=False)
    model[2].output_quantizer = LinearQuantizer(8, symmetric=False)
    model[2].weight_quantizer = LinearQuantizer(8, granularity=granularity.PerChannel(-1))
    model[3].output_quantizer = LinearQuantizer(8, symmetric=False)

    print("#1. Model pre-range setting", model, "", sep="\n")

    with estimate_ranges(model, smoothed_minmax(gamma=0.8)):
        print("#2. Model during range setting, pre-data", model, "", sep="\n")
        for _ in range(2):
            model(make_fake_data())

        print("#3. Model during range setting, post-data", model, "", sep="\n")

    print("#4. Model post range setting", model, "", sep="\n")

    print("#5. Final quantizer ranges")
    for name, quantizer in named_quantizers(model):
        quantizer = cast(LinearQuantizer, quantizer)

        print(f"{name=}")
        min, max = quantizer.quantization_range

        print(f"{quantizer.scale=}")
        print(f"{quantizer.offset=}")
        print(f"{min=}")
        print(f"{max=}")
        print()
