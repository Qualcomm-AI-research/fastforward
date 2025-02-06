# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import torch

from typing_extensions import override

from fastforward.nn import QuantizedModule, Quantizer, QuantizerStub
from fastforward.nn.functional import linear


class QuantizedLinear(QuantizedModule, torch.nn.Linear):
    """Quantized implementation of torch.nn.Linear."""

    weight_quantizer: Quantizer
    bias_quantizer: Quantizer | None
    input_quantizer: Quantizer
    output_quantizer: Quantizer

    @override
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.input_quantizer = QuantizerStub(input_quantizer=True)
        self.weight_quantizer = QuantizerStub(weight_quantizer=True, shape=self.weight.shape)
        if self.bias is not None:
            self.bias_quantizer = QuantizerStub(bias_quantizer=True, shape=self.bias.shape)
        else:
            self.register_quantizer("bias_quantizer", None)  # type: ignore[unreachable]
        self.output_quantizer = QuantizerStub(output_quantizer=True)

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = self.input_quantizer(input)
        weight = self.weight_quantizer(self.weight)
        if self.bias is not None and self.bias_quantizer is not None:  # type: ignore[redundant-expr]
            bias = self.bias_quantizer(self.bias)
        else:
            bias = self.bias
        return linear(input, weight, bias, output_quantizer=self.output_quantizer)
