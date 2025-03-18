# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import torch

from typing_extensions import override

from fastforward.nn import QuantizedModule, Quantizer, QuantizerStub
from fastforward.nn.functional import conv1d, conv2d


class QuantizedConv2d(QuantizedModule, torch.nn.Conv2d):
    """Quantized implementation of Conv2d."""

    weight_quantizer: Quantizer
    bias_quantizer: Quantizer | None
    input_quantizer: Quantizer
    output_quantizer: Quantizer

    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.input_quantizer = QuantizerStub(input_quantizer=True)
        self.weight_quantizer = QuantizerStub(weight_quantizer=True, shape=self.weight.shape)
        if self.bias is not None:
            self.bias_quantizer = QuantizerStub(bias_quantizer=True, shape=self.bias.shape)
        else:
            self.register_quantizer("bias_quantizer", None)
        self.output_quantizer = QuantizerStub(output_quantizer=True)

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = self.input_quantizer(input)
        weight = self.weight_quantizer(self.weight)
        bias: torch.Tensor | None
        if self.bias is not None and self.bias_quantizer is not None:
            bias = self.bias_quantizer(self.bias)
        else:
            bias = self.bias
        return conv2d(
            input,
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            output_quantizer=self.output_quantizer,
        )


class QuantizedConv1d(QuantizedModule, torch.nn.Conv1d):
    """Quantized implementation of Conv1d."""

    weight_quantizer: Quantizer
    bias_quantizer: Quantizer | None
    input_quantizer: Quantizer
    output_quantizer: Quantizer

    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.input_quantizer = QuantizerStub(input_quantizer=True)
        self.weight_quantizer = QuantizerStub(weight_quantizer=True, shape=self.weight.shape)
        if self.bias is not None:
            self.bias_quantizer = QuantizerStub(bias_quantizer=True, shape=self.bias.shape)
        else:
            self.register_quantizer("bias_quantizer", None)
        self.output_quantizer = QuantizerStub(output_quantizer=True)

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = self.input_quantizer(input)
        weight = self.weight_quantizer(self.weight)
        bias: torch.Tensor | None
        if self.bias is not None and self.bias_quantizer is not None:
            bias = self.bias_quantizer(self.bias)
        else:
            bias = self.bias
        return conv1d(
            input,
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            output_quantizer=self.output_quantizer,
        )
