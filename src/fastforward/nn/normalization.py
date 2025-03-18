# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import torch

from typing_extensions import override

import fastforward.nn as ffnn


class QuantizedLayerNorm(torch.nn.LayerNorm, ffnn.QuantizedModule):
    """Quantized implementation of torch.nn.LayerNorm."""

    weight: torch.Tensor | None  # type: ignore[assignment]
    bias: torch.Tensor | None  # type: ignore[assignment]

    @override
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.input_quantizer = ffnn.QuantizerStub(input_quantizer=True)

        if self.elementwise_affine:
            self.weight_quantizer = ffnn.QuantizerStub(weight_quantizer=True)
            self.bias_quantizer = ffnn.QuantizerStub(bias_quantizer=True)
        else:
            self.register_quantizer("weight_quantizer", None)
            self.register_quantizer("bias_quantizer", None)
        self.output_quantizer = ffnn.QuantizerStub(output_quantizer=True)

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = self.input_quantizer(input)
        weight = self.weight
        bias = self.bias

        if weight is not None and self.weight_quantizer is not None:
            weight = self.weight_quantizer(self.weight)
        if bias is not None and self.bias_quantizer is not None:
            bias = self.bias_quantizer(self.bias)

        return ffnn.functional.layer_norm(
            input,
            self.normalized_shape,
            weight=weight,
            bias=bias,
            eps=self.eps,
            output_quantizer=self.output_quantizer,
        )
