# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import torch

from typing_extensions import override

from fastforward.nn import QuantizedModule, QuantizerStub, functional


class QuantizedActivation(QuantizedModule, include_in_module_map=False):
    """Base class for quantized activations."""

    @override
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.input_quantizer = QuantizerStub(input_quantizer=True)
        self.output_quantizer = QuantizerStub(output_quantizer=True)


class QuantizedRelu(torch.nn.ReLU, QuantizedActivation):
    """Applies quantized Relu.

    # Quantizers
    1. input_quantizer: input activation before relu is applied.
    2. output_quantizer: output activation after relu is applied.
    """

    @override
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.inplace = False

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = self.input_quantizer(input)
        return functional.relu(input, output_quantizer=self.output_quantizer)


class QuantizedSilu(torch.nn.SiLU, QuantizedActivation):
    """Applies quantized Silu.

    # Quantizers
    1. input_quantizer: input activation before silu is applied.
    2. output_quantizer: output activation after silu is applied.
    """

    @override
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.inplace = False

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = self.input_quantizer(input)
        return functional.silu(input, output_quantizer=self.output_quantizer)
