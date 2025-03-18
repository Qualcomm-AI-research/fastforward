# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import torch

from typing_extensions import override

from fastforward.nn import QuantizedModule, QuantizerStub, functional


class QuantizedEmbedding(torch.nn.Embedding, QuantizedModule):
    """Quantized implementation of torch.nn.Embedding."""

    @override
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.weight_quantizer = QuantizerStub(weight_quantizer=True)
        self.output_quantizer = QuantizerStub(output_quantizer=True)

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return functional.embedding(
            input,
            self.weight_quantizer(self.weight),
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
            output_quantizer=self.output_quantizer,
        )
