# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#
# Code adapted from https://github.com/huggingface/transformers
# Copyright 2018- The Hugging Face team. All rights reserved. Licensed under the Apache License, Version 2.0
# License is provided for attribution purposes only, Not a Contribution

import fastforward as ff
import torch

from fastforward.nn import QuantizedModule
from transformers.models.llama.modeling_llama import LlamaRMSNorm


class QuantizedLlamaRMSNorm(LlamaRMSNorm, QuantizedModule):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.input_quantizer = ff.nn.QuantizerMetadata(input_quantizer=True).to_stub()
        self.output_quantizer = ff.nn.QuantizerMetadata(output_quantizer=True).to_stub()
        self.weight_quantizer = ff.nn.QuantizerMetadata(weight_quantizer=True).to_stub()

    def forward(self, hidden_states: torch.Tensor):
        with ff.set_strict_quantization(False):
            hidden_states = self.input_quantizer(hidden_states)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

            weight = self.weight_quantizer(self.weight)
            out = ff.nn.functional.mul(
                weight, hidden_states, output_quantizer=self.output_quantizer
            )

            return out
