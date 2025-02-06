# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
#
# Code adapted from https://github.com/huggingface/transformers
# Copyright 2018- The Hugging Face team. All rights reserved. Licensed under the Apache License, Version 2.0
# License is provided for attribution purposes only, Not a Contribution

import fastforward.nn.functional as FFF
import torch

from fastforward.nn import QuantizedModule, QuantizerMetadata
from fastforward.quantization.strict_quantization import strict_quantization
from transformers.models.llama.modeling_llama import LlamaRMSNorm


class QuantizedLlamaRMSNorm(LlamaRMSNorm, QuantizedModule):
    """LlamaRMSNorm is equivalent to T5LayerNorm."""

    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.input_quantizer = QuantizerMetadata(input_quantizer=True).to_stub()
        self.output_quantizer = QuantizerMetadata(output_quantizer=True).to_stub()
        self.weight_quantizer = QuantizerMetadata(weight_quantizer=True).to_stub()

    def forward(self, hidden_states):
        # NOTE: this module is not strict-quantized!
        with strict_quantization(False):
            hidden_states = self.input_quantizer(hidden_states)

            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            hidden_states = hidden_states.to(input_dtype)

            weight = self.weight_quantizer(self.weight)
            out = FFF.mul(weight, hidden_states, output_quantizer=self.output_quantizer)
        return out
