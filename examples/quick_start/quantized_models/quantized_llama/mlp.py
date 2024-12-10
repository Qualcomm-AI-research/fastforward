# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import logging

from transformers.models.llama.modeling_llama import LlamaMLP

import fastforward.nn.functional as FFF

from fastforward.nn import QuantizedModule, QuantizerMetadata

logger = logging.getLogger(__name__)


class QuantizedLlamaMLP(LlamaMLP, QuantizedModule):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.input_quantizer = QuantizerMetadata(input_quantizer=True).to_stub()
        self.gate_act_quantizer = QuantizerMetadata(input_quantizer=True).to_stub()
        self.gated_up_proj_output_quantizer = QuantizerMetadata(output_quantizer=True).to_stub()

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            raise NotImplementedError()

        x = self.input_quantizer(x)
        gate = self.gate_proj(x)
        gate_act = self.act_fn(gate)
        gated_up_proj = FFF.mul(
            gate_act, self.up_proj(x), output_quantizer=self.gated_up_proj_output_quantizer
        )
        down_proj = self.down_proj(gated_up_proj)

        return down_proj
