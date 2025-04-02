# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#
# Code adapted from https://github.com/huggingface/transformers
# Copyright 2018- The Hugging Face team. All rights reserved. Licensed under the Apache License, Version 2.0
# License is provided for attribution purposes only, Not a Contribution

import fastforward as ff
import fastforward.nn.functional as FFF
import torch

from fastforward.nn import QuantizedModule, QuantizerMetadata
from transformers.models.llama.modeling_llama import LlamaMLP


class QuantizedLlamaMLP(LlamaMLP, QuantizedModule):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.gated_up_proj_output_quantizer = QuantizerMetadata(output_quantizer=True).to_stub()

    def forward(self, x: torch.Tensor):
        gate = self.gate_proj(x)
        with ff.set_strict_quantization(False):
            gate_act = self.act_fn(gate)

            gated_up_proj = FFF.mul(
                gate_act,
                self.up_proj(x),
                output_quantizer=self.gated_up_proj_output_quantizer,
            )
            down_proj = self.down_proj(gated_up_proj)
        return down_proj
