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
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


class QuantizedLlamaDecoderLayer(LlamaDecoderLayer, QuantizedModule):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.attn_res_act_quantizer = QuantizerMetadata(output_quantizer=True).to_stub()
        self.mlp_res_act_quantizer = QuantizerMetadata(output_quantizer=True).to_stub()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: tuple[torch.Tensor] | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        **kwargs,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

        with ff.set_strict_quantization(False):
            hidden_states = FFF.add(
                residual, hidden_states, output_quantizer=self.attn_res_act_quantizer
            )
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        with ff.set_strict_quantization(False):
            hidden_states = FFF.add(
                residual, hidden_states, output_quantizer=self.mlp_res_act_quantizer
            )

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
