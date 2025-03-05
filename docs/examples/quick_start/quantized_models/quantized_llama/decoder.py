# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
#
# Code adapted from https://github.com/huggingface/transformers
# Copyright 2018- The Hugging Face team. All rights reserved. Licensed under the Apache License, Version 2.0
# License is provided for attribution purposes only, Not a Contribution

import logging

import fastforward.nn.functional as FFF
import torch

from fastforward.nn import QuantizedModule, QuantizerMetadata
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

logger = logging.getLogger(__name__)


class QuantizedLlamaDecoderLayer(LlamaDecoderLayer, QuantizedModule):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()

        # activation quantizers
        self.input_quantizer = QuantizerMetadata(input_quantizer=True).to_stub()
        self.attn_res_act_quantizer = QuantizerMetadata(output_quantizer=True).to_stub()
        self.mlp_res_act_quantizer = QuantizerMetadata(output_quantizer=True).to_stub()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor]
        | None = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        """Quantized llama decoder layer forward.

        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            position_ids: ...
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model.
        """
        # quantize input
        hidden_states = self.input_quantizer(hidden_states)

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        # hidden_states = residual + hidden_states
        hidden_states = FFF.add(
            residual, hidden_states, output_quantizer=self.attn_res_act_quantizer
        )

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        # hidden_states = residual + hidden_states
        hidden_states = FFF.add(
            residual, hidden_states, output_quantizer=self.mlp_res_act_quantizer
        )

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
