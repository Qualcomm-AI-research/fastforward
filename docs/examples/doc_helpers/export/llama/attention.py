# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#
# Code adapted from https://github.com/huggingface/transformers
# Copyright 2018- The Hugging Face team. All rights reserved. Licensed under the Apache License, Version 2.0
# License is provided for attribution purposes only, Not a Contribution

import math

import fastforward as ff
import fastforward.nn.functional as FFF
import torch

from fastforward.nn import QuantizedModule, QuantizerMetadata
from torch import nn
from transformers import Cache
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
)

from doc_helpers.export.llama.rotary_embedding import apply_quantized_rotary_pos_emb


class QuantizedLlamaAttention(LlamaAttention, QuantizedModule):
    """Multi-headed attention from 'Attention Is All You Need' paper."""

    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.attn_weights_quantizer = QuantizerMetadata(output_quantizer=True).to_stub()
        self.attn_weights_div_output_quantizer = QuantizerMetadata(output_quantizer=True).to_stub()
        self.q_quantizer = QuantizerMetadata(output_quantizer=True).to_stub()
        self.k_quantizer = QuantizerMetadata(output_quantizer=True).to_stub()
        self.attn_probs_quantizer = QuantizerMetadata(output_quantizer=True).to_stub()
        self.attn_re_quantizer = QuantizerMetadata(output_quantizer=True).to_stub()
        self.mask_add_quantizer = QuantizerMetadata(output_quantizer=True).to_stub()
        self.q_cos_quantizer = QuantizerMetadata(output_quantizer=True).to_stub()
        self.q_rot_quantizer = QuantizerMetadata(output_quantizer=True).to_stub()
        self.q_rot_sin_quantizer = QuantizerMetadata(output_quantizer=True).to_stub()
        self.k_cos_quantizer = QuantizerMetadata(output_quantizer=True).to_stub()
        self.k_rot_quantizer = QuantizerMetadata(output_quantizer=True).to_stub()
        self.k_rot_sin_quantizer = QuantizerMetadata(output_quantizer=True).to_stub()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        del use_cache
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(
            1, 2
        )
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_quantized_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            position_ids,
            q_quantizer=self.q_quantizer,
            k_quantizer=self.k_quantizer,
            q_cos_quantizer=self.q_cos_quantizer,
            q_rot_quantizer=self.q_rot_quantizer,
            q_rot_sin_quantizer=self.q_rot_sin_quantizer,
            k_cos_quantizer=self.k_cos_quantizer,
            k_rot_quantizer=self.k_rot_quantizer,
            k_rot_sin_quantizer=self.k_rot_sin_quantizer,
        )

        def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
            """This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).

            The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to (
            batch, num_attention_heads, seqlen, head_dim)
            """
            batch, num_key_value_heads, slen, head_dim = hidden_states.shape
            if n_rep == 1:
                return hidden_states
            dim_1_end = hidden_states.shape[0]
            dim_2_end = hidden_states.shape[1]
            dim_3_end = hidden_states.shape[2]
            dim_4_end = hidden_states.shape[3]
            hidden_states = hidden_states[0:dim_1_end, 0:dim_2_end, None, 0:dim_3_end, 0:dim_4_end]
            hidden_states = hidden_states.expand(batch, num_key_value_heads, n_rep, slen, head_dim)
            return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = FFF.matmul(
            query_states, key_states.transpose(2, 3), output_quantizer=self.attn_weights_quantizer
        )
        attn_weights = FFF.div(
            attn_weights,
            math.sqrt(self.head_dim),
            output_quantizer=self.attn_weights_div_output_quantizer,
        )

        with ff.set_strict_quantization(False):
            attn_weights = FFF.add(
                attn_weights, attention_mask, output_quantizer=self.mask_add_quantizer
            )

            attn_weights = FFF.softmax(
                attn_weights.to(torch.float32), dim=-1, output_quantizer=self.attn_probs_quantizer
            )
            attn_weights = attn_weights.to(query_states.dtype)
            attn_weights = nn.functional.dropout(
                attn_weights, p=self.attention_dropout, training=self.training
            )

            attn_output = FFF.matmul(
                attn_weights, value_states, output_quantizer=self.attn_re_quantizer
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
