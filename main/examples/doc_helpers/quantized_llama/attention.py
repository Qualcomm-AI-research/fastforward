# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#
# Code adapted from https://github.com/huggingface/transformers/blob/v5.9.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2018- The Hugging Face team. All rights reserved. Licensed under the Apache License, Version 2.0
# License is provided for attribution purposes only, Not a Contribution

import logging

import fastforward.nn.functional as FFF
import torch
import torch.nn.functional as F

from fastforward.nn import QuantizedModule, QuantizerMetadata
from transformers.models.llama.modeling_llama import LlamaAttention, repeat_kv

from .rotary_embedding import apply_rotary_pos_emb

logger = logging.getLogger(__name__)


class QuantizedLlamaAttention(LlamaAttention, QuantizedModule):
    """Quantized multi-headed attention for Llama models (transformers >= 5.x)."""

    def __init_quantization__(self) -> None:
        super().__init_quantization__()

        self.input_quantizer = QuantizerMetadata(input_quantizer=True).to_stub()
        self.attn_weights_quantizer = QuantizerMetadata(output_quantizer=True).to_stub()
        self.attn_probs_quantizer = QuantizerMetadata(output_quantizer=True).to_stub()
        self.attn_output_quantizer = QuantizerMetadata(output_quantizer=True).to_stub()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        **_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        hidden_states = self.input_quantizer(hidden_states)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        assert position_embeddings is not None
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = FFF.matmul(
            query_states,
            key_states.transpose(2, 3),
            output_quantizer=self.attn_weights_quantizer,
        )
        attn_weights = attn_weights * self.scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = FFF.softmax(
            attn_weights.to(torch.float32),
            dim=-1,
            output_quantizer=self.attn_probs_quantizer,
        ).to(query_states.dtype)
        attn_weights = FFF.dropout(
            attn_weights,
            p=float(self.attention_dropout or 0.0) if self.training else 0.0,
            training=self.training,
            output_quantizer=self.attn_probs_quantizer,
        )

        attn_output = FFF.matmul(
            attn_weights, value_states, output_quantizer=self.attn_output_quantizer
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class QuantizedLlamaSDPAttention(LlamaAttention, QuantizedModule, include_in_module_map=False):
    """Quantized attention using fused scaled_dot_product_attention (no intermediate quantizers)."""

    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.input_quantizer = QuantizerMetadata(input_quantizer=True).to_stub()

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        **_kwargs,
    ) -> tuple[torch.Tensor, None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        hidden_states = self.input_quantizer(hidden_states)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        assert position_embeddings is not None
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # HF's sdpa path returns attention_mask=None to signal "plain causal" (so SDPA
        # can dispatch to the fused kernel via is_causal rather than a materialized
        # mask). Mirror stock LlamaAttention: reconstruct causality with is_causal when
        # no mask is provided. A single-token query (q_len == 1) needs no causal mask.
        is_causal = attention_mask is None and query_states.shape[-2] > 1

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=float(self.attention_dropout or 0.0) if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None
