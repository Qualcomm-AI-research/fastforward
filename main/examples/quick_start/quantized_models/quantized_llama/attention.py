# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#
# Code adapted from https://github.com/huggingface/transformers
# Copyright 2018- The Hugging Face team. All rights reserved. Licensed under the Apache License, Version 2.0
# License is provided for attribution purposes only, Not a Contribution

import logging  # noqa: I001
import math


import torch

from transformers import Cache
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaFlashAttention2,
    LlamaSdpaAttention,
    repeat_kv,
)

from fastforward.nn import QuantizedModule, QuantizerMetadata

from .rotary_embedding import apply_rotary_pos_emb

from fastforward.dispatcher import dispatch, register
from fastforward.nn import Quantizer
from fastforward import get_strict_quantization, set_strict_quantization
from fastforward.quantized_tensor import QuantizedTensor
from torch import Tensor
from fastforward.nn import functional as FFF

logger = logging.getLogger(__name__)


class QuantizedLlamaFlashAttention2(LlamaFlashAttention2, QuantizedModule):
    """Llama flash attention module.

    This module inherits from `LlamaAttention` as the weights of the module
    stays untouched. The only required change would be on the forward pass
    where it needs to correctly call the public API of flash attention and deal
    with padding tokens in case the input contains any of them.
    """

    def __init_quantization__(self) -> None:
        super().__init_quantization__()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor]
        | None = None,  # will become mandatory in v4.46
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        raise NotImplementedError()


class QuantizedLlamaAttention(LlamaAttention, QuantizedModule):
    """Multi-headed attention from 'Attention Is All You Need' paper."""

    def __init_quantization__(self) -> None:
        super().__init_quantization__()

        # activation quantizers
        self.input_quantizer = QuantizerMetadata(input_quantizer=True).to_stub()
        self.attn_weights_quantizer = QuantizerMetadata(output_quantizer=True).to_stub()
        # [X] self.attn_weights_div_output_quantizer = QuantizerMetadata(output_quantizer=True).to_stub()
        # [?] self.q_quantizer = QuantizerMetadata(output_quantizer=True).to_stub()
        # [?] self.k_quantizer = QuantizerMetadata(output_quantizer=True).to_stub()
        self.attn_probs_quantizer = QuantizerMetadata(output_quantizer=True).to_stub()
        self.attn_output_quantizer = QuantizerMetadata(output_quantizer=True).to_stub()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor]
        | None = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        bsz, q_len, _ = hidden_states.size()

        # quantize inputs
        hidden_states = self.input_quantizer(hidden_states)

        if self.config.pretraining_tp > 1:
            raise NotImplementedError()
        else:
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

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = FFF.matmul(
            query_states, key_states.transpose(2, 3), output_quantizer=self.attn_weights_quantizer
        )
        attn_weights = FFF.div(
            attn_weights, math.sqrt(self.head_dim), output_quantizer=self.attn_weights_quantizer
        )
        # NOTE: /= math.sqrt(self.head_dim) can be absorbed into the scale

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = FFF.softmax(
            attn_weights.to(torch.float32), dim=-1, output_quantizer=self.attn_probs_quantizer
        )
        attn_weights = attn_weights.to(query_states.dtype)

        # attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_weights = FFF.dropout(
            attn_weights,
            p=self.attention_dropout,
            training=self.training,
            output_quantizer=self.attn_probs_quantizer,
        )

        # attn_output = torch.matmul(attn_weights, value_states)
        attn_output = FFF.matmul(
            attn_weights, value_states, output_quantizer=self.attn_output_quantizer
        )

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            raise NotImplementedError()
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class QuantizedLlamaSdpaAttention(LlamaSdpaAttention, QuantizedModule):
    """Llama attention module using torch.nn.functional.scaled_dot_product_attention.

    This module inherits from `LlamaAttention` as the weights of the module
    stays untouched. The only changes are on the forward pass to adapt to SDPA
    API.
    """

    def __init_quantization__(self) -> None:
        super().__init_quantization__()

        self.input_quantizer = QuantizerMetadata(input_quantizer=True).to_stub()

        self.rope_q_quantizer = QuantizerMetadata(output_quantizer=True).to_stub()
        self.rope_k_quantizer = QuantizerMetadata(output_quantizer=True).to_stub()

        # scaled_dot_product_attention quantizers sdpa
        self.sdpa_matmul_quantizer = QuantizerMetadata(output_quantizer=True).to_stub()
        self.sdpa_softmax_quantizer = QuantizerMetadata(output_quantizer=True).to_stub()
        self.sdpa_dropout_quantizer = QuantizerMetadata(output_quantizer=True).to_stub()
        self.sdpa_output_quantizer = QuantizerMetadata(output_quantizer=True).to_stub()

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor]
        | None = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        if output_attentions:
            raise NotImplementedError()

        bsz, q_len, _ = hidden_states.size()

        hidden_states = self.input_quantizer(hidden_states)

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

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            q_output_quantizer=self.rope_q_quantizer,
            k_output_quantizer=self.rope_k_quantizer,
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False

        # attn_output = torch.nn.functional.scaled_dot_product_attention(
        #     query_states,
        #     key_states,
        #     value_states,
        #     attn_mask=causal_mask,
        #     dropout_p=self.attention_dropout if self.training else 0.0,
        #     is_causal=is_causal,
        # )
        attn_output = scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
            matmul_k_quantizer=self.sdpa_matmul_quantizer,
            softmax_quantizer=self.sdpa_softmax_quantizer,
            dropout_quantizer=self.sdpa_dropout_quantizer,
            output_quantizer=self.sdpa_output_quantizer,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask: QuantizedTensor | None = None,
    dropout_p=0.0,
    is_causal=False,
    scale: float = None,
    *,
    matmul_k_quantizer: Quantizer | None = None,
    softmax_quantizer: Quantizer | None = None,
    dropout_quantizer: Quantizer | None = None,
    output_quantizer: Quantizer | None = None,
    strict_quantization: bool | None = None,
) -> torch.Tensor:
    if strict_quantization is None:
        strict_quantization = get_strict_quantization()

    dispatch_op = dispatch(
        "scaled_dot_product_attention",
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        matmul_k_quantizer=matmul_k_quantizer,
        softmax_quantizer=softmax_quantizer,
        dropout_quantizer=dropout_quantizer,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )
    selected_op = dispatch_op

    return selected_op(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        matmul_k_quantizer=matmul_k_quantizer,
        softmax_quantizer=softmax_quantizer,
        dropout_quantizer=dropout_quantizer,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )


@register("scaled_dot_product_attention", None)
def _scaled_dot_product_attention(
    query: QuantizedTensor,
    key: QuantizedTensor,
    value: QuantizedTensor,
    attn_mask: QuantizedTensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float = None,
    matmul_k_quantizer: Quantizer | None = None,
    softmax_quantizer: Quantizer | None = None,
    dropout_quantizer: Quantizer | None = None,
    output_quantizer: Quantizer | None = None,
    strict_quantization: bool | None = None,
) -> QuantizedTensor | Tensor:
    """Quantized version of scaled_dot_product_attention from torch documentation.

    Reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    """
    # with torch.nn.attention.sdpa_kernel(SDPBackend.MATH):
    #     return torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask, dropout_p, is_causal)

    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    with set_strict_quantization(False):
        # with estimate_ranges([matmul_k_quantizer], running_minmax):
        attn_scores = FFF.matmul(
            query,
            key.transpose(-2, -1),
            output_quantizer=matmul_k_quantizer,
            strict_quantization=strict_quantization,
        )

        # attn_scores = attn_scores.dequantize()
        # !!! This is breaking everything if attn_scores is quantized!!!
        attn_scores = FFF.mul(attn_scores, scale_factor)
        # attn_scores *= scale_factor
        attn_scores += attn_bias.to(attn_scores.device)

    attn_weight = FFF.softmax(
        attn_scores,
        dim=-1,
        output_quantizer=softmax_quantizer,
        strict_quantization=strict_quantization,
    )

    attn_weight = FFF.dropout(
        attn_weight,
        dropout_p,
        training=True,
        output_quantizer=dropout_quantizer,
        strict_quantization=strict_quantization,
    )

    return FFF.matmul(
        attn_weight,
        value,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )

    #
    # #####################
    # def _get_sqnr_per_sample_markus(org_out, quant_out, in_db=True, eps=1e-15):
    #     org_out = org_out.to(torch.float32)
    #     quant_out = quant_out.to(torch.float32)
    #
    #     quant_error = org_out - quant_out
    #     exp_noise = quant_error.pow(2).view(quant_error.shape[0], -1).mean(1) + eps
    #     exp_signal = org_out.pow(2).view(org_out.shape[0], -1).mean(1)
    #     sqnr = (exp_signal / exp_noise)
    #     sqnr_db = 10 * torch.log10(sqnr)
    #     return sqnr_db if in_db else sqnr
    #
    # attn_scores_orig = torch.matmul(query, key.transpose(-2, -1))
    # sqnr = _get_sqnr_per_sample_markus(attn_scores_orig, attn_scores)
    #
    # if sqnr.min() < 60:
    #     err = attn_scores_orig - attn_scores.dequantize()
    #     max_err = err.max()
    #     argmax = err.argmax()
    #     argmax_multidim = torch.unravel_index(argmax, err.shape)
    #     max_err2 = err[argmax_multidim]
    #
    #     print(max_err)
    #     print(max_err2)
    #
    # attn_scores_orig = attn_scores_orig.cpu()
    # del attn_scores_orig
    ############################
