# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from fastforward.nn import QuantizedModule
from fastforward.quantization.strict_quantization import strict_quantization
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, rotate_half


class QuantizedLlamaRotaryEmbedding(LlamaRotaryEmbedding, QuantizedModule):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()


def apply_rotary_pos_emb(
    q,
    k,
    cos,
    sin,
    position_ids=None,
    unsqueeze_dim=1,
    q_output_quantizer=None,
    k_output_quantizer=None,
):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        q_output_quantizer: output quantizer for query output.
        k_output_quantizer: output quantizer for key output.

    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # NOTE: this operation is not quantized
    with strict_quantization(False):
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)

        # if isinstance(q, QuantizedTensor):
        #     q = q.dequantize()
        #
        # if isinstance(k, QuantizedTensor):
        #     k = k.dequantize()

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)

        # q_embed = FFF.add(q * cos, rotate_half(q) * sin, output_quantizer=q_quantizer)
        # k_embed = FFF.add(k * cos, rotate_half(k) * sin, output_quantizer=k_quantizer)

        if q_output_quantizer is not None:
            q_embed = q_output_quantizer(q_embed)
        if k_output_quantizer is not None:
            k_embed = k_output_quantizer(k_embed)

    return q_embed, k_embed
