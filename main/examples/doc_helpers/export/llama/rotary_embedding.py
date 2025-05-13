# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#
# Code adapted from https://github.com/huggingface/transformers
# Copyright 2018- The Hugging Face team. All rights reserved. Licensed under the Apache License, Version 2.0
# License is provided for attribution purposes only, Not a Contribution

import fastforward as ff
import torch

from fastforward.nn import QuantizedModule
from fastforward.nn import functional as FFF
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding  # rotate_half


class QuantizedLlamaRotaryEmbedding(LlamaRotaryEmbedding, QuantizedModule):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()

    def forward(self, x, position_ids):
        inv_freq_dimensions = self.inv_freq.shape
        inv_freq_expanded = (
            self.inv_freq[None, 0 : inv_freq_dimensions[0], None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )

        position_ids_dimensions = position_ids.shape
        position_ids_expanded = position_ids[
            0 : position_ids_dimensions[0], None, 0 : position_ids_dimensions[1]
        ].float()

        device_type = (
            x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos, sin


def apply_quantized_rotary_pos_emb(
    q,
    k,
    cos,
    sin,
    position_ids,
    q_quantizer,
    k_quantizer,
    q_cos_quantizer,
    q_rot_quantizer,
    q_rot_sin_quantizer,
    k_cos_quantizer,
    k_rot_quantizer,
    k_rot_sin_quantizer,
    unsqueeze_dim=1,
):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        q_quantizer: Activation quantizer
        k_quantizer: Activation quantizer
        q_cos_quantizer: Activation quantizer
        q_rot_quantizer: Activation quantizer
        q_rot_sin_quantizer: Activation quantizer
        k_cos_quantizer: Activation quantizer
        k_rot_quantizer: Activation quantizer
        k_rot_sin_quantizer: Activation quantizer
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.

    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    def rotate_half(x, output_quantizer):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 : x.shape[-1]]
        neg_x2 = FFF.mul(x2, -1, output_quantizer=output_quantizer)
        return torch.cat((neg_x2, x1), dim=-1)

    with ff.set_strict_quantization(False):
        q_cos = FFF.mul(q, cos, output_quantizer=q_cos_quantizer)
        q_rot = rotate_half(q, output_quantizer=q_rot_quantizer)
        q_rot_sin = FFF.mul(q_rot, sin, output_quantizer=q_rot_sin_quantizer)
        q_embed = FFF.add(q_cos, q_rot_sin, output_quantizer=q_quantizer)

        k_cos = FFF.mul(k, cos, output_quantizer=k_cos_quantizer)
        k_rot = rotate_half(k, output_quantizer=k_rot_quantizer)
        k_rot_sin = FFF.mul(k_rot, sin, output_quantizer=k_rot_sin_quantizer)
        k_embed = FFF.add(k_cos, k_rot_sin, output_quantizer=k_quantizer)

        return q_embed, k_embed
