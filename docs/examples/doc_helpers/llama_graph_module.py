# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


"""'Linearized' GraphModule representation of a Llama v1 model."""

import torch

from fastforward._orchestration.graph_module import GraphModule
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaModel,
    apply_rotary_pos_emb,
)


class ReshapeQKV(torch.nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.n_heads: int = config.num_attention_heads
        self.head_dim: int = config.head_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)


class ApplyRope(torch.nn.Module):
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return apply_rotary_pos_emb(q, k, cos, sin)


class SDPA(torch.nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.dropout_p = config.attention_dropout

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p, is_causal=True
        )


class ReshapeOutput(torch.nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.n_heads: int = config.num_attention_heads
        self.head_dim: int = config.head_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().reshape(bsz, seq_len, self.n_heads * self.head_dim)


def attention(self_attn: LlamaAttention) -> GraphModule:
    """Generate a Llama Attention layer GraphModule."""
    graph = GraphModule()

    hidden = graph.add_input("hidden")
    position_embeddings = graph.add_input("position_embeddings")
    cos = position_embeddings[0]
    sin = position_embeddings[1]

    # Q, K, V projections (quantizable linear layers)
    q = graph.add_node("q_proj", self_attn.q_proj, [hidden])
    k = graph.add_node("k_proj", self_attn.k_proj, [hidden])
    v = graph.add_node("v_proj", self_attn.v_proj, [hidden])

    q_reshaped = graph.add_node("q_reshaped", ReshapeQKV(self_attn.config), [q])
    k_reshaped = graph.add_node("k_reshaped", ReshapeQKV(self_attn.config), [k])
    v_reshaped = graph.add_node("v_reshaped", ReshapeQKV(self_attn.config), [v])

    rope_out = graph.add_node("apply_rope", ApplyRope(), [q_reshaped, k_reshaped, cos, sin])
    q_rope = rope_out[0]
    k_rope = rope_out[1]

    attn_output = graph.add_node("sdpa", SDPA(self_attn.config), [q_rope, k_rope, v_reshaped])

    attn_reshaped = graph.add_node("reshape_output", ReshapeOutput(self_attn.config), [attn_output])
    output = graph.add_node("o_proj", self_attn.o_proj, [attn_reshaped])

    graph.add_output(output)
    return graph


class MLPActivation(torch.nn.Module):
    """MLP activation: SiLU(gate) * up."""

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(gate) * up


def mlp(llama_mlp: LlamaMLP) -> GraphModule:
    """Generate a complete Llama MLP layer GraphModule."""
    graph = GraphModule()
    hidden = graph.add_input("hidden")

    gate = graph.add_node("gate_proj", llama_mlp.gate_proj, [hidden])
    up = graph.add_node("up_proj", llama_mlp.up_proj, [hidden])
    mlp_hidden = graph.add_node("mlp_activation", MLPActivation(), [gate, up])

    output = graph.add_node("down_proj", llama_mlp.down_proj, [mlp_hidden])
    graph.add_output(output)
    return graph


class PositionIDs(torch.nn.Module):
    """Generate position IDs for RoPE."""

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        position_ids = torch.arange(0, input_ids.shape[1], device=input_ids.device)
        return position_ids.unsqueeze(dim=0)


class Add(torch.nn.Module):
    """Residual connection: x + residual."""

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return x + residual


def to_graph_module(model: LlamaForCausalLM) -> GraphModule:
    """Create a fine-grained GraphModule representation of LlamaForCausalLM."""
    llama: LlamaModel = model.model

    graph = GraphModule()
    input_ids = graph.add_input("batch")

    # The Llama model first embeds the tokens.
    hidden = graph.add_node("embed_tokens", llama.embed_tokens, [input_ids])

    # It then generates Rotary Positional Embeddings, based on the input length.
    position_ids = graph.add_node("position_ids", PositionIDs(), [input_ids])
    position_embeddings = graph.add_node(
        "position_embeddings", llama.rotary_emb, [hidden, position_ids]
    )

    for i, layer in enumerate(llama.layers):
        assert isinstance(layer, LlamaDecoderLayer)
        # Each Decoder layer primarily consist of a norm -> attention -> norm -> mlp stream.

        residual = hidden
        hidden = graph.add_node(f"layer_{i}_input_layernorm", layer.input_layernorm, [hidden])

        # Note the tuple unpacking, when adding a subgraph we could expect several outputs.
        (attn_output,) = graph.add_subgraph(
            f"layer_{i}_attention",
            attention(layer.self_attn),
            [
                hidden,
            ],
            {"position_embeddings": position_embeddings},
        )

        hidden = graph.add_node(f"layer_{i}_attn_residual", Add(), [attn_output, residual])

        residual = hidden
        hidden = graph.add_node(
            f"layer_{i}_post_attention_layernorm",
            layer.post_attention_layernorm,
            [hidden],
        )

        (mlp_output,) = graph.add_subgraph(f"layer_{i}_mlp", mlp(layer.mlp), [hidden])

        hidden = graph.add_node(f"layer_{i}_mlp_residual", Add(), [mlp_output, residual])

    # After the decoder layers, we perform one final normalization before output projection to logits.
    hidden = graph.add_node("norm", llama.norm, [hidden])
    lm_logits = graph.add_node("lm_head", model.lm_head, [hidden])
    graph.add_output(lm_logits)

    return graph
