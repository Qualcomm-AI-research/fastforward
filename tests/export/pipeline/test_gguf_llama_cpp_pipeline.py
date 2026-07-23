# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""End-to-end test for the FastForward -> GGUF (llama.cpp) export pipeline.

Builds a tiny 2-layer Llama-shaped model, quantizes its linears with
FastForward, runs the pipeline, and reads the produced GGUF back with the
``gguf`` reader.
"""

import pathlib

from types import SimpleNamespace
from typing import Any, cast

import fastforward as ff
import gguf
import pytest
import torch

from fastforward.export.pipeline import ExportRequest, GgufLlamaCppOptions, export_with_pipeline
from fastforward.export.stages.gguf import GGUF_Q4_0, LLAMA_ADAPTER
from fastforward.nn.linear import QuantizedLinear

_PER_BLOCK = ff.granularity.PerBlock(block_dims=1, block_sizes=32, per_channel_dims=0)


class _Attention(torch.nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.k_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.v_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.o_proj = torch.nn.Linear(hidden, hidden, bias=False)


class _Mlp(torch.nn.Module):
    def __init__(self, hidden: int, inter: int) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden, inter, bias=False)
        self.up_proj = torch.nn.Linear(hidden, inter, bias=False)
        self.down_proj = torch.nn.Linear(inter, hidden, bias=False)


class _Layer(torch.nn.Module):
    def __init__(self, hidden: int, inter: int) -> None:
        super().__init__()
        self.input_layernorm = torch.nn.LayerNorm(hidden)
        self.self_attn = _Attention(hidden)
        self.post_attention_layernorm = torch.nn.LayerNorm(hidden)
        self.mlp = _Mlp(hidden, inter)


class _Inner(torch.nn.Module):
    def __init__(self, vocab: int, hidden: int, inter: int, n_layers: int) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab, hidden)
        self.layers = torch.nn.ModuleList([_Layer(hidden, inter) for _ in range(n_layers)])
        self.norm = torch.nn.LayerNorm(hidden)


class _TinyLlama(ff.nn.QuantizedModule):
    def __init__(self, vocab: int, hidden: int, inter: int, n_layers: int) -> None:
        super().__init__()
        self.model = _Inner(vocab, hidden, inter, n_layers)
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            layer = cast(_Layer, layer)
            residual = hidden
            hidden = layer.input_layernorm(hidden)
            attn = layer.self_attn.o_proj(layer.self_attn.v_proj(hidden))
            hidden = residual + attn
            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp.down_proj(layer.mlp.up_proj(hidden))
            hidden = residual + hidden
        hidden = self.model.norm(hidden)
        return cast(torch.Tensor, self.lm_head(hidden))


def _quantize_linears(model: torch.nn.Module) -> None:
    for module in model.modules():
        for child_name, child in list(module.named_children()):
            if type(child) is torch.nn.Linear:
                child.__class__ = QuantizedLinear
                cast(QuantizedLinear, child).__init_quantization__()
    targets = ff.mpath.query("**/layers/**/[cls:ff.nn.QuantizedLinear]")
    weight_quantizers = ff.find_quantizers(model, targets / "[quantizer:parameter/weight]")
    weight_quantizers.initialize(
        ff.nn.LinearQuantizer, num_bits=4, granularity=_PER_BLOCK, symmetric=True
    )
    with ff.estimate_ranges(model, ff.range_setting.running_minmax), torch.no_grad():
        for module in model.modules():
            quantizer = getattr(module, "weight_quantizer", None)
            if isinstance(quantizer, ff.nn.LinearQuantizer):
                quantizer(module.weight)


@pytest.mark.slow
def test_gguf_llama_cpp_pipeline_writes_readable_file(
    tmp_path: pathlib.Path, _seed_prngs: int
) -> None:
    # GIVEN: a tiny 2-layer Llama-shaped model with FF-quantized linears.
    vocab, hidden, inter, n_layers = 64, 32, 64, 2
    model = _TinyLlama(vocab, hidden, inter, n_layers)
    _quantize_linears(model)
    model.eval()

    config = SimpleNamespace(
        max_position_embeddings=128,
        hidden_size=hidden,
        num_hidden_layers=n_layers,
        intermediate_size=inter,
        num_attention_heads=4,
        num_key_value_heads=4,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        vocab_size=vocab,
        tie_word_embeddings=False,
        rope_scaling=None,
    )
    sample_inputs: list[tuple[tuple[Any, ...], dict[str, Any]]] = [
        ((torch.randint(0, vocab, (1, 8)),), {})
    ]

    # WHEN: exporting through the registered gguf/llama_cpp pipeline.
    options = GgufLlamaCppOptions(
        arch_adapter=LLAMA_ADAPTER, quant_format=GGUF_Q4_0, model_config=config
    )
    artifacts = export_with_pipeline(
        ExportRequest(
            model=model,
            sample_inputs=sample_inputs,
            output_dir=tmp_path,
            model_name="tiny-llama",
            target="gguf",
            format="llama_cpp",
            options=options.to_context(),
        )
    )

    # THEN: the pipeline reports the written path and the file exists.
    output_path = artifacts.stage_outputs["write_gguf"]
    assert output_path == tmp_path / "tiny-llama.gguf"
    assert output_path.exists()

    # THEN: the GGUF reader can parse it, sees the llama architecture, and the
    # attention/MLP projections are stored as Q4_0 block tensors.
    reader = gguf.GGUFReader(str(output_path))
    arch_field = reader.get_field("general.architecture")
    assert arch_field is not None

    tensor_names = {tensor.name for tensor in reader.tensors}
    assert "blk.0.attn_q.weight" in tensor_names
    assert "blk.1.ffn_down.weight" in tensor_names
    assert "token_embd.weight" in tensor_names

    q_tensor = next(t for t in reader.tensors if t.name == "blk.0.attn_q.weight")
    assert q_tensor.tensor_type == gguf.GGMLQuantizationType.Q4_0


@pytest.mark.slow
def test_gguf_pipeline_skips_tied_lm_head(tmp_path: pathlib.Path, _seed_prngs: int) -> None:
    # GIVEN: a tiny Llama model with tie_word_embeddings=True.
    vocab, hidden, inter, n_layers = 64, 32, 64, 2
    model = _TinyLlama(vocab, hidden, inter, n_layers)
    _quantize_linears(model)
    model.eval()

    config = SimpleNamespace(
        max_position_embeddings=128,
        hidden_size=hidden,
        num_hidden_layers=n_layers,
        intermediate_size=inter,
        num_attention_heads=4,
        num_key_value_heads=4,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        vocab_size=vocab,
        tie_word_embeddings=True,
        rope_scaling=None,
    )
    sample_inputs: list[tuple[tuple[Any, ...], dict[str, Any]]] = [
        ((torch.randint(0, vocab, (1, 8)),), {})
    ]

    # WHEN: exporting with tied embeddings.
    options = GgufLlamaCppOptions(
        arch_adapter=LLAMA_ADAPTER, quant_format=GGUF_Q4_0, model_config=config
    )
    artifacts = export_with_pipeline(
        ExportRequest(
            model=model,
            sample_inputs=sample_inputs,
            output_dir=tmp_path,
            model_name="tiny-llama-tied",
            target="gguf",
            format="llama_cpp",
            options=options.to_context(),
        )
    )

    # THEN: the output GGUF does not contain "output.weight" (the lm_head mapping),
    # because the default is_tied predicate skips it when tie_word_embeddings=True.
    output_path = artifacts.stage_outputs["write_gguf"]
    reader = gguf.GGUFReader(str(output_path))
    tensor_names = {tensor.name for tensor in reader.tensors}
    assert "output.weight" not in tensor_names
    assert "token_embd.weight" in tensor_names
