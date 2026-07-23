# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Extensibility test: a user-defined ``ArchAdapter`` for a fictional model.

This is the marquee test proving that FastForward's GGUF pipeline is truly
model-agnostic. It builds a small ``QuantizedModule`` whose parameter names do
NOT match HuggingFace's Llama/Qwen convention (``net.blocks.<i>.attn.q.weight``
instead of ``model.layers.<i>.self_attn.q_proj.weight``), constructs a user
:class:`ArchAdapter` that maps this custom module tree to GGUF names, and runs
the pipeline. The test asserts the produced GGUF carries the user's chosen
architecture string (``"toy"``) and renamed tensors (``blk.0.attn_q.weight``).
"""

import pathlib
import re

from types import SimpleNamespace
from typing import Any, cast

import fastforward as ff
import gguf
import pytest
import torch

from fastforward.export.pipeline import ExportRequest, GgufLlamaCppOptions, export_with_pipeline
from fastforward.export.stages.gguf import GGUF_Q4_0, ArchAdapter
from fastforward.nn.linear import QuantizedLinear

_PER_BLOCK = ff.granularity.PerBlock(block_dims=1, block_sizes=32, per_channel_dims=0)


class _ToyAttn(torch.nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        # Note: user's naming (q/k/v/o), not HF's (q_proj/k_proj/v_proj/o_proj).
        self.q = torch.nn.Linear(hidden, hidden, bias=False)
        self.k = torch.nn.Linear(hidden, hidden, bias=False)
        self.v = torch.nn.Linear(hidden, hidden, bias=False)
        self.o = torch.nn.Linear(hidden, hidden, bias=False)


class _ToyBlock(torch.nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden)
        self.attn = _ToyAttn(hidden)


class _ToyNet(torch.nn.Module):
    def __init__(self, vocab: int, hidden: int, n_blocks: int) -> None:
        super().__init__()
        self.tok_emb = torch.nn.Embedding(vocab, hidden)
        self.blocks = torch.nn.ModuleList([_ToyBlock(hidden) for _ in range(n_blocks)])


class _ToyModel(ff.nn.QuantizedModule):
    """A quantized model whose parameter names do NOT match the HF convention."""

    def __init__(self, vocab: int, hidden: int, n_blocks: int) -> None:
        super().__init__()
        self.net = _ToyNet(vocab, hidden, n_blocks)
        self.head = torch.nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.net.tok_emb(input_ids)
        for block in self.net.blocks:
            block = cast(_ToyBlock, block)
            hidden = block.norm(hidden)
            hidden = block.attn.o(block.attn.v(hidden))
        return cast(torch.Tensor, self.head(hidden))


def _quantize_linears(model: torch.nn.Module) -> None:
    for module in model.modules():
        for _, child in list(module.named_children()):
            if type(child) is torch.nn.Linear:
                child.__class__ = QuantizedLinear
                cast(QuantizedLinear, child).__init_quantization__()
    targets = ff.mpath.query("**/blocks/**/[cls:ff.nn.QuantizedLinear]")
    weight_quantizers = ff.find_quantizers(model, targets / "[quantizer:parameter/weight]")
    weight_quantizers.initialize(
        ff.nn.LinearQuantizer, num_bits=4, granularity=_PER_BLOCK, symmetric=True
    )
    with ff.estimate_ranges(model, ff.range_setting.running_minmax), torch.no_grad():
        for module in model.modules():
            quantizer = getattr(module, "weight_quantizer", None)
            if isinstance(quantizer, ff.nn.LinearQuantizer):
                quantizer(module.weight)


def _toy_name_map(hf_name: str) -> str | None:
    """Map the toy model's parameter names to GGUF's blk.<i>.<slot>.weight scheme."""
    # Static tensors.
    if hf_name == "net.tok_emb.weight":
        return "token_embd.weight"
    if hf_name == "head.weight":
        return "output.weight"

    # net.blocks.<i>.norm.weight  ->  blk.<i>.attn_norm.weight
    match = re.match(r"net\.blocks\.(\d+)\.norm\.(weight|bias)$", hf_name)
    if match:
        idx, kind = match.group(1), match.group(2)
        return f"blk.{idx}.attn_norm.{kind}"

    # net.blocks.<i>.attn.<q|k|v|o>.weight  ->  blk.<i>.attn_<q|k|v|output>.weight
    match = re.match(r"net\.blocks\.(\d+)\.attn\.([qkvo])\.weight$", hf_name)
    if match:
        idx, slot = match.group(1), match.group(2)
        gguf_slot = {"q": "attn_q", "k": "attn_k", "v": "attn_v", "o": "attn_output"}[slot]
        return f"blk.{idx}.{gguf_slot}.weight"

    return None


def _toy_write_metadata(writer: Any, config: Any) -> None:
    """Minimal metadata for the toy arch.

    Enough for llama.cpp to reject it, but not enough to run inference — this
    test only asserts round-trip, not downstream usability.
    """
    writer.add_context_length(config.max_position_embeddings)
    writer.add_embedding_length(config.hidden_size)
    writer.add_block_count(config.num_hidden_layers)
    writer.add_head_count(config.num_attention_heads)
    writer.add_head_count_kv(config.num_key_value_heads)
    writer.add_vocab_size(config.vocab_size)


@pytest.mark.slow
def test_user_defined_arch_adapter_round_trips_custom_naming(
    tmp_path: pathlib.Path, _seed_prngs: int
) -> None:
    # GIVEN: a toy 2-block model with non-HF parameter names, quantized in-place.
    vocab, hidden, n_blocks = 32, 32, 2
    model = _ToyModel(vocab, hidden, n_blocks)
    _quantize_linears(model)
    model.eval()

    config = SimpleNamespace(
        max_position_embeddings=128,
        hidden_size=hidden,
        num_hidden_layers=n_blocks,
        intermediate_size=hidden,
        num_attention_heads=4,
        num_key_value_heads=4,
        rms_norm_eps=1e-5,
        vocab_size=vocab,
        tie_word_embeddings=False,
        rope_scaling=None,
    )

    # GIVEN: a user-defined adapter targeting the fictional "toy" architecture.
    adapter = ArchAdapter(
        gguf_arch="toy",
        name_map=_toy_name_map,
        transforms=[],
        write_metadata=_toy_write_metadata,
        tokenizer_model="gpt2",
        tokenizer_pre="default",
    )

    sample_inputs: list[tuple[tuple[Any, ...], dict[str, Any]]] = [
        ((torch.randint(0, vocab, (1, 8)),), {})
    ]
    options = GgufLlamaCppOptions(arch_adapter=adapter, quant_format=GGUF_Q4_0, model_config=config)

    # WHEN: exporting through the registered gguf/llama_cpp pipeline.
    artifacts = export_with_pipeline(
        ExportRequest(
            model=model,
            sample_inputs=sample_inputs,
            output_dir=tmp_path,
            model_name="toy-model",
            target="gguf",
            format="llama_cpp",
            options=options.to_context(),
        )
    )

    # THEN: the file was written.
    output_path = artifacts.stage_outputs["write_gguf"]
    assert output_path == tmp_path / "toy-model.gguf"
    assert output_path.exists()

    # THEN: the GGUF header carries the user's chosen architecture string.
    reader = gguf.GGUFReader(str(output_path))
    arch_field = reader.get_field("general.architecture")
    assert arch_field is not None
    arch_bytes = bytes(arch_field.parts[arch_field.data[-1]])
    assert arch_bytes == b"toy"

    # THEN: the user's custom names round-trip through the adapter into GGUF's
    # blk.<i>.<slot>.weight scheme — the marquee assertion.
    tensor_names = {tensor.name for tensor in reader.tensors}
    assert "blk.0.attn_q.weight" in tensor_names
    assert "blk.0.attn_k.weight" in tensor_names
    assert "blk.0.attn_v.weight" in tensor_names
    assert "blk.0.attn_output.weight" in tensor_names
    assert "blk.1.attn_q.weight" in tensor_names
    assert "token_embd.weight" in tensor_names
    assert "output.weight" in tensor_names

    # THEN: the toy adapter's Q4_0 tensors were packed as Q4_0 blocks.
    q_tensor = next(t for t in reader.tensors if t.name == "blk.0.attn_q.weight")
    assert q_tensor.tensor_type == gguf.GGMLQuantizationType.Q4_0
