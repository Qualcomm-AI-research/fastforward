# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Tests for weight extraction from a quantized FastForward model."""

import logging

from typing import Callable, cast

import fastforward as ff
import pytest
import torch

from fastforward.export.stages.gguf._extract import extract_module_tensors
from fastforward.export.stages.gguf._packing import GGUF_Q4_0, GGUF_Q8_0
from fastforward.export.stages.gguf.adapter import ArchAdapter
from fastforward.nn.linear import QuantizedLinear

_PER_BLOCK = ff.granularity.PerBlock(block_dims=1, block_sizes=32, per_channel_dims=0)


class _Config:
    def __init__(self, tie_word_embeddings: bool = False) -> None:
        self.tie_word_embeddings = tie_word_embeddings
        self.num_attention_heads = 1
        self.num_key_value_heads = 1
        self.hidden_size = 0
        self.num_hidden_layers = 0
        self.intermediate_size = 0
        self.max_position_embeddings = 0
        self.rms_norm_eps = 1e-5
        self.vocab_size = 0


def _make_quantized_linear(out_features: int, in_features: int) -> QuantizedLinear:
    """Build a QuantizedLinear with an initialized 4-bit per-block weight quantizer."""
    linear = torch.nn.Linear(in_features, out_features, bias=False)
    linear.__class__ = QuantizedLinear
    quantized = cast(QuantizedLinear, linear)
    quantized.__init_quantization__()
    quantizers = ff.find_quantizers(quantized, "[quantizer:parameter/weight]")
    quantizers.initialize(ff.nn.LinearQuantizer, num_bits=4, granularity=_PER_BLOCK, symmetric=True)
    with ff.estimate_ranges(quantized.weight_quantizer, ff.range_setting.running_minmax):
        quantized.weight_quantizer(quantized.weight)
    return quantized


def _accept_weights(hf_name: str) -> str | None:
    """A name map that keeps weight tensors and drops quantizer parameters."""
    return hf_name if hf_name.endswith(".weight") and "quantizer" not in hf_name else None


def _make_adapter(name_map: Callable[[str], str | None] | None = None) -> ArchAdapter:
    """A minimal adapter suitable for extract-only tests."""
    if name_map is None:
        name_map = _accept_weights
    return ArchAdapter(
        gguf_arch="test",
        name_map=name_map,
        transforms=[],
        write_metadata=lambda writer, config: None,
        tokenizer_model="gpt2",
        tokenizer_pre="default",
    )


def test_extract_quantized_linear_reads_learned_codes(_seed_prngs: int) -> None:
    # GIVEN: a model with a single initialized QuantizedLinear.
    class Model(ff.nn.QuantizedModule):
        def __init__(self) -> None:
            super().__init__()
            self.weight = _make_quantized_linear(4, 32)

    model = Model()

    # WHEN: extracting exportable tensors.
    extracted = extract_module_tensors(
        model, adapter=_make_adapter(), config=_Config(), quant_format=GGUF_Q4_0
    )

    # THEN: the linear weight is extracted as quantized, with block-shaped codes
    # and one scale per block, tagged as quantized.
    (tensor,) = [t for t in extracted if t.hf_name == "weight.weight"]
    assert tensor.kind == "quantized"
    assert tensor.rows == 4 and tensor.cols == 32
    assert tensor.int_codes is not None and tensor.int_codes.shape == (4, 32)
    assert tensor.scales is not None


def test_extract_passes_through_unquantized_as_float(_seed_prngs: int) -> None:
    # GIVEN: a model with a plain embedding (no FastForward quantizer).
    class Model(ff.nn.QuantizedModule):
        def __init__(self) -> None:
            super().__init__()
            self.embed = torch.nn.Embedding(8, 32)

    model = Model()

    # WHEN: extracting.
    extracted = extract_module_tensors(
        model, adapter=_make_adapter(), config=_Config(), quant_format=GGUF_Q4_0
    )

    # THEN: the embedding passes through as float (no fallback quantization).
    (tensor,) = [t for t in extracted if t.hf_name == "embed.weight"]
    assert tensor.kind == "float"
    assert tensor.float_data is not None
    torch.testing.assert_close(tensor.float_data, model.embed.weight.detach().float())


def test_extract_passes_through_float_norms(_seed_prngs: int) -> None:
    # GIVEN: a model with a norm weight (1-D parameter, not a linear).
    class Model(ff.nn.QuantizedModule):
        def __init__(self) -> None:
            super().__init__()
            self.norm = torch.nn.LayerNorm(32)

    model = Model()

    # WHEN: extracting.
    extracted = extract_module_tensors(
        model, adapter=_make_adapter(), config=_Config(), quant_format=GGUF_Q4_0
    )

    # THEN: the norm weight passes through as float.
    (tensor,) = [t for t in extracted if t.hf_name == "norm.weight"]
    assert tensor.kind == "float"
    assert tensor.float_data is not None
    torch.testing.assert_close(tensor.float_data, model.norm.weight.detach().float())


def test_extract_skips_tied_lm_head(_seed_prngs: int) -> None:
    # GIVEN: a model whose `lm_head` QuantizedLinear yields the parameter name
    # `lm_head.weight`, with tied embeddings.
    class Model(ff.nn.QuantizedModule):
        def __init__(self) -> None:
            super().__init__()
            self.lm_head = _make_quantized_linear(8, 32)

    model = Model()
    adapter = _make_adapter(lambda hf_name: hf_name if hf_name.endswith(".weight") else None)

    # WHEN: extracting with tied embeddings.
    extracted = extract_module_tensors(
        model, adapter=adapter, config=_Config(tie_word_embeddings=True), quant_format=GGUF_Q4_0
    )

    # THEN: the tied lm_head weight is skipped entirely.
    assert all(t.hf_name != "lm_head.weight" for t in extracted)


def test_extract_keeps_lm_head_when_not_tied(_seed_prngs: int) -> None:
    # GIVEN: the same model, but embeddings are NOT tied.
    class Model(ff.nn.QuantizedModule):
        def __init__(self) -> None:
            super().__init__()
            self.lm_head = _make_quantized_linear(8, 32)

    model = Model()
    adapter = _make_adapter(lambda hf_name: hf_name if hf_name == "lm_head.weight" else None)

    # WHEN: extracting without tied embeddings.
    extracted = extract_module_tensors(
        model, adapter=adapter, config=_Config(), quant_format=GGUF_Q4_0
    )

    # THEN: the lm_head weight is retained and quantized.
    (tensor,) = [t for t in extracted if t.hf_name == "lm_head.weight"]
    assert tensor.kind == "quantized"


def test_extract_uses_adapter_is_tied(_seed_prngs: int) -> None:
    # GIVEN: a model whose output projection lives at a non-HF name, and an
    # adapter that identifies it via a custom is_tied predicate.
    class Model(ff.nn.QuantizedModule):
        def __init__(self) -> None:
            super().__init__()
            self.output_proj = _make_quantized_linear(8, 32)

    model = Model()
    adapter = ArchAdapter(
        gguf_arch="test",
        name_map=lambda hf_name: hf_name if hf_name.endswith(".weight") else None,
        transforms=[],
        write_metadata=lambda writer, config: None,
        tokenizer_model="gpt2",
        tokenizer_pre="default",
        is_tied=lambda hf_name, config: hf_name == "output_proj.weight",
    )

    # WHEN: extracting with a config (is_tied decides independently).
    extracted = extract_module_tensors(
        model, adapter=adapter, config=_Config(), quant_format=GGUF_Q4_0
    )

    # THEN: the adapter-identified tensor is skipped.
    assert all(t.hf_name != "output_proj.weight" for t in extracted)


def test_extract_logs_warning_for_skipped_params(
    _seed_prngs: int, caplog: pytest.LogCaptureFixture
) -> None:
    # GIVEN: a model with a parameter the adapter's name_map explicitly drops.
    class Model(ff.nn.QuantizedModule):
        def __init__(self) -> None:
            super().__init__()
            self.embed = torch.nn.Embedding(8, 32)
            self.extra = torch.nn.Parameter(torch.zeros(4))

    model = Model()

    def name_map(hf_name: str) -> str | None:
        if hf_name == "extra":
            return None
        return hf_name if hf_name.endswith(".weight") else None

    adapter = _make_adapter(name_map)

    # WHEN: extracting, with WARNING-level capture enabled.
    with caplog.at_level(logging.WARNING, logger="fastforward.export.stages.gguf._extract"):
        extract_module_tensors(model, adapter=adapter, config=_Config(), quant_format=GGUF_Q4_0)

    # THEN: a single WARNING record was emitted naming the dropped parameter.
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warning_records) == 1
    assert "extra" in warning_records[0].getMessage()


def test_extract_uninitialized_quantizer_passes_as_float(_seed_prngs: int) -> None:
    # GIVEN: a QuantizedLinear whose weight quantizer is NOT initialized.
    class Model(ff.nn.QuantizedModule):
        def __init__(self) -> None:
            super().__init__()
            linear = torch.nn.Linear(32, 4, bias=False)
            linear.__class__ = QuantizedLinear
            cast(QuantizedLinear, linear).__init_quantization__()
            # Quantizer exists but is uninitialized (no range estimation done).
            self.proj = linear

    model = Model()

    # WHEN: extracting.
    extracted = extract_module_tensors(
        model, adapter=_make_adapter(), config=_Config(), quant_format=GGUF_Q4_0
    )

    # THEN: the weight arrives as float because no initialized quantizer exists.
    (tensor,) = [t for t in extracted if t.hf_name == "proj.weight"]
    assert tensor.kind == "float"


def _make_quantized_linear_custom(
    out_features: int,
    in_features: int,
    *,
    num_bits: int = 4,
    symmetric: bool = True,
    granularity: ff.granularity.Granularity | None = None,
) -> QuantizedLinear:
    """Build a QuantizedLinear with configurable quantizer settings."""
    linear = torch.nn.Linear(in_features, out_features, bias=False)
    linear.__class__ = QuantizedLinear
    quantized = cast(QuantizedLinear, linear)
    quantized.__init_quantization__()
    gran = granularity if granularity is not None else _PER_BLOCK
    quantizers = ff.find_quantizers(quantized, "[quantizer:parameter/weight]")
    quantizers.initialize(
        ff.nn.LinearQuantizer, num_bits=num_bits, granularity=gran, symmetric=symmetric
    )
    with ff.estimate_ranges(quantized.weight_quantizer, ff.range_setting.running_minmax):
        quantized.weight_quantizer(quantized.weight)
    return quantized


def test_extract_rejects_asymmetric_quantizer(_seed_prngs: int) -> None:
    # GIVEN: a model with an asymmetric quantizer.
    class Model(ff.nn.QuantizedModule):
        def __init__(self) -> None:
            super().__init__()
            self.weight = _make_quantized_linear_custom(4, 32, symmetric=False)

    model = Model()

    # WHEN/THEN: extraction raises ExportError mentioning asymmetric.
    with pytest.raises(ff.exceptions.ExportError, match="asymmetric"):
        extract_module_tensors(
            model, adapter=_make_adapter(), config=_Config(), quant_format=GGUF_Q4_0
        )


def test_extract_rejects_mismatched_num_bits(_seed_prngs: int) -> None:
    # GIVEN: a model quantized at 4 bits, but exporting with an 8-bit format.
    class Model(ff.nn.QuantizedModule):
        def __init__(self) -> None:
            super().__init__()
            self.weight = _make_quantized_linear_custom(4, 32, num_bits=4)

    model = Model()

    # WHEN/THEN: extraction raises ExportError mentioning num_bits.
    with pytest.raises(ff.exceptions.ExportError, match="num_bits"):
        extract_module_tensors(
            model, adapter=_make_adapter(), config=_Config(), quant_format=GGUF_Q8_0
        )


def test_extract_rejects_mismatched_block_size(_seed_prngs: int) -> None:
    # GIVEN: a model quantized with block_size=64, but format expects 32.
    gran_64 = ff.granularity.PerBlock(block_dims=1, block_sizes=64, per_channel_dims=0)

    class Model(ff.nn.QuantizedModule):
        def __init__(self) -> None:
            super().__init__()
            self.weight = _make_quantized_linear_custom(4, 64, granularity=gran_64)

    model = Model()

    # WHEN/THEN: extraction raises ExportError mentioning block_size.
    with pytest.raises(ff.exceptions.ExportError, match="block_size"):
        extract_module_tensors(
            model, adapter=_make_adapter(), config=_Config(), quant_format=GGUF_Q4_0
        )


def test_extract_accepts_matching_8bit_format(_seed_prngs: int) -> None:
    # GIVEN: a model quantized at 8 bits, exporting with GGUF_Q8_0.
    class Model(ff.nn.QuantizedModule):
        def __init__(self) -> None:
            super().__init__()
            self.weight = _make_quantized_linear_custom(4, 32, num_bits=8)

    model = Model()

    # WHEN: extracting with matching format.
    extracted = extract_module_tensors(
        model, adapter=_make_adapter(), config=_Config(), quant_format=GGUF_Q8_0
    )

    # THEN: the tensor is extracted successfully.
    (tensor,) = [t for t in extracted if t.hf_name == "weight.weight"]
    assert tensor.kind == "quantized"
