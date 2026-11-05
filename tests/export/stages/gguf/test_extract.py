# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Tests for weight extraction from a quantized FastForward model."""

import logging

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Callable, cast

import fastforward as ff
import pytest
import torch

from fastforward.export.stages.gguf._extract import extract_module_tensors
from fastforward.export.stages.gguf._packing import GGUF_Q4_0, GGUF_Q4_1, GGUF_Q8_0
from fastforward.export.stages.gguf.adapter import ArchAdapter
from fastforward.export.stages.gguf.rules import DROP_TIED_LM_HEAD, Rename
from fastforward.nn.linear import QuantizedLinear

_PER_BLOCK = ff.granularity.PerBlock(block_dims=1, block_sizes=32, per_channel_dims=0)


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
        tensor_rules=(Rename(name_map),),
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
    extracted = extract_module_tensors(model, adapter=_make_adapter(), quant_format=GGUF_Q4_0)

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
    extracted = extract_module_tensors(model, adapter=_make_adapter(), quant_format=GGUF_Q4_0)

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
    extracted = extract_module_tensors(model, adapter=_make_adapter(), quant_format=GGUF_Q4_0)

    # THEN: the norm weight passes through as float.
    (tensor,) = [t for t in extracted if t.hf_name == "norm.weight"]
    assert tensor.kind == "float"
    assert tensor.float_data is not None
    torch.testing.assert_close(tensor.float_data, model.norm.weight.detach().float())


def test_extract_keeps_source_float_dtype(_seed_prngs: int) -> None:
    # GIVEN: a bf16 model with an unquantized embedding.
    class Model(ff.nn.QuantizedModule):
        def __init__(self) -> None:
            super().__init__()
            self.embed = torch.nn.Embedding(8, 32)

    model = Model().to(torch.bfloat16)

    # WHEN: extracting.
    (tensor,) = extract_module_tensors(model, adapter=_make_adapter(), quant_format=GGUF_Q4_0)

    # THEN: the float tensor keeps its bf16 dtype; the write stage casts it later.
    assert tensor.float_data is not None
    assert tensor.float_data.dtype == torch.bfloat16
    torch.testing.assert_close(tensor.float_data, model.embed.weight.detach())


class _BufferModel(ff.nn.QuantizedModule):
    def __init__(self) -> None:
        super().__init__()
        self.block = torch.nn.Module()
        self.block.register_buffer("layer_scalar", torch.full((1,), 0.5))
        self.block.register_buffer("scalar", torch.tensor(2.0))
        self.block.register_buffer("inv_freq", torch.ones(4), persistent=False)


def _accept_all(hf_name: str) -> str | None:
    return hf_name


def test_extract_includes_claimed_persistent_buffers() -> None:
    # GIVEN: a model with a 1-element persistent buffer claimed by a rename rule.
    model = _BufferModel()
    adapter = _make_adapter(lambda name: name if name == "block.layer_scalar" else None)

    # WHEN: extracting.
    (tensor,) = extract_module_tensors(model, adapter=adapter, quant_format=GGUF_Q4_0)

    # THEN: the buffer is extracted as a float tensor.
    assert tensor.hf_name == "block.layer_scalar"
    assert tensor.kind == "float"
    assert tensor.rows == 1
    assert tensor.float_data is not None
    torch.testing.assert_close(tensor.float_data, torch.full((1,), 0.5))


def test_extract_writes_scalar_buffers_as_one_element_vectors() -> None:
    # GIVEN: a model with a 0-d persistent buffer claimed by a rename rule.
    model = _BufferModel()
    adapter = _make_adapter(lambda name: name if name == "block.scalar" else None)

    # WHEN: extracting.
    (tensor,) = extract_module_tensors(model, adapter=adapter, quant_format=GGUF_Q4_0)

    # THEN: the scalar becomes a 1-element vector.
    assert tensor.rows == 1
    assert tensor.float_data is not None
    torch.testing.assert_close(tensor.float_data, torch.tensor([2.0]))


def test_extract_skips_non_persistent_buffers() -> None:
    # GIVEN: a rule that claims every name, including a non-persistent buffer.
    model = _BufferModel()

    # WHEN: extracting.
    extracted = extract_module_tensors(
        model, adapter=_make_adapter(_accept_all), quant_format=GGUF_Q4_0
    )

    # THEN: only the persistent buffers are extracted.
    assert [t.hf_name for t in extracted] == ["block.layer_scalar", "block.scalar"]


def _shared_buffer_model() -> ff.nn.QuantizedModule:
    """Two submodules registering the same tensor as a persistent buffer."""
    model = ff.nn.QuantizedModule()
    shared = torch.full((2,), 3.0)
    model.first = torch.nn.Module()
    model.first.register_buffer("scalar", shared)
    model.second = torch.nn.Module()
    model.second.register_buffer("scalar", shared)
    return model


def _shared_submodule_model() -> ff.nn.QuantizedModule:
    """One submodule with a persistent buffer, reachable under two names."""
    model = ff.nn.QuantizedModule()
    model.first = torch.nn.Module()
    model.first.register_buffer("scalar", torch.full((2,), 3.0))
    model.second = model.first
    return model


@pytest.mark.parametrize(
    "make_model",
    [_shared_buffer_model, _shared_submodule_model],
    ids=["shared-buffer", "shared-submodule"],
)
def test_extract_includes_buffer_claimed_under_its_second_name(
    make_model: Callable[[], ff.nn.QuantizedModule],
) -> None:
    # GIVEN: a persistent buffer reachable under two names, only the second claimed.
    model = make_model()
    adapter = _make_adapter(lambda name: name if name == "second.scalar" else None)

    # WHEN: extracting.
    extracted = extract_module_tensors(model, adapter=adapter, quant_format=GGUF_Q4_0)

    # THEN: the buffer is extracted under the claimed name, not hidden by its first alias.
    assert [t.hf_name for t in extracted] == ["second.scalar"]
    assert extracted[0].float_data is not None
    torch.testing.assert_close(extracted[0].float_data, torch.full((2,), 3.0))


def test_extract_skips_unclaimed_buffers_silently(caplog: pytest.LogCaptureFixture) -> None:
    # GIVEN: a model whose buffers no tensor rule claims.
    model = _BufferModel()

    # WHEN: extracting, with WARNING-level capture enabled.
    with caplog.at_level(logging.WARNING, logger="fastforward.export.stages.gguf._extract"):
        extracted = extract_module_tensors(model, adapter=_make_adapter(), quant_format=GGUF_Q4_0)

    # THEN: nothing is extracted and no warning is emitted.
    assert extracted == []
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_extract_rejects_claimed_non_float_buffer() -> None:
    # GIVEN: a claimed integer buffer.
    class Model(ff.nn.QuantizedModule):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("counts", torch.arange(4))

    # WHEN/THEN: extraction refuses to pass it through as a float tensor.
    with pytest.raises(ff.exceptions.ExportError, match="non-float dtype"):
        extract_module_tensors(Model(), adapter=_make_adapter(_accept_all), quant_format=GGUF_Q4_0)


def test_extract_logs_warning_for_skipped_params(
    _seed_prngs: int, caplog: pytest.LogCaptureFixture
) -> None:
    # GIVEN: a model with a parameter no tensor rule claims.
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
        extract_module_tensors(model, adapter=adapter, quant_format=GGUF_Q4_0)

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
    extracted = extract_module_tensors(model, adapter=_make_adapter(), quant_format=GGUF_Q4_0)

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

    # WHEN/THEN: extraction raises ExportError (no format matches the asymmetric quantizer).
    with pytest.raises(ff.exceptions.ExportError, match="no registered GGUF format matches"):
        extract_module_tensors(model, adapter=_make_adapter(), quant_format=GGUF_Q4_0)


def test_extract_rejects_one_sided_symmetric_quantizer(_seed_prngs: int) -> None:
    # GIVEN: a symmetric quantizer that went one-sided on a non-negative weight.
    linear = torch.nn.Linear(32, 4, bias=False)
    with torch.no_grad():
        linear.weight.abs_()
    linear.__class__ = QuantizedLinear
    quantized = cast(QuantizedLinear, linear)
    quantized.__init_quantization__()
    quantizers = ff.find_quantizers(quantized, "[quantizer:parameter/weight]")
    quantizers.initialize(ff.nn.LinearQuantizer, num_bits=4, granularity=_PER_BLOCK, symmetric=True)
    with ff.estimate_ranges(quantized.weight_quantizer, ff.range_setting.running_minmax):
        quantized.weight_quantizer(quantized.weight)

    class Model(ff.nn.QuantizedModule):
        def __init__(self) -> None:
            super().__init__()
            self.weight = quantized

    # WHEN/THEN: extraction refuses instead of dropping the offset.
    with pytest.raises(ff.exceptions.ExportError, match="one-sided"):
        extract_module_tensors(Model(), adapter=_make_adapter(), quant_format=GGUF_Q4_0)


def test_extract_reads_rounded_asymmetric_offsets(_seed_prngs: int) -> None:
    # GIVEN: a model with an asymmetric 4-bit quantizer.
    class Model(ff.nn.QuantizedModule):
        def __init__(self) -> None:
            super().__init__()
            self.weight = _make_quantized_linear_custom(4, 32, symmetric=False)

    model = Model()

    # WHEN: extracting for Q4_1.
    (tensor,) = extract_module_tensors(model, adapter=_make_adapter(), quant_format=GGUF_Q4_1)

    # THEN: codes, scales, and offsets reproduce the quantizer's dequantized weight,
    # i.e. offsets are rounded like the quantizer rounds them.
    assert tensor.int_codes is not None and tensor.scales is not None
    assert tensor.offsets is not None
    codes = tensor.int_codes.reshape(4, 1, 32).float()
    dequantized = (codes + tensor.offsets[..., None]) * tensor.scales[..., None]
    expected = model.weight.weight_quantizer(model.weight.weight).dequantize()
    torch.testing.assert_close(dequantized.reshape(4, 32), expected)


def test_extract_rejects_mismatched_num_bits(_seed_prngs: int) -> None:
    # GIVEN: a model quantized at 4 bits, but exporting with an 8-bit format.
    class Model(ff.nn.QuantizedModule):
        def __init__(self) -> None:
            super().__init__()
            self.weight = _make_quantized_linear_custom(4, 32, num_bits=4)

    model = Model()

    # WHEN/THEN: extraction raises ExportError mentioning num_bits.
    with pytest.raises(ff.exceptions.ExportError, match="num_bits"):
        extract_module_tensors(model, adapter=_make_adapter(), quant_format=GGUF_Q8_0)


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
        extract_module_tensors(model, adapter=_make_adapter(), quant_format=GGUF_Q4_0)


def test_extract_skips_parameters_dropped_before_any_claim(_seed_prngs: int) -> None:
    # GIVEN: a tied-embedding config and an lm_head whose 4-bit quantizer does
    # not match the 8-bit export format.
    class Model(ff.nn.QuantizedModule):
        def __init__(self) -> None:
            super().__init__()
            self.lm_head = _make_quantized_linear_custom(4, 32, num_bits=4)

    adapter = replace(_make_adapter(), tensor_rules=(DROP_TIED_LM_HEAD, Rename(_accept_weights)))
    config: Any = SimpleNamespace(tie_word_embeddings=True)

    # WHEN: extracting with the config.
    extracted = extract_module_tensors(
        Model(), adapter=adapter, config=config, quant_format=GGUF_Q8_0
    )

    # THEN: the leading Drop skips lm_head before its quantizer is validated.
    assert extracted == []


def test_extract_accepts_matching_8bit_format(_seed_prngs: int) -> None:
    # GIVEN: a model quantized at 8 bits, exporting with GGUF_Q8_0.
    class Model(ff.nn.QuantizedModule):
        def __init__(self) -> None:
            super().__init__()
            self.weight = _make_quantized_linear_custom(4, 32, num_bits=8)

    model = Model()

    # WHEN: extracting with matching format.
    extracted = extract_module_tensors(model, adapter=_make_adapter(), quant_format=GGUF_Q8_0)

    # THEN: the tensor is extracted successfully.
    (tensor,) = [t for t in extracted if t.hf_name == "weight.weight"]
    assert tensor.kind == "quantized"
