# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Iterator

import fastforward as ff
import pytest
import torch

from fastforward.nn.quantizer import Quantizer
from fastforward.quantization.fuse import (
    ConventionDiscovery,
    WeightQuantizerTarget,
    fuse_qdq_weights,
)


def _quantized_linear_model(
    num_bits: int = 4, granularity: ff.granularity.Granularity | None = None
) -> ff.nn.QuantizedSequential:
    """Build a small quantized Sequential of Linear layers with weight quantizers."""
    granularity = ff.PerTensor() if granularity is None else granularity
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.Linear(4, 4),
    )
    ff.quantize_model(model)
    assert isinstance(model, ff.nn.QuantizedSequential)

    ff.find_quantizers(model, "**/[quantizer:parameter/weight]").initialize(
        ff.nn.LinearQuantizer, num_bits=num_bits, granularity=granularity
    )
    with ff.estimate_ranges(model, ff.range_setting.smoothed_minmax), ff.strict_quantization(False):
        model(torch.randn(4, 4))
    return model


def _layers(model: ff.nn.QuantizedSequential) -> list[ff.nn.QuantizedLinear]:
    layers: list[ff.nn.QuantizedLinear] = []
    for layer in model:
        assert isinstance(layer, ff.nn.QuantizedLinear)
        layers.append(layer)
    return layers


def _expected_qdq(module: ff.nn.QuantizedLinear) -> torch.Tensor:
    with ff.strict_quantization(False):
        return module.weight_quantizer(module.weight).dequantize()


def test_fuse_qdq_weights_snaps_weights_to_grid() -> None:
    """fuse_qdq_weights replaces each weight with its QDQ (grid-snapped) value."""
    # GIVEN A quantized model with initialized weight quantizers
    model = _quantized_linear_model()
    layers = _layers(model)
    expected = [_expected_qdq(layer) for layer in layers]

    # WHEN Fusing QDQ weights (quantizers left active)
    fuse_qdq_weights(model, stub_quantizers=False)

    # THEN Each weight is bit-exact with the manual quantize->dequantize of the original
    for layer, expected_weight in zip(layers, expected):
        assert torch.equal(layer.weight, expected_weight)


def test_fuse_qdq_weights_keeps_quantizers_active_by_default() -> None:
    """Without stub_quantizers, weight quantizers stay active (non-stub)."""
    # GIVEN A quantized model with initialized weight quantizers
    model = _quantized_linear_model()

    # WHEN Fusing QDQ weights with quantizers kept active
    fuse_qdq_weights(model, stub_quantizers=False)

    # THEN Weight quantizers remain active LinearQuantizers, not stubs
    for layer in _layers(model):
        assert isinstance(layer.weight_quantizer, ff.nn.LinearQuantizer)
        assert not layer.weight_quantizer.is_stub()


def test_fuse_qdq_weights_stubs_quantizers_when_requested() -> None:
    """With stub_quantizers=True, weight quantizers become QuantizerStubs."""
    # GIVEN A quantized model with initialized weight quantizers
    model = _quantized_linear_model()
    layers = _layers(model)
    original_metadata = [layer.weight_quantizer.quant_metadata for layer in layers]

    # WHEN Fusing QDQ weights and stubbing the weight quantizers
    fuse_qdq_weights(model, stub_quantizers=True)

    # THEN Each weight quantizer is a stub that preserves the original metadata
    for layer, metadata in zip(layers, original_metadata):
        assert isinstance(layer.weight_quantizer, ff.nn.QuantizerStub)
        assert layer.weight_quantizer.quant_metadata == metadata


def test_fuse_qdq_weights_active_fuse_is_idempotent_on_forward() -> None:
    """Re-quantizing an already grid-snapped weight is a no-op (active mode)."""
    # GIVEN A quantized model fused with quantizers left active
    model = _quantized_linear_model()
    fuse_qdq_weights(model, stub_quantizers=False)
    layers = _layers(model)
    fused_weights = [layer.weight.clone() for layer in layers]

    # WHEN Re-running the quantizer over the (already-snapped) weights
    # THEN The grid-snapped values are bit-exact (idempotent for affine quantizers)
    for layer, fused in zip(layers, fused_weights):
        with ff.strict_quantization(False):
            requantized = layer.weight_quantizer(layer.weight).dequantize()
        assert torch.equal(requantized, fused)


@pytest.mark.parametrize("num_bits", [4, 8])
@pytest.mark.parametrize(
    "granularity_factory",
    [ff.PerTensor, ff.PerChannel],
)
def test_fuse_qdq_weights_across_bitwidths_and_granularities(
    num_bits: int, granularity_factory: type
) -> None:
    """Fusing works for W4/W8 and per-tensor/per-channel weight quantizers."""
    # GIVEN A quantized model at the given bit-width / granularity
    model = _quantized_linear_model(num_bits=num_bits, granularity=granularity_factory())
    layers = _layers(model)
    expected = [_expected_qdq(layer) for layer in layers]

    # WHEN Fusing QDQ weights
    fuse_qdq_weights(model, stub_quantizers=True)

    # THEN Weights are grid-snapped and quantizers stubbed
    for layer, expected_weight in zip(layers, expected):
        torch.testing.assert_close(layer.weight, expected_weight)
        assert isinstance(layer.weight_quantizer, ff.nn.QuantizerStub)


def test_fuse_qdq_weights_leaves_activation_quantizers_untouched() -> None:
    """Only weight quantizers are fused; activation quantizers stay active."""
    # GIVEN A quantized model with both weight and activation quantizers
    model = _quantized_linear_model()
    ff.find_quantizers(model, "**/[quantizer:activation]").initialize(
        ff.nn.LinearQuantizer, num_bits=8, granularity=ff.PerTensor()
    )
    with ff.estimate_ranges(model, ff.range_setting.smoothed_minmax), ff.strict_quantization(False):
        model(torch.randn(4, 4))

    # WHEN Fusing QDQ weights and stubbing weight quantizers
    fuse_qdq_weights(model, stub_quantizers=True)

    # THEN Weight quantizers are stubs, activation quantizers remain active
    for layer in _layers(model):
        assert isinstance(layer.weight_quantizer, ff.nn.QuantizerStub)
        assert isinstance(layer.input_quantizer, ff.nn.LinearQuantizer)
        assert not layer.input_quantizer.is_stub()


def test_fuse_qdq_weights_skips_stub_weight_quantizers() -> None:
    """Weights whose quantizer is a stub are left unchanged."""
    # GIVEN A model whose weight quantizers were never initialized (still stubs)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    ff.quantize_model(model)
    assert isinstance(model, ff.nn.QuantizedSequential)
    layers = _layers(model)
    original_weights = [layer.weight.clone() for layer in layers]

    # WHEN Fusing QDQ weights
    fuse_qdq_weights(model, stub_quantizers=True)

    # THEN Weights are unchanged (stubs are not discovered)
    for layer, original in zip(layers, original_weights):
        assert torch.equal(layer.weight, original)


def test_fuse_qdq_weights_disabled_quantization_is_noop() -> None:
    """Under disabled quantization the fuse is a no-op and quantizers are kept."""
    # GIVEN A quantized, initialized model
    model = _quantized_linear_model()
    layers = _layers(model)
    original_weights = [layer.weight.clone() for layer in layers]

    # WHEN Fusing while quantization is globally disabled
    with ff.disable_quantization(model):
        fuse_qdq_weights(model, stub_quantizers=True)

    # THEN Weights are unchanged and weight quantizers are not stubbed
    for layer, original in zip(layers, original_weights):
        assert torch.equal(layer.weight, original)
        assert not isinstance(layer.weight_quantizer, ff.nn.QuantizerStub)


def test_fuse_qdq_weights_accepts_custom_discovery_strategy() -> None:
    """A custom WeightQuantizerDiscovery is honored by fuse_qdq_weights."""
    # GIVEN A quantized model and a discovery that yields only the first layer
    model = _quantized_linear_model()
    layers = _layers(model)
    expected_first = _expected_qdq(layers[0])
    original_second = layers[1].weight.clone()

    def only_first(model: torch.nn.Module) -> Iterator[WeightQuantizerTarget]:
        del model  # Discovery ignores the model; it targets a captured layer.
        first = layers[0]
        quantizer = first.weight_quantizer
        assert isinstance(quantizer, Quantizer)
        yield first, "weight", quantizer

    # WHEN Fusing with the custom discovery strategy
    fuse_qdq_weights(model, stub_quantizers=True, discovery=only_first)

    # THEN Only the discovered (first) layer is fused/stubbed
    torch.testing.assert_close(layers[0].weight, expected_first)
    assert isinstance(layers[0].weight_quantizer, ff.nn.QuantizerStub)
    torch.testing.assert_close(layers[1].weight, original_second)
    assert not isinstance(layers[1].weight_quantizer, ff.nn.QuantizerStub)


def test_convention_discovery_finds_weight_quantizers() -> None:
    """ConventionDiscovery yields (module, 'weight', quantizer) for each layer."""
    # GIVEN A quantized model with initialized weight quantizers
    model = _quantized_linear_model()
    layers = _layers(model)

    # WHEN Running convention-based discovery
    targets = list(ConventionDiscovery()(model))

    # THEN One target per Linear layer, each pointing at its weight quantizer
    assert len(targets) == 2
    for module, weight_attr, quantizer in targets:
        assert weight_attr == "weight"
        matching = [layer for layer in layers if layer.weight is module.weight]
        assert len(matching) == 1
        assert quantizer is matching[0].weight_quantizer


def _tied_weight_model(bits_first: int = 4, bits_second: int = 4) -> ff.nn.QuantizedSequential:
    """Build a Sequential whose two layers share one weight `Parameter`.

    Mirrors the tied `lm_head.weight`/`embed_tokens.weight` pattern common in
    language models. Each layer keeps its own weight quantizer, so the two can be
    configured to disagree on the grid.
    """
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 8),
        torch.nn.Linear(8, 8),
    )
    model[1].weight = model[0].weight
    ff.quantize_model(model)
    assert isinstance(model, ff.nn.QuantizedSequential)

    for index, bits in enumerate((bits_first, bits_second)):
        ff.find_quantizers(model, f"{index}/[quantizer:parameter/weight]").initialize(
            ff.nn.LinearQuantizer, num_bits=bits, granularity=ff.PerTensor()
        )
    with ff.estimate_ranges(model, ff.range_setting.smoothed_minmax), ff.strict_quantization(False):
        model(torch.randn(4, 8))
    return model


def test_fuse_qdq_weights_rejects_tied_weights_with_conflicting_quantizers() -> None:
    """A tied weight whose quantizers disagree on the grid must not be fused.

    Fusing writes into the shared weight once per target, so the last write would
    win and leave the weight wrong for the other layer.
    """
    # GIVEN Two layers sharing one weight Parameter but quantized differently
    model = _tied_weight_model(bits_first=8, bits_second=2)
    layers = _layers(model)
    assert layers[1].weight is layers[0].weight
    original = layers[0].weight.clone()

    # WHEN Fusing QDQ weights
    # THEN The conflict is reported instead of silently corrupting the weight
    with pytest.raises(ff.exceptions.QuantizationError) as exc_info:
        fuse_qdq_weights(model)
    assert "tied" in str(exc_info.value)

    # THEN The weight is left untouched
    torch.testing.assert_close(layers[0].weight, original)


def test_fuse_qdq_weights_allows_tied_weights_with_agreeing_quantizers() -> None:
    """Tied weights whose quantizers snap to the same grid fuse normally."""
    # GIVEN Two layers sharing one weight Parameter, quantized identically
    model = _tied_weight_model(bits_first=4, bits_second=4)
    layers = _layers(model)
    expected = _expected_qdq(layers[0])

    # WHEN Fusing QDQ weights
    fuse_qdq_weights(model)

    # THEN The shared weight is snapped once and stays tied
    torch.testing.assert_close(layers[0].weight, expected)
    assert layers[1].weight is layers[0].weight


def test_fuse_qdq_weights_allows_tied_weights_sharing_one_quantizer() -> None:
    """A tied weight with a single shared quantizer instance fuses normally."""
    # GIVEN Two layers sharing both the weight Parameter and the quantizer
    model = _tied_weight_model(bits_first=8, bits_second=2)
    layers = _layers(model)
    layers[1].weight_quantizer = layers[0].weight_quantizer
    expected = _expected_qdq(layers[0])

    # WHEN Fusing QDQ weights
    fuse_qdq_weights(model)

    # THEN There is only one grid, so fusing is unambiguous
    torch.testing.assert_close(layers[0].weight, expected)


def test_find_weight_quantizers_reports_targets_without_fusing() -> None:
    """find_weight_quantizers exposes discovery without mutating the model."""
    # GIVEN A quantized model with initialized weight quantizers
    model = _quantized_linear_model()
    layers = _layers(model)
    original = [layer.weight.clone() for layer in layers]

    # WHEN Querying the weight-quantizer targets
    targets = ff.quantization.find_weight_quantizers(model)

    # THEN Every layer is reported and no weight was modified
    assert len(targets) == 2
    assert {quantizer for _, _, quantizer in targets} == {
        layer.weight_quantizer for layer in layers
    }
    for layer, previous in zip(layers, original):
        torch.testing.assert_close(layer.weight, previous)
