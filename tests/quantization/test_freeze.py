# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import fastforward as ff
import torch

from fastforward.quantization.freeze import freeze_parameters


def test_freeze_parameters_quantizes_weights_in_place() -> None:
    """Test that freeze_parameters quantizes model weights in-place during forward pass."""
    # GIVEN: A quantized model with initialized weight quantizers
    model = _quantized_model()

    # Store original weights for comparison
    original_weights = [layer.weight.clone() for layer in model]

    # WHEN: freeze_parameters is used during a forward pass
    with freeze_parameters(model):
        model(torch.randn(2, 2))

    # THEN: The weights should be quantized (different from original)
    for i, layer in enumerate(model):
        assert not torch.equal(layer.weight, original_weights[i]), (
            f"Layer {i} weight was not quantized"
        )


def test_freeze_parameters_removes_quantizers_by_default() -> None:
    """Test that freeze_parameters replaces quantizers with stubs by default."""
    # GIVEN: A quantized model with weight quantizers
    model = _quantized_model()

    # Verify quantizers exist before freezing
    quantizers_before = list(ff.nn.quantized_module.named_quantizers(model[0], recurse=False))
    assert len(quantizers_before) > 0, "No quantizers found before freezing"

    # WHEN: freeze_parameters is used with default settings
    with freeze_parameters(model):
        model(torch.randn(2, 2))

    # THEN: Quantizers should be replaced with QuantizerStubs
    for name, quantizer in ff.nn.quantized_module.named_quantizers(model[0], recurse=False):
        assert isinstance(quantizer, ff.nn.QuantizerStub), (
            f"Quantizer {name} was not replaced with stub"
        )


def test_freeze_parameters_preserves_quantizers_when_disabled() -> None:
    """Test that freeze_parameters preserves quantizers when remove_quantizers=False."""
    # GIVEN: A quantized model with weight quantizers
    model = _quantized_model()

    # Store original quantizer types
    original_quantizer_types = {
        name: type(quantizer)
        for name, quantizer in ff.nn.quantized_module.named_quantizers(model[0], recurse=False)
    }

    # WHEN: freeze_parameters is used with remove_quantizers=False
    with freeze_parameters(model, remove_quantizers=False):
        model(torch.randn(2, 2))

    # THEN: Original quantizers should be preserved
    for name, quantizer in ff.nn.quantized_module.named_quantizers(model[0], recurse=False):
        assert type(quantizer) is original_quantizer_types[name], (
            f"Quantizer {name} type changed unexpectedly"
        )
        assert not isinstance(quantizer, ff.nn.QuantizerStub), (
            f"Quantizer {name} was replaced with stub"
        )


def test_freeze_parameters_with_empty_model() -> None:
    """Test that freeze_parameters handles models without quantizers gracefully."""
    # GIVEN: A regular (non-quantized) model
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    # Note: Not calling ff.quantize_model, so no quantizers present

    # WHEN: freeze_parameters is applied to a model without quantizers
    # THEN: The operation should complete without errors
    with freeze_parameters(model):
        model(torch.randn(2, 2))


def test_freeze_parameters_preserves_quantizer_metadata() -> None:
    """Test that QuantizerStubs preserve original quantizer metadata."""
    # GIVEN: A quantized model with initialized quantizers
    model = _quantized_model()

    # Store original metadata
    original_metadata = {}
    for name, quantizer in ff.nn.quantized_module.named_quantizers(model[0], recurse=False):
        original_metadata[name] = quantizer.quant_metadata

    # WHEN: freeze_parameters replaces quantizers with stubs
    with freeze_parameters(model):
        model(torch.randn(2, 2))

    # THEN: QuantizerStubs should preserve the original metadata
    for name, quantizer in ff.nn.quantized_module.named_quantizers(model[0], recurse=False):
        assert isinstance(quantizer, ff.nn.QuantizerStub), f"Quantizer {name} should be a stub"
        assert quantizer._metadata == original_metadata[name], (
            f"Metadata not preserved for quantizer {name}"
        )


def test_freeze_parameters_with_disabled_quantizer() -> None:
    """Test that disabled quantizers are not removed and parameters are not updated."""
    # GIVEN A quantized model with initialized weight quantizers
    model = _quantized_model()

    # Store original weights and quantizer types
    original_weights = [layer.weight.clone() for layer in model]
    original_quantizer_types = [type(layer.weight_quantizer) for layer in model]

    # Ensure that initial quantizers are not stubs.
    assert [quant_type == ff.nn.QuantizerStub for quant_type in original_quantizer_types]

    # WHEN freeze_parameters is used with quantization disabled
    with ff.disable_quantization(model):
        with freeze_parameters(model):
            model(torch.randn(2, 2))

    # THEN The weights should NOT be quantized (should remain unchanged)
    for i, layer in enumerate(model):
        assert torch.equal(layer.weight, original_weights[i])

    # THEN The quantizers should NOT be replaced with stubs (should remain original type)
    assert original_quantizer_types == [type(layer.weight_quantizer) for layer in model]


def _quantized_model() -> ff.nn.QuantizedSequential:
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2),
        torch.nn.Linear(2, 2),
    )
    ff.quantize_model(model)
    assert isinstance(model, ff.nn.QuantizedSequential)

    # Initialize weight quantizers with 2-bit precision
    ff.find_quantizers(model, "**/[quantizer:parameter/weight]").initialize(
        ff.nn.LinearQuantizer, num_bits=2, granularity=ff.PerTensor()
    )

    # Estimate quantization ranges
    with ff.estimate_ranges(model, ff.range_setting.smoothed_minmax), ff.strict_quantization(False):
        model(torch.randn(4, 2, 2))
    return model
