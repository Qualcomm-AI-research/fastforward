# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import fastforward as ff
import pytest
import torch

from fastforward.overrides import (
    DisableQuantizationOverride,
    disable_quantization,
    enable_quantization,
)


@ff.flags.context(ff.strict_quantization, False)
def test_DisableQuantizationOverride(_seed_prngs: int) -> None:
    # GIVEN a quantized and non-quantized model with the same weights
    module = torch.nn.Linear(10, 10, bias=False)
    quantized_module = ff.nn.QuantizedLinear(10, 10, bias=False)
    quantized_module.weight = module.weight

    input_data = torch.randn(12, 10)

    # Assert that, without quantizers, both modules define the same linear
    # layer
    torch.testing.assert_close(module(input_data), quantized_module(input_data))

    # GIVEN the quantizers are properly initialized
    ff.find_quantizers(quantized_module, "**").initialize(ff.nn.LinearQuantizer, num_bits=8)
    with ff.estimate_ranges(quantized_module, ff.range_setting.smoothed_minmax):
        quantized_module(input_data)

    # GIVEN an expected output for the quantized and non-quantized model
    expected = module(input_data)
    expected_quantized = quantized_module(input_data)

    # WHEN DisableQuantizationOverride is attached to each quantizer in the
    # quantized model
    quantization_override = DisableQuantizationOverride()
    quantization_override.attach_to(ff.find_quantizers(quantized_module, "**"))

    # THEN the output for quantized_module and module must be the same
    torch.testing.assert_close(quantized_module(input_data), expected)

    # WHEN quantizers on quantized_module are enabled
    quantization_override.enable_quantization()

    # THEN quantized_module must produce the same quantized output as expected
    torch.testing.assert_close(
        quantized_module(input_data).dequantize(),
        expected_quantized.dequantize(),
    )

    # WHEN quantizers on quantized_module are enabled
    quantization_override.disable_quantization()

    # THEN the output for quantized_module and module must be the same
    torch.testing.assert_close(quantized_module(input_data), expected)

    # WHEN quantizers on quantized_module are enabled using using the returned
    # context
    with quantization_override.enable_quantization():
        # THEN quantized_module must produce the same quantized output as expected
        torch.testing.assert_close(
            quantized_module(input_data).dequantize(),
            expected_quantized.dequantize(),
        )

    # WHEN quantizers on quantized_module are disabled because the context ended
    # THEN the output for quantized_module and module must be the same
    torch.testing.assert_close(quantized_module(input_data), expected)

    # WHEN the disable quantization override is removed from quantized_module
    quantization_override.detach()
    # THEN quantized_module must produce the same quantized output as expected
    torch.testing.assert_close(
        quantized_module(input_data).dequantize(),
        expected_quantized.dequantize(),
    )


def test_disable_quantization_override_context(_seed_prngs: int) -> None:
    # GIVEN a quantized and non-quantized model with the same weights
    module = torch.nn.Linear(10, 10, bias=False)
    quantized_module = ff.nn.QuantizedLinear(10, 10, bias=False)
    quantized_module.weight = module.weight

    input_data = torch.randn(12, 10)

    # Assert that, without quantizers, both modules define the same linear
    # layer
    with ff.strict_quantization(False):
        torch.testing.assert_close(module(input_data), quantized_module(input_data))

    # GIVEN the quantizers are properly initialized
    ff.find_quantizers(quantized_module, "**").initialize(ff.nn.LinearQuantizer, num_bits=8)
    with ff.estimate_ranges(quantized_module, ff.range_setting.smoothed_minmax):
        quantized_module(input_data)

    # GIVEN an expected output for the quantized and non-quantized model
    expected = module(input_data)
    expected_quantized = quantized_module(input_data)

    # WHEN quantization is disables using the disable_quantization context
    with disable_quantization(quantized_module):
        # THEN the output of the quantized and non-quantized module must be the same
        torch.testing.assert_close(quantized_module(input_data), expected)

        # WHEN quantized is enabled using the enable_quantization context
        with enable_quantization(quantized_module):
            # THEN quantized_module must produce the same quantized output as expected
            torch.testing.assert_close(
                quantized_module(input_data).dequantize(),
                expected_quantized.dequantize(),
            )
        # WHEN the enable_quantization context terminates
        # THEN the output of the quantized and non-quantized module must be the same
        torch.testing.assert_close(quantized_module(input_data), expected)

    # WHEN the disable_quantization context terminates
    # THEN quantized_module must produce the same quantized output as expected
    torch.testing.assert_close(
        quantized_module(input_data).dequantize(),
        expected_quantized.dequantize(),
    )


def test_disable_quantization_quantizer_attachment() -> None:
    # DisableQuantizationOverride.attach_to must only attach to quantizers that
    # are passed to `attach_to`.
    #
    # Setup quantized module
    module = ff.nn.QuantizedLinear(10, 10, bias=False)
    ff.find_quantizers(module, "**").initialize(ff.nn.LinearQuantizer, num_bits=8)

    # Utility to test if override is added to a quantizer
    def _has_override(quantizer: ff.nn.Quantizer, override: DisableQuantizationOverride) -> bool:
        overrides = quantizer._quantizer_overrides.values()
        return override in overrides

    # case: weight only
    quantization_override = DisableQuantizationOverride()
    quantization_override.attach_to(ff.find_quantizers(module, "**/[quantizer:parameter/weight]"))
    assert not _has_override(module.input_quantizer, quantization_override)
    assert not _has_override(module.output_quantizer, quantization_override)
    assert _has_override(module.weight_quantizer, quantization_override)
    quantization_override.detach()

    # case: activation only
    quantization_override.attach_to(ff.find_quantizers(module, "**/[quantizer:activation]"))
    assert _has_override(module.input_quantizer, quantization_override)
    assert _has_override(module.output_quantizer, quantization_override)
    assert not _has_override(module.weight_quantizer, quantization_override)
    quantization_override.detach()

    # case: output activation only
    quantization_override.attach_to(ff.find_quantizers(module, "**/[quantizer:activation/output]"))
    assert not _has_override(module.input_quantizer, quantization_override)
    assert _has_override(module.output_quantizer, quantization_override)
    assert not _has_override(module.weight_quantizer, quantization_override)
    quantization_override.detach()


def test_partial_model_enable_disable_quantization_context_managers(
    multi_output_model: "_MultiOutputModel", _seed_prngs: int
) -> None:
    # GIVEN a model with multiple outputs
    model = multi_output_model

    # GIVEN expected outputs for either a quantized or non-quantized model
    input = torch.randn(4, 10)
    with ff.strict_quantization(False):
        expected_quantized_left, expected_quantized_right = model(input)

    with disable_quantization(model):
        expected_nonquantized_left, expected_nonquantized_right = model(input)

    # WHEN the quantization is disabled at the root level but enabled for a
    # submodule (left)
    with disable_quantization(model):
        with enable_quantization(model.left):
            actual_left, actual_right = model(input)

            # THEN the left output must match the expected quantized output
            # and the right output the expected unquantized output
            torch.testing.assert_close(
                actual_left.dequantize(), expected_quantized_left.dequantize()
            )
            torch.testing.assert_close(actual_right, expected_nonquantized_right)

    # WHEN the quantization is disabled at the root level but enabled for a
    # submodule (right)
    with disable_quantization(model):
        with enable_quantization(model.right):
            actual_left, actual_right = model(input)

            # THEN the left output must match the expected unquantized output
            # and the right output the expected quantized output
            torch.testing.assert_close(actual_left, expected_nonquantized_left)
            torch.testing.assert_close(
                actual_right.dequantize(), expected_quantized_right.dequantize()
            )

        # WHEN the enable_quantization context exits
        actual_left, actual_right = model(input)

        # THEN both outputs must match the expected unquantized output
        torch.testing.assert_close(actual_left, expected_nonquantized_left)
        torch.testing.assert_close(actual_right, expected_nonquantized_right)

    # WHEN the disable_quantization context exits
    with ff.strict_quantization(False):
        actual_left, actual_right = model(input)

    # THEN both outputs must match the expected quantized output
    torch.testing.assert_close(actual_left.dequantize(), expected_quantized_left.dequantize())
    torch.testing.assert_close(actual_right.dequantize(), expected_quantized_right.dequantize())


class _MultiOutputModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = ff.nn.QuantizedLinear(10, 10)
        self.right = ff.nn.QuantizedLinear(10, 10)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.left(x), self.right(x)


@pytest.fixture
def multi_output_model() -> _MultiOutputModel:
    """Define, construct and initialize a model with two linear heads."""
    model = _MultiOutputModel()

    def _init_quant(_name: str, _current: torch.nn.Module) -> ff.nn.Quantizer:
        del _name, _current
        quantizer = ff.nn.LinearQuantizer(num_bits=8)
        quantizer.quantization_range = (-1.0, 1.0)
        return quantizer

    ff.find_quantizers(model, "**/{[qtag:parameter/weight], [qtag:activation/output]}").initialize(
        _init_quant
    )
    return model
