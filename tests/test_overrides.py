# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import fastforward as ff
import torch

from fastforward.overrides import DisableQuantizationOverride, disable_quantization


@ff.flags.context(ff.strict_quantization, False)
def test_DisableQuantizationOverride(_seed_prngs: int) -> None:
    # Setup of non quantized and quantized module using the same weight
    module = torch.nn.Linear(10, 10, bias=False)
    quantized_module = ff.nn.QuantizedLinear(10, 10, bias=False)
    quantized_module.weight = module.weight

    input_data = torch.randn(12, 10)

    # Assert that, without quantizers, both modules define the same linear
    # layer
    torch.testing.assert_close(module(input_data), quantized_module(input_data))

    # Initialize quantizers and set ranges on quantized_module
    ff.find_quantizers(quantized_module, "**").initialize(ff.nn.LinearQuantizer, num_bits=8)
    with ff.estimate_ranges(quantized_module, ff.range_setting.smoothed_minmax):
        quantized_module(input_data)

    # Obtain expected output for both quantized and non-quantized forward pass
    expected = module(input_data)
    expected_quantized = quantized_module(input_data)

    # Create override and attach to all quantizers in quantized_module
    quantization_override = DisableQuantizationOverride()
    quantization_override.attach_to(ff.find_quantizers(quantized_module, "**"))

    # Test if quantized_module and module are the same when quantizers
    # are disabled.
    torch.testing.assert_close(quantized_module(input_data), expected)

    # Test if quantized_module produces the same quantized output as before
    # when quantizers are enabled again.
    quantization_override.enable_quantization()
    torch.testing.assert_close(
        quantized_module(input_data).dequantize(),
        expected_quantized.dequantize(),
    )

    # Test if quantized_module and module are the same when quantizers
    # are disabled again.
    quantization_override.disable_quantization()
    torch.testing.assert_close(quantized_module(input_data), expected)

    # Test if quantized_module producess the same quantized output as expected
    # after override has been removed.
    quantization_override.detach()
    torch.testing.assert_close(
        quantized_module(input_data).dequantize(),
        expected_quantized.dequantize(),
    )


def test_disable_quantization_override_context(_seed_prngs: int) -> None:
    # Setup of non quantized and quantized module using the same weight
    module = torch.nn.Linear(10, 10, bias=False)
    quantized_module = ff.nn.QuantizedLinear(10, 10, bias=False)
    quantized_module.weight = module.weight

    input_data = torch.randn(12, 10)

    # Assert that, without quantizers, both modules define the same linear
    # layer
    with ff.strict_quantization(False):
        torch.testing.assert_close(module(input_data), quantized_module(input_data))

    # Initialize quantizers and set ranges on quantized_module
    ff.find_quantizers(quantized_module, "**").initialize(ff.nn.LinearQuantizer, num_bits=8)
    with ff.estimate_ranges(quantized_module, ff.range_setting.smoothed_minmax):
        quantized_module(input_data)

    # Obtain expected output for both quantized and non-quantized forward pass
    expected = module(input_data)
    expected_quantized = quantized_module(input_data)

    # Test if quantized_module and module are the same when quantizers
    # are disabled.
    with disable_quantization(quantized_module):
        torch.testing.assert_close(quantized_module(input_data), expected)

    # Test if quantized_module producess the same quantized output as expected
    # after disable_quantization context ends.
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
