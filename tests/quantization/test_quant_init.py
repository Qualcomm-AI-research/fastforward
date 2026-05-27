# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from dataclasses import dataclass

import fastforward as ff
import pytest
import torch


class _MockQuantizer1(ff.nn.Quantizer):
    pass


class _MockQuantizer2(ff.nn.Quantizer):
    pass


@dataclass
class _Fixture:
    module: torch.nn.Module
    layer1_linear: ff.nn.QuantizedLinear
    layer1_relu: ff.nn.QuantizedRelu
    layer2_conv: ff.nn.QuantizedConv2d
    layer2_relu: ff.nn.QuantizedRelu


@pytest.fixture()
def model() -> _Fixture:
    quantized = ff.quantize_model(
        torch.nn.ModuleDict({
            "layer1": torch.nn.ModuleDict({
                "linear": torch.nn.Linear(10, 10),
                "relu": torch.nn.ReLU(),
            }),
            "layer2": torch.nn.ModuleDict({
                "conv": torch.nn.Conv2d(10, 10, 3),
                "relu": torch.nn.ReLU(),
            }),
        })
    )
    assert isinstance(quantized, torch.nn.ModuleDict)
    layer1 = quantized["layer1"]
    layer2 = quantized["layer2"]
    assert isinstance(layer1, torch.nn.ModuleDict)
    assert isinstance(layer2, torch.nn.ModuleDict)
    layer1_linear = layer1["linear"]
    layer1_relu = layer1["relu"]
    layer2_conv = layer2["conv"]
    layer2_relu = layer2["relu"]
    assert isinstance(layer1_linear, ff.nn.QuantizedLinear)
    assert isinstance(layer1_relu, ff.nn.QuantizedRelu)
    assert isinstance(layer2_conv, ff.nn.QuantizedConv2d)
    assert isinstance(layer2_relu, ff.nn.QuantizedRelu)
    return _Fixture(quantized, layer1_linear, layer1_relu, layer2_conv, layer2_relu)


def test_quantization_quantizer_collection_initialize(model: _Fixture) -> None:
    ff.find_quantizers(model.module, "layer1/*/[quantizer:activation/output]").initialize(
        _MockQuantizer1
    )
    ff.find_quantizers(model.module, "layer1/**").initialize(
        _MockQuantizer2, overwrite_policy="skip"
    )

    with pytest.raises(ff.exceptions.QuantizationError):
        ff.find_quantizers(model.module, "layer1/**").initialize(_MockQuantizer2)

    assert isinstance(model.layer1_linear.output_quantizer, _MockQuantizer1)
    assert isinstance(model.layer1_relu.output_quantizer, _MockQuantizer1)
    assert isinstance(model.layer1_linear.input_quantizer, _MockQuantizer2)
    assert isinstance(model.layer1_linear.weight_quantizer, _MockQuantizer2)
    assert isinstance(model.layer1_linear.bias_quantizer, _MockQuantizer2)
    assert isinstance(model.layer1_relu.input_quantizer, _MockQuantizer2)

    assert isinstance(model.layer2_conv.output_quantizer, ff.nn.QuantizerStub)
    assert isinstance(model.layer2_relu.output_quantizer, ff.nn.QuantizerStub)
    assert isinstance(model.layer2_conv.input_quantizer, ff.nn.QuantizerStub)
    assert isinstance(model.layer2_conv.weight_quantizer, ff.nn.QuantizerStub)
    assert isinstance(model.layer2_conv.bias_quantizer, ff.nn.QuantizerStub)
    assert isinstance(model.layer2_relu.input_quantizer, ff.nn.QuantizerStub)

    ff.find_quantizers(model.module, "layer1/**").initialize(
        _MockQuantizer2, overwrite_policy="overwrite"
    )

    assert isinstance(model.layer1_linear.output_quantizer, _MockQuantizer2)
    assert isinstance(model.layer1_relu.output_quantizer, _MockQuantizer2)
    assert isinstance(model.layer1_linear.input_quantizer, _MockQuantizer2)
    assert isinstance(model.layer1_linear.weight_quantizer, _MockQuantizer2)
    assert isinstance(model.layer1_linear.bias_quantizer, _MockQuantizer2)
    assert isinstance(model.layer1_relu.input_quantizer, _MockQuantizer2)

    assert isinstance(model.layer2_conv.output_quantizer, ff.nn.QuantizerStub)
    assert isinstance(model.layer2_relu.output_quantizer, ff.nn.QuantizerStub)
    assert isinstance(model.layer2_conv.input_quantizer, ff.nn.QuantizerStub)
    assert isinstance(model.layer2_conv.weight_quantizer, ff.nn.QuantizerStub)
    assert isinstance(model.layer2_conv.bias_quantizer, ff.nn.QuantizerStub)
    assert isinstance(model.layer2_relu.input_quantizer, ff.nn.QuantizerStub)

    with pytest.raises(ff.exceptions.QuantizationError):
        ff.find_quantizers(model.module, "layer1/**").initialize(
            _MockQuantizer2, overwrite_policy="error"
        )


def test_quantization_config_precedence(model: _Fixture) -> None:
    config = (
        ff
        .QuantizationConfig()
        .add_rule("layer1/**", _MockQuantizer2)
        .add_rule("layer1/*/[quantizer:activation/output]", _MockQuantizer1)
    )
    config.initialize(model.module)

    assert isinstance(model.layer1_linear.output_quantizer, _MockQuantizer1)
    assert isinstance(model.layer1_relu.output_quantizer, _MockQuantizer1)
    assert isinstance(model.layer1_linear.input_quantizer, _MockQuantizer2)
    assert isinstance(model.layer1_linear.weight_quantizer, _MockQuantizer2)
    assert isinstance(model.layer1_linear.bias_quantizer, _MockQuantizer2)
    assert isinstance(model.layer1_relu.input_quantizer, _MockQuantizer2)

    assert isinstance(model.layer2_conv.output_quantizer, ff.nn.QuantizerStub)
    assert isinstance(model.layer2_relu.output_quantizer, ff.nn.QuantizerStub)
    assert isinstance(model.layer2_conv.input_quantizer, ff.nn.QuantizerStub)
    assert isinstance(model.layer2_conv.weight_quantizer, ff.nn.QuantizerStub)
    assert isinstance(model.layer2_conv.bias_quantizer, ff.nn.QuantizerStub)
    assert isinstance(model.layer2_relu.input_quantizer, ff.nn.QuantizerStub)
