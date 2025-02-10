# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import fastforward as ff
import pytest
import torch


class _MockQuantizer1(ff.nn.Quantizer):
    pass


class _MockQuantizer2(ff.nn.Quantizer):
    pass


@pytest.fixture()
def model():
    return ff.quantize_model(
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


def test_quantization_quantizer_collection_initialize(model):
    ff.find_quantizers(model, "layer1/*/[quantizer:activation/output]").initialize(_MockQuantizer1)
    ff.find_quantizers(model, "layer1/**").initialize(_MockQuantizer2, overwrite_policy="skip")

    with pytest.raises(ff.exceptions.QuantizationError):
        ff.find_quantizers(model, "layer1/**").initialize(_MockQuantizer2)

    assert isinstance(model.layer1.linear.output_quantizer, _MockQuantizer1)
    assert isinstance(model.layer1.relu.output_quantizer, _MockQuantizer1)
    assert isinstance(model.layer1.linear.input_quantizer, _MockQuantizer2)
    assert isinstance(model.layer1.linear.weight_quantizer, _MockQuantizer2)
    assert isinstance(model.layer1.linear.bias_quantizer, _MockQuantizer2)
    assert isinstance(model.layer1.relu.input_quantizer, _MockQuantizer2)

    assert isinstance(model.layer2.conv.output_quantizer, ff.nn.QuantizerStub)
    assert isinstance(model.layer2.relu.output_quantizer, ff.nn.QuantizerStub)
    assert isinstance(model.layer2.conv.input_quantizer, ff.nn.QuantizerStub)
    assert isinstance(model.layer2.conv.weight_quantizer, ff.nn.QuantizerStub)
    assert isinstance(model.layer2.conv.bias_quantizer, ff.nn.QuantizerStub)
    assert isinstance(model.layer2.relu.input_quantizer, ff.nn.QuantizerStub)

    ff.find_quantizers(model, "layer1/**").initialize(_MockQuantizer2, overwrite_policy="overwrite")

    assert isinstance(model.layer1.linear.output_quantizer, _MockQuantizer2)
    assert isinstance(model.layer1.relu.output_quantizer, _MockQuantizer2)
    assert isinstance(model.layer1.linear.input_quantizer, _MockQuantizer2)
    assert isinstance(model.layer1.linear.weight_quantizer, _MockQuantizer2)
    assert isinstance(model.layer1.linear.bias_quantizer, _MockQuantizer2)
    assert isinstance(model.layer1.relu.input_quantizer, _MockQuantizer2)

    assert isinstance(model.layer2.conv.output_quantizer, ff.nn.QuantizerStub)
    assert isinstance(model.layer2.relu.output_quantizer, ff.nn.QuantizerStub)
    assert isinstance(model.layer2.conv.input_quantizer, ff.nn.QuantizerStub)
    assert isinstance(model.layer2.conv.weight_quantizer, ff.nn.QuantizerStub)
    assert isinstance(model.layer2.conv.bias_quantizer, ff.nn.QuantizerStub)
    assert isinstance(model.layer2.relu.input_quantizer, ff.nn.QuantizerStub)

    with pytest.raises(ff.exceptions.QuantizationError):
        ff.find_quantizers(model, "layer1/**").initialize(_MockQuantizer2, overwrite_policy="error")


def test_quantization_config_presedence(model):
    config = (
        ff.QuantizationConfig()
        .add_rule("layer1/**", _MockQuantizer2)
        .add_rule("layer1/*/[quantizer:activation/output]", _MockQuantizer1)
    )
    config.initialize(model)

    assert isinstance(model.layer1.linear.output_quantizer, _MockQuantizer1)
    assert isinstance(model.layer1.relu.output_quantizer, _MockQuantizer1)
    assert isinstance(model.layer1.linear.input_quantizer, _MockQuantizer2)
    assert isinstance(model.layer1.linear.weight_quantizer, _MockQuantizer2)
    assert isinstance(model.layer1.linear.bias_quantizer, _MockQuantizer2)
    assert isinstance(model.layer1.relu.input_quantizer, _MockQuantizer2)

    assert isinstance(model.layer2.conv.output_quantizer, ff.nn.QuantizerStub)
    assert isinstance(model.layer2.relu.output_quantizer, ff.nn.QuantizerStub)
    assert isinstance(model.layer2.conv.input_quantizer, ff.nn.QuantizerStub)
    assert isinstance(model.layer2.conv.weight_quantizer, ff.nn.QuantizerStub)
    assert isinstance(model.layer2.conv.bias_quantizer, ff.nn.QuantizerStub)
    assert isinstance(model.layer2.relu.input_quantizer, ff.nn.QuantizerStub)
