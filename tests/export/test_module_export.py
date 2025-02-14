# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import json
import pathlib

from typing import Any, TypeAlias

import pytest
import torch

import fastforward as ff

from fastforward.export.module_export import export_modules
from fastforward.nn.quantizer import Quantizer
from fastforward.quantization.granularity import Granularity
from fastforward.quantization.quant_init import QuantizerCollection

QuantizedModelFixture: TypeAlias = tuple[torch.nn.Module, QuantizerCollection, QuantizerCollection]


@pytest.fixture
def simple_model() -> QuantizedModelFixture:
    class FFNet(torch.nn.Module):
        def __init__(self) -> None:
            super(FFNet, self).__init__()
            self.fc1 = ff.nn.QuantizedLinear(10, 10)
            self.relu1 = ff.nn.QuantizedRelu()
            self.fc2 = ff.nn.QuantizedLinear(10, 10)
            self.relu2 = ff.nn.QuantizedRelu()
            self.fc3 = ff.nn.QuantizedLinear(10, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.fc3(x)

            return x

    quant_model = FFNet()

    activation_quantizers = ff.find_quantizers(quant_model, "**/[quantizer:activation/output]")
    activation_quantizers |= ff.find_quantizers(quant_model, "fc1/[quantizer:activation/input]")
    parameter_quantizers = ff.find_quantizers(quant_model, "**/[quantizer:parameter]")

    return quant_model, activation_quantizers, parameter_quantizers


def initialize_quantizers(
    quantizers: QuantizerCollection, quantizer: type[Quantizer], **quantizer_params: Any
) -> None:
    quantizers.initialize(quantizer, **quantizer_params)


def activate_quantizers(
    quant_model: torch.nn.Module,
    data: torch.Tensor,
    activation_quantizers: QuantizerCollection,
    parameter_quantizers: QuantizerCollection,
    param_granularity: Granularity = ff.PerTensor(),
) -> None:
    initialize_quantizers(activation_quantizers, ff.nn.LinearQuantizer, num_bits=8)
    initialize_quantizers(
        parameter_quantizers, ff.nn.LinearQuantizer, num_bits=8, granularity=param_granularity
    )

    with ff.estimate_ranges(quant_model, ff.range_setting.smoothed_minmax):
        quant_model(data)


@ff.flags.context(ff.strict_quantization, False)
def test_module_export(simple_model) -> None:
    def check_module_files(path: pathlib.Path, module_name: str) -> None:
        encodings_file = path / f"{module_name}.encodings"
        onnx_file = path / f"{module_name}.onnx"

        assert encodings_file.is_file() and encodings_file.stat().st_size > 0
        assert onnx_file.is_file() and onnx_file.stat().st_size > 0

    def check_input_encodings_have_been_added(path: pathlib.Path, module_name: str) -> None:
        encodings_file = path / f"{module_name}.encodings"

        with open(encodings_file) as fp:
            encodings_dictionary = json.load(fp)

        assert "input" in encodings_dictionary["activation_encodings"]

    # GIVEN: a model with quantizer quantizers and a collection of modules of interest
    # (in this case linear and relu)
    data = torch.randn(2, 32, 10)
    model, activation_quantizers, parameter_quantizers = simple_model

    activate_quantizers(model, data, activation_quantizers, parameter_quantizers)

    linear_modules = ff.mpath.search("**/[cls:torch.nn.Linear]", model)
    relu_modules = ff.mpath.search("**/[cls:torch.nn.ReLU]", model)

    modules = linear_modules | relu_modules
    paths = export_modules(model, data, modules, "test_linear_model")

    # THEN: the number of exported modules paths should match the number
    # of modules of interest.
    assert len(modules) == len(paths)

    # THEN: the individual modules should be have been exported (ie there
    # exist ONNX file and encodings files for each of them) and inputs
    # encodings should have been added to the modules that did not have
    # a set input quantizer.
    for module, path in zip(modules, paths):
        check_module_files(paths[path], module.full_name)

        if module.full_name != "fc1":
            check_input_encodings_have_been_added(paths[path], module.full_name)
