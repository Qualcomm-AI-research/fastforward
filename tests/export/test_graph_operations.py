# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import json
import pathlib

from typing import TypeAlias

import fastforward as ff
import pytest
import torch

from fastforward.export.export import export
from fastforward.quantization.granularity import Granularity
from fastforward.quantization.quant_init import QuantizerCollection

QuantizedModelFixture: TypeAlias = tuple[torch.nn.Module, QuantizerCollection, QuantizerCollection]


@pytest.fixture
def simple_model() -> QuantizedModelFixture:
    class FFNet(torch.nn.Module):
        """Simple FF model with quantized linear/relu modules."""

        def __init__(self) -> None:
            super().__init__()
            net_in_out_dim = 10
            self.fc1 = ff.nn.QuantizedLinear(net_in_out_dim, net_in_out_dim, bias=False)
            self.relu1 = ff.nn.QuantizedRelu()
            self.fc2 = ff.nn.QuantizedLinear(net_in_out_dim, net_in_out_dim, bias=False)
            self.relu2 = ff.nn.QuantizedRelu()
            self.fc3 = ff.nn.QuantizedLinear(net_in_out_dim, net_in_out_dim, bias=False)

            self.extra_weight = torch.nn.Parameter(
                torch.rand(size=(net_in_out_dim, net_in_out_dim))
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.fc1(x)
            x = torch.reshape(x, (x.shape[1], x.shape[0]))
            x = torch.reshape(x, (x.shape[1], x.shape[0]))
            x = self.relu1(x)
            x = self.fc2(x)
            x = torch.nn.functional.softmax(x)
            x = self.relu2(x)
            x = self.fc3(x)
            x = torch.matmul(x, self.extra_weight)

            return x

    quant_model = FFNet()

    activation_quantizers = ff.find_quantizers(quant_model, "**/[quantizer:activation/output]")
    activation_quantizers |= ff.find_quantizers(quant_model, "fc1/[quantizer:activation/input]")
    parameter_quantizers = ff.find_quantizers(quant_model, "**/[quantizer:parameter]")

    return quant_model, activation_quantizers, parameter_quantizers


def activate_quantizers(
    quant_model: torch.nn.Module,
    data: torch.Tensor,
    activation_quantizers: QuantizerCollection,
    parameter_quantizers: QuantizerCollection,
    param_granularity: Granularity = ff.PerTensor(),
) -> None:
    activation_quantizers.initialize(ff.nn.LinearQuantizer, num_bits=8)
    parameter_quantizers.initialize(
        ff.nn.LinearQuantizer, num_bits=8, granularity=param_granularity
    )

    with ff.estimate_ranges(quant_model, ff.range_setting.smoothed_minmax):
        quant_model(data)


@pytest.mark.xfail_due_to_too_new_torch
@pytest.mark.slow
@ff.flags.context(ff.strict_quantization, False)
def test_encodings_propagation(
    tmp_path: pathlib.Path,
    simple_model: QuantizedModelFixture,
    _seed_prngs: int,
) -> None:
    # GIVEN a quantized model.
    granularity = ff.PerTensor()
    data = torch.randn(32, 10)
    quant_model, activation_quantizers, parameter_quantizers = simple_model
    output_directory = tmp_path
    model_name = "test_export_function"

    output_model_directory = pathlib.Path(output_directory) / model_name

    activate_quantizers(quant_model, data, activation_quantizers, parameter_quantizers, granularity)

    # GIVEN the exported artifacts from that model and its original encodings file.
    export(quant_model, (data,), output_directory, model_name, propagate_encodings=False)
    encodings_file_path = (output_model_directory / model_name).with_suffix(".encodings")

    with open(encodings_file_path, "r") as file:
        encodings_dictionary = json.load(file)

    original_param_names = encodings_dictionary["param_encodings"].keys()
    original_activation_names = encodings_dictionary["activation_encodings"].keys()

    # WHEN exporting the same model with encoding propagation.
    export(quant_model, (data,), output_directory, model_name, propagate_encodings=True)
    encodings_file_path = (output_model_directory / model_name).with_suffix(".encodings")

    with open(encodings_file_path, "r") as file:
        encodings_dictionary = json.load(file)

    new_param_names = encodings_dictionary["param_encodings"].keys()
    new_activation_names = encodings_dictionary["activation_encodings"].keys()

    # THEN the parameters should be the same, as there is no encoding propagation to new parameters.
    assert new_param_names == original_param_names

    # THEN all the original activation should be preserved.
    assert original_activation_names <= new_activation_names

    # THEN the encodings for new activations should be associated as displayed in the below dictionar.
    # Case 1: these are derived from other activations
    activation_to_activation_association = {"view": "mm", "view_1": "mm"}

    for new_activation, original_activation in activation_to_activation_association.items():
        assert (
            encodings_dictionary["activation_encodings"][new_activation]
            == encodings_dictionary["activation_encodings"][original_activation]
        )

    # THEN the encodings for new activations should be associated as displayed in the below dictionar.
    # Case 2: these are derived from parameters
    activation_to_parameter_association = {
        "t": "fc1.weight",
        "t_1": "fc2.weight",
        "t_2": "fc3.weight",
    }

    for new_activation, original_activation in activation_to_parameter_association.items():
        assert (
            encodings_dictionary["activation_encodings"][new_activation]
            == encodings_dictionary["param_encodings"][original_activation]
        )

    # THEN encodings should not have propagated to the following parameters as these are not quantized.
    assert encodings_dictionary["param_encodings"].get("extra_weight") is None
    assert encodings_dictionary["activation_encodings"].get("_softmax") is None
    assert encodings_dictionary["activation_encodings"].get("mm_3") is None
