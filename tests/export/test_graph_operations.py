# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import json
import pathlib

from typing import TypeAlias

import fastforward as ff
import pytest
import torch

from fastforward.export.export import export
from fastforward.quantization.quant_init import QuantizerCollection
from fastforward.testing.initialization import initialize_quantizers_to_linear_quantizer

QuantizedModelFixture: TypeAlias = tuple[torch.nn.Module, QuantizerCollection, QuantizerCollection]


@pytest.mark.slow
def test_encodings_propagation(
    tmp_path: pathlib.Path,
    simple_quant_model_with_non_quant_ops: QuantizedModelFixture,
    _seed_prngs: int,
) -> None:
    # GIVEN a quantized model.
    granularity = ff.PerTensor()
    data = torch.randn(32, 10)
    quant_model, activation_quantizers, parameter_quantizers = simple_quant_model_with_non_quant_ops
    model_name = "test_export_function"
    output_directory = tmp_path / model_name
    output_model_directory = pathlib.Path(output_directory) / model_name
    encodings_file_path = output_model_directory.with_suffix(".encodings")

    with ff.strict_quantization(False):
        estimate_model_ranges = initialize_quantizers_to_linear_quantizer(
            quant_model,
            activation_quantizers,
            parameter_quantizers,
            granularity_parameters=granularity,
        )
        estimate_model_ranges(data)

    # GIVEN the exported artifacts from that model and its original encodings file.
    export(quant_model, (data,), output_directory, model_name, enable_encodings_propagation=False)

    with open(encodings_file_path, "r") as file:
        encodings_dictionary = json.load(file)

    original_param_names = encodings_dictionary["param_encodings"].keys()
    original_activation_names = encodings_dictionary["activation_encodings"].keys()

    # WHEN exporting the same model with encoding propagation.
    export(quant_model, (data,), output_directory, model_name, enable_encodings_propagation=True)

    with open(encodings_file_path, "r") as file:
        encodings_dictionary = json.load(file)

    new_param_names = encodings_dictionary["param_encodings"].keys()
    new_activation_names = encodings_dictionary["activation_encodings"].keys()

    # THEN the parameters should be the same, as there is no encoding propagation to new parameters.
    assert new_param_names == original_param_names

    # THEN all the original activation should be preserved.
    assert original_activation_names <= new_activation_names

    # THEN the encodings for new activations should be associated as displayed in the below dictionary.
    # Case 1: these are derived from other activations
    activation_to_activation_association = {"view": "mm", "view_1": "mm"}

    for new_activation, original_activation in activation_to_activation_association.items():
        assert (
            encodings_dictionary["activation_encodings"][new_activation]
            == encodings_dictionary["activation_encodings"][original_activation]
        )

    # THEN the encodings for new activations should be associated as displayed in the below dictionary.
    # Case 2: these are derived from parameters
    activation_to_parameter_association = {
        "permute": "fc1.weight",
        "permute_1": "fc2.weight",
        "permute_2": "fc3.weight",
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
