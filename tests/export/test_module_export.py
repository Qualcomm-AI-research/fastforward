# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import json
import pathlib
import pickle

import fastforward as ff
import pytest
import torch

from fastforward.export.module_export import export_modules
from tests.export.export_utils import QuantizedModelFixture, activate_quantizers


@pytest.mark.xfail_due_to_too_new_torch
@pytest.mark.slow
@ff.flags.context(ff.strict_quantization, False)
def test_module_export(
    simple_model: QuantizedModelFixture, tmp_path: pathlib.Path, _seed_prngs: int
) -> None:
    # GIVEN: a model with quantizers and a collection of modules of interest
    # (in this case linear and relu)
    data = torch.randn(2, 32, 10)
    model, activation_quantizers, parameter_quantizers = simple_model
    model_name = "test_linear_model"

    activate_quantizers(model, data, activation_quantizers, parameter_quantizers)

    # WHEN: exporting the individual modules AND the entire model.
    linear_modules = ff.mpath.search("**/[cls:torch.nn.Linear]", model)
    relu_modules = ff.mpath.search("**/[cls:torch.nn.ReLU]", model)

    modules = linear_modules | relu_modules
    paths = export_modules(model, (data,), modules, model_name, tmp_path)
    model_path = export_modules(model, (data,), model, model_name, tmp_path)[model_name]

    # THEN: the number of exported modules paths should match the number
    # of modules of interest.
    assert len(modules) == len(paths)

    # THEN: the individual modules should be have been exported (i.e. there
    # exists an ONNX file and an encodings file for each of them) and input
    # encodings should have been added to the modules that did not have
    # an activated input quantizer. Also, pickle files containing inputs/outputs/kwargs
    # gathered from torch hooks should be present and have a set structure.
    for module, path in zip(modules, paths):
        _check_module_files(paths[path], module.full_name)
        _check_module_input_output_has_been_stored(paths[path], module.full_name)

        if module.full_name != "fc1":
            _check_input_encodings_have_been_added(paths[path], module.full_name)

    # THEN: the above checks should also work when exporting the full model.
    # NB: Input encodings are not altered, since the full model has an activated
    # input quantizer, so no need to check that.
    _check_module_files(model_path, model_name)
    _check_module_input_output_has_been_stored(model_path, model_name)


def _check_module_files(path: pathlib.Path, module_name: str) -> None:
    encodings_file = path / f"{module_name}.encodings"
    onnx_file = path / f"{module_name}.onnx"

    assert encodings_file.is_file() and encodings_file.stat().st_size > 0
    assert onnx_file.is_file() and onnx_file.stat().st_size > 0


def _check_input_encodings_have_been_added(path: pathlib.Path, module_name: str) -> None:
    encodings_file = path / f"{module_name}.encodings"

    with open(encodings_file) as fp:
        encodings_dictionary = json.load(fp)

    assert "input" in encodings_dictionary["activation_encodings"]


def _check_module_input_output_has_been_stored(path: pathlib.Path, module_name: str) -> None:
    input_output_location = path / f"{module_name}_input_output.pickle"
    expected_keys = ["input", "output", "kwargs"]

    with open(input_output_location, "rb") as fp:
        input_output_dictionary = pickle.load(fp)

    assert isinstance(input_output_dictionary, dict)
    assert set(expected_keys) <= set(input_output_dictionary.keys())

    assert len(input_output_dictionary["input"]) == 1
    assert len(input_output_dictionary["output"]) == 1

    assert isinstance(input_output_dictionary["input"][0], torch.Tensor)
    assert isinstance(input_output_dictionary["output"][0], torch.Tensor)
    assert input_output_dictionary["kwargs"] == {}
