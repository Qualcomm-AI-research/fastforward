# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import json
import pathlib
import pickle

from typing import Any, TypeAlias

import fastforward as ff
import pytest
import torch

from fastforward.export._export_schemas import LegacySchemaHandler, V1SchemaHandler, V2SchemaHandler
from fastforward.export.module_export import (
    ModuleIORecorder,
    export_modules,
    maybe_dequantize_kwargs,
    maybe_dequantize_tensors,
)
from fastforward.quantization.quant_init import QuantizerCollection
from fastforward.quantization.strict_quantization import strict_quantization_for_module
from fastforward.testing.initialization import initialize_quantizers_to_linear_quantizer

QuantizedModelFixture: TypeAlias = tuple[torch.nn.Module, QuantizerCollection, QuantizerCollection]


def test_module_io_recorder(
    simple_model: QuantizedModelFixture, tmp_path: pathlib.Path, _seed_prngs: int
) -> None:
    """Test that the ModuleIORecorder can record the input and output of a module."""
    data = torch.randn(2, 32, 10)
    model, *_ = simple_model
    model_name = "test_linear_model"
    with ModuleIORecorder(model, model_name) as recorder:
        with strict_quantization_for_module(False, model):
            model(data)
        recorder.store_io_as_dict(tmp_path / f"{model_name}_input_output.pickle")
        _check_module_input_output_has_been_stored(tmp_path, model_name)


@pytest.mark.parametrize(
    "schema_handler_type", [LegacySchemaHandler, V1SchemaHandler, V2SchemaHandler]
)
@pytest.mark.slow
def test_module_export(
    simple_model: QuantizedModelFixture,
    tmp_path: pathlib.Path,
    _seed_prngs: int,
    schema_handler_type: type,
) -> None:
    # GIVEN: a model with quantizers and a collection of modules of interest
    # (in this case linear and relu)
    data = torch.randn(2, 32, 10)
    model, activation_quantizers, parameter_quantizers = simple_model
    model_name = "test_linear_model"

    estimate_model_ranges = initialize_quantizers_to_linear_quantizer(
        model, activation_quantizers, parameter_quantizers
    )
    estimate_model_ranges(data)

    # WHEN: exporting the individual modules AND the entire model.
    linear_modules = ff.mpath.search("**/[cls:torch.nn.Linear]", model)
    relu_modules = ff.mpath.search("**/[cls:torch.nn.ReLU]", model)

    modules = linear_modules | relu_modules
    schema_handler = schema_handler_type()
    paths = export_modules(
        model, (data,), modules, model_name, tmp_path, encoding_schema_handler=schema_handler
    )
    model_path = export_modules(
        model, (data,), model, model_name, tmp_path, encoding_schema_handler=schema_handler
    )[model_name]

    # THEN: the number of exported modules paths should match the number
    # of modules of interest.
    assert len(modules) == len(paths)

    # THEN: the individual modules should be have been exported (i.e. there
    # exists an ONNX file and an encodings file for each of them) and input
    # encodings should have been added to the modules that did not have
    # an activated input quantizer. Also, pickle files containing inputs/outputs/kwargs
    # gathered from torch hooks should be present and have a set structure.
    # Also the stem of each exported module's path should be the same as the name of
    # the module.
    for module, path in zip(modules, paths):
        assert module.full_name == paths[path].stem
        _check_module_files(paths[path], module.full_name)
        _check_module_input_output_has_been_stored(paths[path], module.full_name)

        if module.full_name != "fc1":
            _check_input_encodings_have_been_added(
                paths[path], module.full_name, schema_handler_type
            )

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


def _check_input_encodings_have_been_added(
    path: pathlib.Path, module_name: str, schema_handler_type: type
) -> None:
    encodings_file = path / f"{module_name}.encodings"

    with open(encodings_file) as fp:
        encodings_dictionary = json.load(fp)

    if schema_handler_type is LegacySchemaHandler:
        assert "input" in encodings_dictionary["activation_encodings"]
    elif schema_handler_type is V1SchemaHandler:
        assert "input" in [enc["name"] for enc in encodings_dictionary["activation_encodings"]]
    else:
        assert "input" in [enc["name"] for enc in encodings_dictionary["encodings"]]


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


@pytest.mark.parametrize(
    "schema_handler_type", [LegacySchemaHandler, V1SchemaHandler, V2SchemaHandler]
)
@pytest.mark.slow
def test_schema_handler_cleared_between_modules(
    simple_model: QuantizedModelFixture,
    tmp_path: pathlib.Path,
    schema_handler_type: type,
) -> None:
    # GIVEN a model dummy data and a schema handler
    data = torch.randn(2, 32, 10)
    model, activation_quantizers, parameter_quantizers = simple_model

    estimate_ranges = initialize_quantizers_to_linear_quantizer(
        model, activation_quantizers, parameter_quantizers
    )
    estimate_ranges(data)

    linear_modules = ff.mpath.search("**/[cls:torch.nn.Linear]", model)

    schema_handler = schema_handler_type()

    # WHEN exporting the module through the export_modules function with a schema handler
    export_modules(
        model, (data,), linear_modules, "model", tmp_path, encoding_schema_handler=schema_handler
    )

    # THEN the stored file should be filled with relevant encodings and the handler should be cleared.
    for layer_name in ("fc1", "fc2", "fc3"):
        stored_encodings_file = tmp_path / "model" / layer_name / f"{layer_name}.encodings"

        with open(stored_encodings_file, "r") as f:
            stored_dictionary = json.load(f)

        if not isinstance(schema_handler, V2SchemaHandler):
            # Two parameter encodings (weight/bias)
            assert len(stored_dictionary["param_encodings"]) == 2
            # Two activation encodings (input/output)
            assert len(stored_dictionary["activation_encodings"]) == 2
        else:
            # All the encodings are in a signel dictionary
            assert len(stored_dictionary["encodings"]) == 4

    cleared_dictionary = schema_handler.build_encodings_dictionary()
    if not isinstance(schema_handler, V2SchemaHandler):
        assert len(cleared_dictionary["param_encodings"]) == 0
        assert len(cleared_dictionary["activation_encodings"]) == 0
    else:
        assert len(cleared_dictionary["encodings"]) == 0


@pytest.mark.parametrize(
    "schema_handler_type", [LegacySchemaHandler, V1SchemaHandler, V2SchemaHandler]
)
@pytest.mark.slow
def test_module_export_with_module_kwargs(
    model_with_kwargs: QuantizedModelFixture,
    tmp_path: pathlib.Path,
    _seed_prngs: int,
    schema_handler_type: type,
) -> None:
    # GIVEN: a simple model that accepts kwargs containing quantized tensors
    # Note, some of the custom module submodules also accept kwargs.
    model, activation_quantizers, parameter_quantizers = model_with_kwargs
    model_name = "test_module_kwargs_model"

    data = torch.randn(2, 10)
    module_kwargs = {
        "first_kwarg": torch.randn(2, 10),
        "second_kwarg": torch.randn(2, 10),
    }

    estimate_model_ranges = initialize_quantizers_to_linear_quantizer(
        model, activation_quantizers, parameter_quantizers
    )
    estimate_model_ranges(data, **module_kwargs)

    model(data, **module_kwargs)

    # WHEN exporting the full model through the export_modules function with kwargs
    schema_handler = schema_handler_type()

    paths = export_modules(
        model,
        (data,),
        model,
        model_name,
        tmp_path,
        kwargs=module_kwargs,
        encoding_schema_handler=schema_handler,
    )

    full_model_path = paths[model_name]
    # THEN the module files should be present,
    # the correct number of inputs should be found (in this case 1)
    # the correct number of kwargs should be found(in this case 2)
    _check_module_files(full_model_path, model_name)
    _check_kwarg_module_inputs_have_been_stored(full_model_path, model_name, expected_num_inputs=1)
    _check_module_kwargs_handled(full_model_path, model_name, tuple(module_kwargs.keys()))

    # WHEN: exporting the linear layer
    linear_module_path = export_modules(
        model,
        (data,),
        model.fc1,
        model_name,
        tmp_path,
        kwargs=module_kwargs,
        encoding_schema_handler=schema_handler,
    )

    # THEN the module files should be present,
    # the correct number of inputs should be found (in this case 1)
    # the correct number of kwargs should be found(in this case 0)
    _check_module_files(linear_module_path[model_name], model_name)
    _check_kwarg_module_inputs_have_been_stored(
        linear_module_path[model_name], model_name, expected_num_inputs=1
    )
    _check_module_kwargs_handled(
        linear_module_path[model_name], model_name, original_kwargs_names=()
    )

    custom_mul_module_path = export_modules(
        model,
        (data,),
        model.mul_module,
        model_name,
        tmp_path,
        kwargs=module_kwargs,
        encoding_schema_handler=schema_handler,
    )

    # WHEN: exporting the custom mul layer
    # THEN the module files should be present,
    # the correct number of inputs should be found (in this case 1)
    # the correct number of kwargs should be found(in this case 1)
    _check_module_files(custom_mul_module_path[model_name], model_name)
    _check_kwarg_module_inputs_have_been_stored(
        custom_mul_module_path[model_name], model_name, expected_num_inputs=1
    )
    _check_module_kwargs_handled(
        custom_mul_module_path[model_name], model_name, original_kwargs_names=("other",)
    )


def _check_kwarg_module_inputs_have_been_stored(
    path: pathlib.Path, module_name: str, expected_num_inputs: int
) -> None:
    input_output_location = path / f"{module_name}_input_output.pickle"

    with open(input_output_location, "rb") as fp:
        input_output_dictionary = pickle.load(fp)

    assert len(input_output_dictionary["input"]) == expected_num_inputs

    for entry in input_output_dictionary["input"]:
        assert isinstance(entry[0], torch.Tensor)


def _check_module_kwargs_handled(
    path: pathlib.Path, module_name: str, original_kwargs_names: tuple[str, ...]
) -> None:
    """Verify module kwargs with quantized tensors are properly handled."""
    input_output_location = path / f"{module_name}_input_output.pickle"

    with open(input_output_location, "rb") as fp:
        input_output_dict = pickle.load(fp)

    # verify kwargs were captured and processed
    assert "kwargs" in input_output_dict
    kwargs = input_output_dict["kwargs"]
    assert sorted(original_kwargs_names) == sorted(kwargs)

    # check that quantized tensor kwargs were dequantized
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            assert not isinstance(value, ff.QuantizedTensor), f"kwarg {key} should be dequantized"


def test_maybe_dequantize_kwargs() -> None:
    def _recursive_compare(original: Any, processed: Any) -> None:
        if isinstance(original, torch.Tensor):
            torch.testing.assert_close(original.dequantize(), processed)
        elif isinstance(original, (list, tuple)):
            assert len(original) == len(processed)
            assert type(original) is type(processed)
            for orig_item, proc_item in zip(original, processed):
                _recursive_compare(orig_item, proc_item)
        elif isinstance(original, dict):
            assert set(original.keys()) == set(processed.keys())
            for key in original:
                _recursive_compare(original[key], processed[key])
        else:
            # Non-tensor, non-container values should be identical
            assert original == processed

    # GIVEN a dictionary of kwargs, containing different types.
    quantized_tensors_list = [ff.random.random_quantized((5, 5)) for _ in range(5)]
    quantized_tensors_tuple = tuple([ff.random.random_quantized((5, 5)) for _ in range(5)])
    unquantized_tensors_list = [torch.randn(2, 2) for _ in range(5)]
    unquantized_tensors_tuple = tuple([torch.randn(2, 2) for _ in range(5)])

    single_quantized_tensor = ff.random.random_quantized((5, 5))
    single_unquantized_tensor = torch.randn(5, 5)

    # Additional test data for complex structures
    nested_quantized_list = [
        [ff.random.random_quantized((3, 3)), ff.random.random_quantized((2, 2))]
    ]
    mixed_nested_list = [[single_quantized_tensor, single_unquantized_tensor], [42, "string"]]
    nested_dict = {
        "inner": {"quantized": ff.random.random_quantized((4, 4)), "regular": torch.randn(3, 3)}
    }
    complex_nested_structure = {
        "tensors": [
            ff.random.random_quantized((2, 2)),
            [ff.random.random_quantized((3, 3)), single_unquantized_tensor],
        ],
        "metadata": {"count": 2, "nested_tensors": [ff.random.random_quantized((1, 1))]},
    }

    kwargs = {
        "single_quantized_tensor": single_quantized_tensor,
        "single_unquantized_tensor": single_unquantized_tensor,
        "quantized_tensors_list": quantized_tensors_list,
        "quantized_tensors_tuple": quantized_tensors_tuple,
        "unquantized_tensors_list": unquantized_tensors_list,
        "unquantized_tensors_tuple": unquantized_tensors_tuple,
        "list_int": [1, 2, 3],
        "none_val": None,
        "empty_list": [],
        "boolean": False,
        "string": "sth",
        "float": 0.5,
        "none": None,
        # More complex cases
        "nested_quantized_list": nested_quantized_list,
        "mixed_nested_list": mixed_nested_list,
        "nested_dict": nested_dict,
        "complex_nested_structure": complex_nested_structure,
    }

    to_dequantize_keys = set([
        "single_quantized_tensor",
        "quantized_tensors_list",
        "quantized_tensors_tuple",
        "nested_quantized_list",
        "mixed_nested_list",
        "nested_dict",
        "complex_nested_structure",
    ])

    # WHEN calling the `maybe_dequantize_kwargs` function
    result_kwargs, result_quantizers = maybe_dequantize_kwargs(kwargs)

    # THEN the dictionary keys should be the same
    assert result_kwargs.keys() == kwargs.keys()
    # THEN quantizer values should be collected only for the keys that contain quantized tensors
    for key, value in result_quantizers.items():
        if key not in to_dequantize_keys:
            assert value is None or not any(value)

    # THEN the two dictionary should match exactly when the quantized tensors are dequantized
    for key, value in kwargs.items():
        new_value = result_kwargs[key]
        _recursive_compare(value, new_value)


def test_maybe_dequantize_tensors() -> None:
    # GIVEN: quantized and unquantized tensors
    quantized_tensor = ff.random.random_quantized((3, 3))
    unquantized_tensor = torch.randn(2, 2)

    # WHEN calling the maybe dequantize operation.
    result_tensors, result_settings = maybe_dequantize_tensors((quantized_tensor,))

    # THEN for the quantized tensor there should be one tensor and one quantizer
    # settings in the returned variables. The tensor should NOT be a quantized tensor and
    # it should match the original tensor when that is dequantized.
    assert len(result_tensors) == len(result_settings) == 1
    assert not isinstance(result_tensors[0], ff.QuantizedTensor)
    torch.testing.assert_close(result_tensors[0], quantized_tensor.dequantize())
    assert result_settings[0] is not None

    # WHEN calling the maybe dequantize operation.
    result_tensors, result_settings = maybe_dequantize_tensors((unquantized_tensor,))

    # THEN for the quantized tensor there should be one tensor and one quantizer
    # settings in the output. The tensor should be left untouched, and the sole
    # entry in the quantizer settings should be None.
    assert len(result_tensors) == len(result_settings) == 1
    assert result_settings[0] is None
    assert isinstance(result_tensors[0], torch.Tensor)
    torch.testing.assert_close(result_tensors[0], unquantized_tensor)


def test_maybe_dequantize_tensors_mixed_tuple() -> None:
    # GIVEN: A tuple of 5 tensors with mixed quantized and unquantized tensors
    tensor1 = ff.random.random_quantized((2, 2))
    tensor2 = torch.randn(3, 3)
    tensor3 = ff.random.random_quantized((4, 4))
    tensor4 = torch.randn(1, 5)
    tensor5 = ff.random.random_quantized((3, 2))

    input_tensors = (tensor1, tensor2, tensor3, tensor4, tensor5)
    is_input_quantized = (True, False, True, False, True)

    # WHEN calling the maybe dequantize operation
    result_tensors, result_settings = maybe_dequantize_tensors(input_tensors)

    # THEN the result should have 5 tensors and 5 quantizer settings (with None for unquantized)
    assert len(result_tensors) == len(result_settings) == 5

    # THEN verify each tensor is properly processed
    for tensor, org_tensor, setting, is_quant in zip(result_tensors, input_tensors, result_settings, is_input_quantized):
        # No quantized tensors should be present
        assert not isinstance(tensor, ff.QuantizedTensor)
        # The tensors should always much (regardless if they were quant or non-quant)
        torch.testing.assert_close(tensor, org_tensor.dequantize())
        if is_quant:
            assert setting is not None
        else:
            assert setting is None
