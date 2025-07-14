# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


import copy

from pathlib import Path
from typing import Any
from unittest.mock import NonCallableMock, patch

import fastforward as ff
import pytest
import torch
import yaml


def assert_yaml_properties(path: Path, **kwargs: Any) -> None:
    """Assert that a YAML property has the expected value.

    Args:
        path (Path): The path to the YAML file.
        **kwargs: Property paths and their expected values. Use dot notation for nested properties.

    Returns:
        None
    """
    with open(path, "r") as f:
        config = yaml.load(f, yaml.Loader)
    mismatches = []
    for property, expected_value in kwargs.items():
        current_value = config
        for prop in property.split("."):
            if prop not in current_value:
                mismatches.append(f"Property {property} not found in YAML file {path}")
                break
            current_value = current_value[prop]
        else:
            if current_value != str(expected_value):
                mismatches.append(
                    f"Property {property} in YAML file {path} has value {current_value}, "
                    "but expected {expected_value}"
                )
    assert not mismatches, "\n".join(mismatches)


def test_model_save_quantization_state_with_name_or_path(tmp_path: Path) -> None:
    """Test that a model is saved correctly when providing a `name_or_path`."""
    # GIVEN: A Quantized module
    name_or_path = "test"
    model = ff.nn.QuantizedModule()

    # WHEN: Saving the quantization state with the name_or_path
    config_path = model.save_quantization_state(name_or_path=name_or_path, cache_dir=tmp_path)

    # THEN: The saved config should contain the correct name_or_path
    assert_yaml_properties(config_path, name_or_path=name_or_path)


def test_model_save_quantization_state_with_shared_quantizer(tmp_path: Path) -> None:
    """Test if error is raised when shared quantizers are saved."""
    # GIVEN: A Quantized module with a shared quantizer
    name_or_path = "test"
    model = ff.nn.QuantizedModule()

    q = ff.nn.LinearQuantizer(8, granularity=ff.PerTile((1, 2)))
    q.quantization_range = (-10, 10)
    model.register_quantizer("input_quantizer", q)
    model.register_quantizer("output_quantizer", q)

    # WHEN: Saving the quantization state with a shared quantizer
    # THEN: exception should be raised
    with pytest.raises(RuntimeError):
        model.save_quantization_state(name_or_path=name_or_path, cache_dir=tmp_path)


def test_model_save_quantization_state_with_config_name_or_path(tmp_path: Path) -> None:
    """Test that the model identifier is saved correctly for HuggingFace models."""
    name_or_path = "test"
    # GIVEN: A Quantized module with `config.name_or_path` attibute to simulate
    # models from transformers (https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModel)
    model = ff.nn.QuantizedModule()
    with patch.object(
        model,
        "config",
        NonCallableMock(spec=["name_or_path"], name_or_path=name_or_path),
        create=True,
    ):
        # WHEN: Saving the quantization state
        config_path = model.save_quantization_state(cache_dir=tmp_path)
        # THEN: The saved config should contain the correct name_or_path
        assert_yaml_properties(config_path, name_or_path=name_or_path)


def test_model_save_quantization_state_with_config_name_or_path_override(tmp_path: Path) -> None:
    """Test priority of name_or_path."""
    # GIVEN: A Quantized module with `config.name_or_path` attibute to simulate
    # models from transformers (https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModel)
    name_or_path = Path("test2")
    transformers_version = 3
    model = ff.nn.QuantizedModule()
    with patch.object(
        model,
        "config",
        NonCallableMock(
            spec=["name_or_path", "transformers_version"],
            name_or_path="test",
            transformers_version=transformers_version,
        ),
        create=True,
    ):
        # WHEN: Saving the quantization state with provided name_or_path
        config_path = model.save_quantization_state(name_or_path=name_or_path, cache_dir=tmp_path)
        # THEN: The name_or_path should match to manually provided value
        assert_yaml_properties(
            config_path, name_or_path=name_or_path, transformers_version=transformers_version
        )


def test_model_save_quantization_state_without_name_or_path_or_config(tmp_path: Path) -> None:
    """Test that a RuntimeError is raised when saving the model without identifier."""
    # GIVEN: A quantized module
    model = ff.nn.QuantizedModule()
    # WHEN: Saving the quantization state without provided name_or_path
    # THEN: The exception should be raised
    with pytest.raises(RuntimeError) as exc_info:
        model.save_quantization_state(cache_dir=tmp_path)
        assert "name_or_path" in str(exc_info.value)


# Tests for load_quantization_state function


def test_load_quantization_state_with_name_or_path(tmp_path: Path) -> None:
    """Test successful loading of quantization state with explicit name_or_path."""
    name_or_path = "test_model"

    # GIVEN: A saved quantized module
    model = ff.nn.QuantizedModule()
    q = ff.nn.LinearQuantizer(8, granularity=ff.PerTile((1, 2, 3)))
    q.quantization_range = (-9, 9)
    model.register_quantizer("test_quantizer", q)
    model.save_quantization_state(name_or_path=name_or_path, cache_dir=tmp_path)

    # GIVEN: A quantized module
    new_model = ff.nn.QuantizedModule()
    new_model.test_quantizer = ff.nn.QuantizerStub()
    # WHEN: Loading the quantization state
    new_model.load_quantization_state(name_or_path=name_or_path, cache_dir=tmp_path)
    # THEN: No errors raised


def test_load_quantization_state_with_config_name_or_path(tmp_path: Path) -> None:
    """Test loading quantization state for HuggingFace models."""
    name_or_path = "test_model_config"

    # GIVEN: A saved quantized module with `config.name_or_path` attibute to simulate
    # models from transformers (https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModel)
    model = ff.nn.QuantizedModule()
    q = ff.nn.LinearQuantizer(8)
    q.quantization_range = (-9, 9)
    model.register_quantizer("test_quantizer", q)
    with patch.object(
        model,
        "config",
        NonCallableMock(spec=["name_or_path"], name_or_path=name_or_path),
        create=True,
    ):
        model.save_quantization_state(cache_dir=tmp_path)

    # GIVEN: A quantized module with `config.name_or_path` attibute to simulate
    # models from transformers (https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModel)
    new_model = ff.nn.QuantizedModule()
    new_model.test_quantizer = ff.nn.QuantizerStub()
    with patch.object(
        new_model,
        "config",
        NonCallableMock(spec=["name_or_path"], name_or_path=name_or_path),
        create=True,
    ):
        # WHEN: Loading the quantization state
        new_model.load_quantization_state(cache_dir=tmp_path)
    # THEN: No errors raised


def test_load_quantization_state_without_name_or_path_or_config(tmp_path: Path) -> None:
    """Test that RuntimeError is raised when loading without model identifier."""
    # GIVEN: A module without model identifier
    model = ff.nn.QuantizedModule()

    # WHEN: Loading the quantization state
    # THEN: An error is raised
    with pytest.raises(RuntimeError) as exc_info:
        model.load_quantization_state(cache_dir=tmp_path)

    assert "Unable to detect the model identifier" in str(exc_info.value)


def test_load_quantization_state_missing_config_file(tmp_path: Path) -> None:
    """Test that FileNotFoundError is raised when config file is missed."""
    name_or_path = "nonexistent_model"
    # GIVEN: A saved quantization state without 'config.yaml'
    model = ff.nn.QuantizedModule()

    # WHEN: Loading the quantization state
    # THEN: An error about missed 'config.yaml' is raised
    with pytest.raises(FileNotFoundError) as exc_info:
        model.load_quantization_state(name_or_path=name_or_path, cache_dir=tmp_path)

    assert "Quantization state config not found at" in str(exc_info.value)


def test_load_quantization_state_missing_model_file(tmp_path: Path) -> None:
    """Test that FileNotFoundError is raised when model file is missed."""
    name_or_path = "test_model_missing_safetensors"
    # GIVEN: A saved quantization state without 'model.safetensors'
    model = ff.nn.QuantizedModule()
    config_path = model.save_quantization_state(name_or_path=name_or_path, cache_dir=tmp_path)
    tensors_path = config_path.parent / "model.safetensors"
    tensors_path.unlink()

    # WHEN: Loading the quantization state
    # THEN: An error about missed 'model.safetensors' is raised
    with pytest.raises(FileNotFoundError) as exc_info:
        model.load_quantization_state(name_or_path=name_or_path, cache_dir=tmp_path)
    assert "Quantization state model not found at" in str(exc_info.value)


def test_load_quantization_state_unsupported_version(tmp_path: Path) -> None:
    """Test that ValueError is raised for unsupported config version."""
    name_or_path = "test_model_bad_version"

    # GIVEN: A saved quantization state with unknown version
    model = ff.nn.QuantizedModule()
    config_path = model.save_quantization_state(name_or_path=name_or_path, cache_dir=tmp_path)
    tensors_path = config_path.parent / "model.safetensors"
    tensors_path.touch()

    with open(config_path, "r") as file:
        config = yaml.load(file, yaml.Loader)
    config["version"] = "2.0"
    with open(config_path, "w") as file:
        yaml.dump(config, file)

    # WHEN: Loading the quantization state
    # THEN: An error abotu unsupported quantization state versionis raised
    with pytest.raises(ValueError) as exc_info:
        model.load_quantization_state(name_or_path=name_or_path, cache_dir=tmp_path)

    assert "Unsupported quantization state version:" in str(exc_info.value)


def test_load_quantization_state_name_or_path_mismatch_warning(tmp_path: Path) -> None:
    """Test that warning is logged when name_or_path doesn't match saved state."""
    saved_name = "original_model"
    load_name = "different_model"

    # GIVEN: A saved quantization state
    model = ff.nn.QuantizedModule()
    with patch.object(
        model,
        "config",
        NonCallableMock(spec=["name_or_path"], name_or_path=saved_name),
        create=True,
    ):
        config_path = model.save_quantization_state(cache_dir=tmp_path)

    with patch.object(
        model, "config", NonCallableMock(spec=["name_or_path"], name_or_path=load_name), create=True
    ):
        # WHEN: Loading the quantization state with different model identifier
        # THEN: An error about mismatched model identifier raised
        with pytest.raises(RuntimeError) as exc_info:
            model.load_quantization_state(name_or_path=config_path, cache_dir=tmp_path)
        assert "Model identifier mismatch" in str(exc_info.value)


@pytest.mark.parametrize(
    "model,data",
    (
        (torch.nn.Linear(10, 10, bias=True), torch.rand((10, 10))),
        (torch.nn.Conv1d(10, 10, kernel_size=3), torch.rand((10, 10))),
    ),
)
def test_load_quantization_state_integration_with_save(
    model: torch.nn.Module, data: torch.Tensor, tmp_path: Path
) -> None:
    """Test complete save/load cycle with quantizer state preservation."""
    name_or_path = "integration_test_model"

    # GIVEN: A model and deserialized model
    duplicated_model = copy.deepcopy(model)
    ff.quantize_model(model)
    ff.quantize_model(duplicated_model)

    q1 = ff.nn.LinearQuantizer(8, granularity=ff.PerTile((1, 2)), quantized_dtype=torch.float32)
    q1.quantization_range = (-10, 10)
    q2 = ff.nn.LinearQuantizer(16, granularity=ff.PerChannel(), device=torch.device("cpu"))
    q2.quantization_range = (-5, 5)

    model.register_quantizer("input_quantizer", q1)
    model.register_quantizer("output_quantizer", q2)
    model.save_quantization_state(name_or_path=name_or_path, cache_dir=tmp_path)

    # WHEN: deserialize model
    duplicated_model.load_quantization_state(name_or_path=name_or_path, cache_dir=tmp_path)

    # THEN: deserialized quantizers have the same settings
    assert duplicated_model.input_quantizer.quantization_range == q1.quantization_range
    assert duplicated_model.input_quantizer.granularity == q1.granularity
    assert duplicated_model.input_quantizer.quantized_dtype == q1.quantized_dtype
    torch.testing.assert_close(duplicated_model.input_quantizer.scale, q1.scale)
    torch.testing.assert_close(duplicated_model.input_quantizer.offset, q1.offset)
    assert duplicated_model.output_quantizer.quantization_range == q2.quantization_range
    assert duplicated_model.output_quantizer.granularity == q2.granularity
    torch.testing.assert_close(duplicated_model.output_quantizer.scale, q2.scale)
    torch.testing.assert_close(duplicated_model.output_quantizer.offset, q2.offset)
    assert duplicated_model.output_quantizer.scale.device == q2.scale.device

    with ff.strict_quantization(False):
        # WHEN: Feed to both models the same input
        # THEN: Models produce the same output
        torch.testing.assert_close(model(data).dequantize(), duplicated_model(data).dequantize())
