# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import dataclasses

from typing import Any

import pytest
import torch

from fastforward.export._export_schemas import (
    EncodingSchemaHandler,
    LegacySchemaHandler,
    V1SchemaHandler,
    V2SchemaHandler,
)
from fastforward.export._export_types import QNNDefaultConfig, QuantParametersDict

LEGACY_PERTENSOR_PERCHANNEL_EXPECTED_RESULTS = {
    "version": "0.6.1",
    "param_encodings": {
        "conv1.weight": (
            {
                "bitwidth": 8,
                "dtype": "int",
                "is_symmetric": "True",
                "min": -1.5360000133514404,
                "max": 1.5240000486373901,
                "offset": -128,
                "scale": 0.012000000104308128,
            },
            {
                "bitwidth": 8,
                "dtype": "int",
                "is_symmetric": "True",
                "min": -1.9199999570846558,
                "max": 1.9049999713897705,
                "offset": -128,
                "scale": 0.014999999664723873,
            },
            {
                "bitwidth": 8,
                "dtype": "int",
                "is_symmetric": "True",
                "min": -1.4079999923706055,
                "max": 1.3969999551773071,
                "offset": -128,
                "scale": 0.010999999940395355,
            },
        )
    },
    "activation_encodings": {
        "input": (
            {
                "bitwidth": 8,
                "dtype": "int",
                "is_symmetric": "False",
                "min": 0.0,
                "max": 7.904999732971191,
                "offset": 0,
                "scale": 0.03099999949336052,
            },
        ),
        "layer1_output": (
            {
                "bitwidth": 8,
                "dtype": "int",
                "is_symmetric": "True",
                "min": -3.200000047683716,
                "max": 3.174999952316284,
                "offset": -128,
                "scale": 0.02500000037252903,
            },
        ),
    },
    "quantizer_args": dataclasses.asdict(QNNDefaultConfig()),
}


V1_PERTENSOR_PERCHANNEL_EXPECTED_RESULTS = {
    "version": "1.0.0",
    "param_encodings": [
        {
            "name": "conv1.weight",
            "dtype": "INT",
            "enc_type": "PER_CHANNEL",
            "is_sym": True,
            "bw": 8,
            "scale": [0.012000000104308128, 0.014999999664723873, 0.010999999940395355],
            "offset": [-128.0, -128.0, -128.0],
        }
    ],
    "activation_encodings": [
        {
            "name": "input",
            "dtype": "INT",
            "enc_type": "PER_TENSOR",
            "is_sym": False,
            "bw": 8,
            "scale": [0.03099999949336052],
            "offset": [0.0],
        },
        {
            "name": "layer1_output",
            "dtype": "INT",
            "enc_type": "PER_TENSOR",
            "is_sym": True,
            "bw": 8,
            "scale": [0.02500000037252903],
            "offset": [-128.0],
        },
    ],
    "quantizer_args": dataclasses.asdict(QNNDefaultConfig()),
}

V2_PERTENSOR_PERCHANNEL_EXPECTED_RESULTS = {
    "version": "2.0.0",
    "encodings": [
        {
            "name": "input",
            "output_dtype": "uint8",
            "y_scale": 0.03099999949336052,
            "y_zero_point": 0,
        },
        {
            "name": "conv1.weight",
            "output_dtype": "int8",
            "y_scale": [0.012000000104308128, 0.014999999664723873, 0.010999999940395355],
            "axis": 0,
        },
        {"name": "layer1_output", "output_dtype": "int8", "y_scale": 0.02500000037252903},
    ],
    "quantizer_args": dataclasses.asdict(QNNDefaultConfig()),
}

V1_PERBLOCK_1D_EXPECTED = {
    "version": "1.0.0",
    "param_encodings": [
        {
            "name": "linear.weight",
            "dtype": "INT",
            "enc_type": "PER_BLOCK",
            "is_sym": True,
            "bw": 8,
            "block_size": 2,
            "scale": [0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028],
            "offset": [-128.0, -128.0, -128.0, -128.0, -128.0, -128.0, -128.0, -128.0],
        }
    ],
    "activation_encodings": [],
    "quantizer_args": dataclasses.asdict(QNNDefaultConfig()),
}

V2_PERBLOCK_1D_EXPECTED = {
    "version": "2.0.0",
    "encodings": [
        {
            "name": "linear.weight",
            "output_dtype": "int8",
            "y_scale": [
                [0.021, 0.022],
                [0.023, 0.024],
                [0.025, 0.026],
                [0.027, 0.028],
            ],  # Nested: 4 blocks × 2 channels
            "axis": 0,
            "block_size": 2,
        }
    ],
    "quantizer_args": dataclasses.asdict(QNNDefaultConfig()),
}


@pytest.mark.parametrize(
    "handler_class,expected_result",
    [
        (LegacySchemaHandler, LEGACY_PERTENSOR_PERCHANNEL_EXPECTED_RESULTS),
        (V1SchemaHandler, V1_PERTENSOR_PERCHANNEL_EXPECTED_RESULTS),
        (V2SchemaHandler, V2_PERTENSOR_PERCHANNEL_EXPECTED_RESULTS),
    ],
)
def test_schema_handler_pertensor_perchannel(
    handler_class: type[EncodingSchemaHandler],
    expected_result: dict[str, Any],
) -> None:
    """Test basic schema handling with only per tensor and per channel quantization."""
    # GIVEN a schema handler with mixed symmetric/asymmetric quantization
    handler = handler_class()
    tensor_sets = {
        "inputs": {"input"},
        "activations": {"layer1_output"},
        "parameters": {"conv1.weight"},
    }

    quantization_logs: dict[str, QuantParametersDict] = {
        "input": {
            "scale": torch.tensor([0.031]),
            "offset": torch.tensor([128.0]),
            "num_bits": 8,
            "data_shape": [1, 3, 224, 224],
            "tile_size": [1, 3, 224, 224],
        },
        "conv1.weight": {
            "scale": torch.tensor([0.012, 0.015, 0.011]),  # Per-channel
            "offset": torch.tensor([0.0, 0.0, 0.0]),
            "num_bits": 8,
            "data_shape": [3, 3, 3, 3],
            "tile_size": [1, 3, 3, 3],
        },
        "layer1_output": {
            "scale": torch.tensor([0.025]),
            "offset": torch.tensor([0.0]),  # Symmetric
            "num_bits": 8,
            "data_shape": [1, 3, 222, 222],
            "tile_size": [1, 3, 222, 222],
        },
    }

    # WHEN building the encodings dictionary
    for name, encoding in quantization_logs.items():
        is_param_encoding = name in tensor_sets["parameters"]
        handler.add_encoding(name, encoding, is_param_encoding)

    encodings_dict = handler.build_encodings_dictionary()

    # THEN the constructed dictionary should match the expected results.
    _assert_dictionary_structure(expected_result, encodings_dict)


def test_v1_schema_rejects_perchannel_non_zero_axis() -> None:
    """Test that V1 schema rejects per-channel quantization on non-zero axes."""
    # GIVEN a V1 schema handler and per-channel data on axis 1
    handler = V1SchemaHandler()

    per_channel_axis1_logs: dict[str, QuantParametersDict] = {
        "conv2d.weight": {
            "scale": torch.tensor([0.01, 0.02, 0.03]),  # 3 channels on axis 1
            "offset": torch.tensor([0.0, 0.0, 0.0]),
            "num_bits": 8,
            "data_shape": [4, 3, 5, 5],  # [out_channels, in_channels, H, W]
            "tile_size": [4, 1, 5, 5],  # Per-channel on axis 1 (in_channels)
        }
    }

    # WHEN attempting to add per-channel encoding on axis 1
    # THEN it should raise ValueError about axis 0 requirement
    with pytest.raises(ValueError):
        for name, encoding in per_channel_axis1_logs.items():
            handler.add_encoding(name, encoding, is_param=True)


@pytest.mark.parametrize(
    "handler_class,expected_result,should_raise",
    [
        (LegacySchemaHandler, None, ValueError),
        (V1SchemaHandler, V1_PERBLOCK_1D_EXPECTED, None),
        (V2SchemaHandler, V2_PERBLOCK_1D_EXPECTED, None),
    ],
)
def test_schema_handler_perblock_1d(
    handler_class: type[EncodingSchemaHandler],
    expected_result: dict[str, Any] | None,
    should_raise: type[Exception] | None,
) -> None:
    """Test PerBlock 1D quantization handling."""
    # GIVEN a schema handler and some dummy inputs
    handler = handler_class()
    tensor_sets = {"inputs": set(), "activations": set(), "parameters": {"linear.weight"}}
    quantization_logs: dict[str, QuantParametersDict] = {
        "linear.weight": {
            "scale": torch.tensor([
                0.021,
                0.022,
                0.023,
                0.024,
                0.025,
                0.026,
                0.027,
                0.028,
            ]),
            "offset": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # Symmetric
            "num_bits": 8,
            "data_shape": [8, 2],
            "tile_size": [2, 1],
        },
    }

    # WHEN building the dictionary
    # THEN the legacy schema should raise an error as it does not support Per Block quantization.
    if should_raise:
        with pytest.raises(should_raise):
            for name, encoding in quantization_logs.items():
                is_param_encoding = name in tensor_sets["parameters"]
                handler.add_encoding(name, encoding, is_param_encoding)

    # WHEN building the dictionary
    else:
        for name, encoding in quantization_logs.items():
            is_param_encoding = name in tensor_sets["parameters"]
            handler.add_encoding(name, encoding, is_param_encoding)

        encodings_dict = handler.build_encodings_dictionary()
        assert expected_result is not None
        # THEN the constructed dictionary should match the expected results
        _assert_dictionary_structure(expected_result, encodings_dict)


@pytest.mark.parametrize(
    "handler_class,should_raise",
    [
        (LegacySchemaHandler, ValueError),
        (V1SchemaHandler, ValueError),
        (V2SchemaHandler, ValueError),
    ],
)
def test_schema_handler_perblock_2d(
    handler_class: type[EncodingSchemaHandler],
    should_raise: type[Exception],
) -> None:
    """Test PerBlock 2D quantization handling."""
    # GIVEN a schema handler and some dummy inputs
    handler = handler_class()
    tensor_sets = {"inputs": set(), "activations": set(), "parameters": {"conv2d.weight"}}
    quantization_logs: dict[str, QuantParametersDict] = {
        "conv2d.weight": {
            "scale": torch.tensor([0.01, 0.02, 0.03, 0.04, 0.05, 0.06]),
            "offset": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "num_bits": 8,
            "data_shape": [6, 6],
            "tile_size": [2, 3],
        },
    }

    # WHEN building the dictionary
    # THEN all schemas should raise an error:
    # - Legacy: does not support Per Block quantization
    # - V1: does not support 2D block quantization
    # - V2: QAIRT V2.0.0 only supports single-axis block quantization
    with pytest.raises(should_raise):
        for name, encoding in quantization_logs.items():
            is_param_encoding = name in tensor_sets["parameters"]
            handler.add_encoding(name, encoding, is_param_encoding)


def test_v2_schema_perblock_1d_nested_array() -> None:
    """Test V2 schema with valid single-axis block quantization."""
    # GIVEN a 2D tensor [8, 16] with block quantization with (2, 16) tile size
    handler = V2SchemaHandler()
    tensor_sets = {"parameters": {"conv1d.weight"}}

    quantization_logs: dict[str, QuantParametersDict] = {
        "conv1d.weight": {
            "scale": torch.tensor([0.01, 0.02, 0.03, 0.04]),
            "offset": torch.tensor([0.0, 0.0, 0.0, 0.0]),
            "num_bits": 8,
            "data_shape": [8, 16],
            "tile_size": [2, 16],
        },
    }

    expected_results = {
        "version": "2.0.0",
        "encodings": [
            {
                "name": "conv1d.weight",
                "output_dtype": "int8",
                "y_scale": [[0.01], [0.02], [0.03], [0.04]],
                "axis": 0,
                "block_size": 2,
            }
        ],
        "quantizer_args": dataclasses.asdict(QNNDefaultConfig()),
    }

    # WHEN adding encodings with single-axis block quantization
    for name, encoding in quantization_logs.items():
        is_param_encoding = name in tensor_sets["parameters"]
        handler.add_encoding(name, encoding, is_param_encoding)

    encodings_dict = handler.build_encodings_dictionary()

    # THEN the constructed dictionary should match the expected V2 single-axis results
    _assert_dictionary_structure(expected_results, encodings_dict)


def test_v2_schema_perblock_2d_nested_array() -> None:
    # GIVEN a 2D tensor [8, 2] with block quantization with (2, 1) tile size
    handler = V2SchemaHandler()
    quantization_logs: dict[str, QuantParametersDict] = {
        "conv_weight": {
            # 8 scale values: 4 blocks × 2 channels = 8 total
            "scale": torch.tensor([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]),
            "offset": torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]),
            "num_bits": 8,
            "data_shape": [8, 2],
            "tile_size": [2, 1],
        }
    }

    tensor_sets = {"parameters": {"conv2d.weight"}}

    expected_results = {
        "version": "2.0.0",
        "encodings": [
            {
                "name": "conv_weight",
                "output_dtype": "uint8",
                "axis": 0,
                "block_size": 2,
                "y_scale": [[0.01, 0.02], [0.03, 0.04], [0.05, 0.06], [0.07, 0.08]],
                "y_zero_point": [[-118.0, -108.0], [-98.0, -88.0], [-78.0, -68.0], [-58.0, -48.0]],
            }
        ],
        "quantizer_args": dataclasses.asdict(QNNDefaultConfig()),
    }

    # WHEN adding encodings with single-axis block quantization
    for name, encoding in quantization_logs.items():
        is_param_encoding = name in tensor_sets["parameters"]
        handler.add_encoding(name, encoding, is_param_encoding)

    encodings_dict = handler.build_encodings_dictionary()

    # THEN the constructed dictionary should match the expected V2 single-axis results
    _assert_dictionary_structure(expected_results, encodings_dict)


def test_v2_schema_perblock_3d_nested_array() -> None:
    """Test V2 schema with 3D tensor having block + per-channel + unit dimension."""
    # GIVEN a 3D tensor [8, 8, 8] with block quantization on tile size (4, 1, 8)
    handler = V2SchemaHandler()
    quantization_logs: dict[str, QuantParametersDict] = {
        "conv3d_weight": {
            "scale": torch.tensor([
                0.01,
                0.02,
                0.03,
                0.04,
                0.05,
                0.06,
                0.07,
                0.08,
                0.09,
                0.10,
                0.11,
                0.12,
                0.13,
                0.14,
                0.15,
                0.16,
            ]),
            "offset": torch.tensor([0.0] * 16),
            "num_bits": 8,
            "data_shape": [8, 8, 8],
            "tile_size": [4, 1, 8],
        }
    }

    expected_results = {
        "version": "2.0.0",
        "encodings": [
            {
                "name": "conv3d_weight",
                "output_dtype": "int8",
                "axis": 0,
                "block_size": 4,
                "y_scale": [
                    [[0.01], [0.02], [0.03], [0.04], [0.05], [0.06], [0.07], [0.08]],
                    [[0.09], [0.10], [0.11], [0.12], [0.13], [0.14], [0.15], [0.16]],
                ],
            }
        ],
        "quantizer_args": dataclasses.asdict(QNNDefaultConfig()),
    }

    # WHEN adding encodings
    for name, encoding in quantization_logs.items():
        handler.add_encoding(name, encoding, is_param=True)

    # THEN the result should have 3D nested structure
    encodings_dict = handler.build_encodings_dictionary()
    _assert_dictionary_structure(expected_results, encodings_dict)


def test_v2_schema_rejects_multi_block_sizes() -> None:
    """Test that V2 schema properly rejects multi-dimensional block sizes."""
    # GIVEN a V2 schema handler and encoding with multiple block sizes
    handler = V2SchemaHandler()

    encoding: QuantParametersDict = {
        "scale": torch.tensor([0.01, 0.02]),
        "offset": torch.tensor([0.0, 0.0]),
        "num_bits": 8,
        "data_shape": [4, 6],
        "tile_size": [2, 3],
    }

    # WHEN attempting to add encoding with multi-dimensional block sizes
    # THEN it should raise a ValueError about multi-dimensional block quantization not being supported
    with pytest.raises(ValueError):
        handler.add_encoding("test_tensor", encoding, is_param=True)


def _assert_dictionary_structure(dict1: dict[str, Any], dict2: dict[str, Any]) -> None:
    """Assert two structures have the same shape and types.

    Checks types, dict keys, and lengths recursively. Ignores actual values.
    """

    def _assert_structure_matches(entry1: Any, entry2: Any) -> None:
        entry1_type = type(entry1)
        entry2_type = type(entry2)

        assert entry1_type == entry2_type

        if isinstance(entry1, (str, bool, int, float)):
            return

        elif isinstance(entry1, dict):
            assert sorted(entry1.keys()) == sorted(entry2.keys())

            for key in entry1:
                v1, v2 = entry1[key], entry2[key]
                _assert_structure_matches(v1, v2)

        elif isinstance(entry1, (list, tuple)):
            assert len(entry1) == len(entry2)
            for e1, e2 in zip(entry1, entry2):
                _assert_structure_matches(e1, e2)

    _assert_structure_matches(dict1, dict2)


@pytest.mark.parametrize(
    "handler_class",
    [LegacySchemaHandler, V1SchemaHandler, V2SchemaHandler],
)
def test_schema_handler_clear(handler_class: type[EncodingSchemaHandler]) -> None:
    sample_encoding: QuantParametersDict = {
        "scale": torch.tensor([1.0]),
        "offset": torch.tensor([0]),
        "num_bits": 8,
        "data_shape": (1, 10),
        "tile_size": (1, 10),
    }

    handler = handler_class()

    handler.add_encoding("test_param", sample_encoding, is_param=True)
    handler.add_encoding("test_activation", sample_encoding, is_param=False)

    encodings_dict = handler.build_encodings_dictionary()
    assert len(encodings_dict) > 0

    handler.clear()
    cleared_dict = handler.build_encodings_dictionary()

    if isinstance(handler, V2SchemaHandler):
        assert cleared_dict["encodings"] == []
    elif isinstance(handler, V1SchemaHandler):
        assert cleared_dict["param_encodings"] == []
        assert cleared_dict["activation_encodings"] == []
    else:
        assert cleared_dict["param_encodings"] == {}
        assert cleared_dict["activation_encodings"] == {}


@pytest.mark.parametrize(
    "handler_class",
    [LegacySchemaHandler, V1SchemaHandler, V2SchemaHandler],
)
def test_schema_handler_missing_offset(handler_class: type[EncodingSchemaHandler]) -> None:
    """Test that schema handlers can process encodings with missing offset field."""
    # GIVEN a schema handler and encoding without offset
    handler = handler_class()

    encoding_without_offset: QuantParametersDict = {
        "scale": torch.tensor([0.01]),
        # "offset": intentionally missing
        "num_bits": 8,
        "data_shape": [16, 32],
        "tile_size": [16, 32],
    }

    # WHEN adding encoding without offset field
    handler.add_encoding("test_param", encoding_without_offset, is_param=True)

    encodings_dict = handler.build_encodings_dictionary()
    # THEN for Legacy Schema the offset is inferred
    if isinstance(handler, LegacySchemaHandler):
        assert encodings_dict["param_encodings"]["test_param"][0]["offset"]
    # THEN for V1 Schema the offset is inferred
    elif isinstance(handler, V1SchemaHandler):
        assert encodings_dict["param_encodings"][0]["offset"]
    # THEN for V2 Schema there is no offset field, as it is not required
    else:
        assert "offset" not in encodings_dict["encodings"][0]
