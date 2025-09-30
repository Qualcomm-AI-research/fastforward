# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Any

import pytest
import torch

from fastforward.export._export_schemas import (
    EncodingSchemaHandler,
    LegacySchemaHandler,
    QuantParametersDict,
    V1SchemaHandler,
    V2SchemaHandler,
)


@pytest.fixture
def mock_basic_quantization_logs() -> dict[str, dict[str, Any]]:
    """Basic quantization logs without PerBlock (compatible with all schemas)."""
    return {
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


@pytest.fixture
def mock_perblock_1d_logs() -> dict[str, dict[str, Any]]:
    """PerBlock 1D quantization logs (V1 and V2 compatible)."""
    return {
        "linear.bias": {
            "scale": torch.tensor([0.021, 0.022, 0.023, 0.024]),  # 4 blocks
            "offset": torch.tensor([128.0, 120.0, 135.0, 125.0]),
            "num_bits": 8,
            "block_size": [2],
            "data_shape": [8],
            "tile_size": [2],
        },
    }


@pytest.fixture
def mock_perblock_2d_logs() -> dict[str, dict[str, Any]]:
    """PerBlock 2D quantization logs (V2 only)."""
    return {
        "conv2d.weight": {
            "scale": torch.tensor([0.01, 0.02, 0.03, 0.04, 0.05, 0.06]),
            "offset": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "num_bits": 8,
            "block_size": [2, 3],
            "data_shape": [6, 6],
            "tile_size": [2, 3],
        },
    }


LEGACY_EXPECTED_RESULTS = {
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
}

V1_EXPECTED_RESULTS = {
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
        },
        {
            "name": "linear.bias",
            "dtype": "INT",
            "enc_type": "PER_BLOCK",
            "is_sym": False,
            "bw": 8,
            "block_size": 2,
            "scale": [0.021, 0.022, 0.023, 0.024],  # Flat list (1D PerBlock)
            "offset": [128.0, 120.0, 135.0, 125.0],  # Flat list (1D PerBlock)
        },
        {
            "name": "conv2d.weight",
            "dtype": "INT",
            "enc_type": "PER_BLOCK",
            "is_sym": True,
            "bw": 8,
            "block_size": 2,  # V1.0.0 uses single integer
            "scale": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],  # Flat list
            "offset": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
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
}

V2_EXPECTED_RESULTS = {
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
        {
            "name": "linear.bias",
            "output_dtype": "int8",
            "y_scale": [0.021, 0.022, 0.023, 0.024],  # 1D - stays flat
            "y_zero_point": [0.0, -8.0, 7.0, -3.0],
            "axis": 0,
            "block_size": 2,
        },
        {
            "name": "conv2d.weight",
            "output_dtype": "int8",
            "y_scale": [[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]],  # 2D - nested!
            "axis": [0, 1],  # V2.0.0 supports multi-axis
            "block_size": [2, 3],
        },
    ],
}

BASIC_V1_EXPECTED_RESULTS = {
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
}

BASIC_V2_EXPECTED_RESULTS = {
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
}

V1_PERBLOCK_1D_EXPECTED = {
    "version": "1.0.0",
    "param_encodings": [
        {
            "name": "linear.bias",
            "dtype": "INT",
            "enc_type": "PER_BLOCK",
            "is_sym": False,
            "bw": 8,
            "block_size": 2,
            "scale": [0.021, 0.022, 0.023, 0.024],
            "offset": [128.0, 120.0, 135.0, 125.0],
        }
    ],
    "activation_encodings": [],
}

V2_PERBLOCK_1D_EXPECTED = {
    "version": "2.0.0",
    "encodings": [
        {
            "name": "linear.bias",
            "output_dtype": "uint8",
            "y_scale": [0.021, 0.022, 0.023, 0.024],  # Flat list
            "y_zero_point": [0.0, -8.0, 7.0, -3.0],
            "axis": 0,
            "block_size": 2,
        }
    ],
}

V2_PERBLOCK_2D_EXPECTED = {
    "version": "2.0.0",
    "encodings": [
        {
            "name": "conv2d.weight",
            "output_dtype": "int8",
            "y_scale": [[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]],  # Nested structure
            "axis": [0, 1],
            "block_size": [2, 3],
        }
    ],
}


@pytest.mark.parametrize(
    "handler_class,expected_result",
    [
        (LegacySchemaHandler, LEGACY_EXPECTED_RESULTS),
        (V1SchemaHandler, BASIC_V1_EXPECTED_RESULTS),
        (V2SchemaHandler, BASIC_V2_EXPECTED_RESULTS),
    ],
)
def test_schema_handler_basic(
    handler_class: type[EncodingSchemaHandler],
    expected_result: dict[str, Any],
    mock_basic_quantization_logs: dict[str, QuantParametersDict],
) -> None:
    """Test basic schema handling without PerBlock quantization."""
    # GIVEN a schema handler and some dummy inputs
    handler = handler_class()
    tensor_sets = {
        "inputs": {"input"},
        "activations": {"layer1_output"},
        "parameters": {"conv1.weight"},
    }

    # WHEN building the encodings dictionary
    for name, encoding in mock_basic_quantization_logs.items():
        is_param_encoding = name in tensor_sets["parameters"]
        handler.add_encoding(name, encoding, is_param_encoding)

    encodings_dict = handler.build_encodings_dictionary()

    # THEN the constructed dictionary should match the expected results.
    _assert_dictionary_structure(expected_result, encodings_dict)


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
    mock_perblock_1d_logs: dict[str, QuantParametersDict],
) -> None:
    """Test PerBlock 1D quantization handling."""
    # GIVEN a schema handler and some dummy inputs
    handler = handler_class()
    tensor_sets = {"inputs": set(), "activations": set(), "parameters": {"linear.bias"}}

    # WHEN building the dictionary
    # THEN the legacy schema should raise an error as it does not support Per Block quantization.
    if should_raise:
        with pytest.raises(should_raise):
            for name, encoding in mock_perblock_1d_logs.items():
                is_param_encoding = name in tensor_sets["parameters"]
                handler.add_encoding(name, encoding, is_param_encoding)

    # WHEN building the dictionary
    else:
        for name, encoding in mock_perblock_1d_logs.items():
            is_param_encoding = name in tensor_sets["parameters"]
            handler.add_encoding(name, encoding, is_param_encoding)

        encodings_dict = handler.build_encodings_dictionary()
        assert expected_result is not None
        # THEN the constructed dictionary should match the expected results
        _assert_dictionary_structure(expected_result, encodings_dict)


@pytest.mark.parametrize(
    "handler_class,expected_result,should_raise",
    [
        (LegacySchemaHandler, None, ValueError),
        (V1SchemaHandler, None, ValueError),
        (V2SchemaHandler, V2_PERBLOCK_2D_EXPECTED, None),
    ],
)
def test_schema_handler_perblock_2d(
    handler_class: type[EncodingSchemaHandler],
    expected_result: dict[str, Any] | None,
    should_raise: type[Exception] | None,
    mock_perblock_2d_logs: dict[str, QuantParametersDict],
) -> None:
    """Test PerBlock 2D quantization handling."""
    # GIVEN a schema handler and some dummy inputs
    handler = handler_class()
    tensor_sets = {"inputs": set(), "activations": set(), "parameters": {"conv2d.weight"}}

    # WHEN building the dictionary
    # THEN the legacy schema should raise an error as it does not support Per Block quantization.
    # THEN the V1 shcema should raise an error as it does not support 2D block quantization.
    if should_raise:
        with pytest.raises(should_raise):
            for name, encoding in mock_perblock_2d_logs.items():
                is_param_encoding = name in tensor_sets["parameters"]
                handler.add_encoding(name, encoding, is_param_encoding)
    else:
        # WHEN building the dictionary
        for name, encoding in mock_perblock_2d_logs.items():
            is_param_encoding = name in tensor_sets["parameters"]
            handler.add_encoding(name, encoding, is_param_encoding)

        encodings_dict = handler.build_encodings_dictionary()
        assert expected_result is not None
        # THEN the constructed dictionary should match the expected results
        _assert_dictionary_structure(expected_result, encodings_dict)


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
