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
def mock_quantization_logs() -> dict[str, dict[str, Any]]:
    """Mock quantization logs in the raw format before schema processing."""
    return {
        "input": {
            "scale": torch.tensor([0.031]),
            "offset": torch.tensor([128.0]),
            "num_bits": 8,
        },
        "conv1.weight": {
            "scale": torch.tensor([0.012, 0.015, 0.011]),  # Per-channel
            "offset": torch.tensor([0.0, 0.0, 0.0]),
            "num_bits": 8,
        },
        "layer1_output": {
            "scale": torch.tensor([0.025]),
            "offset": torch.tensor([0.0]),  # Symmetric
            "num_bits": 8,
        },
    }


@pytest.fixture
def mock_tensor_sets() -> dict[str, set[str]]:
    """Mock tensor categorization."""
    return {
        "inputs": {"input"},
        "activations": {"layer1_output"},
        "parameters": {"conv1.weight"},
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
        },
        {"name": "layer1_output", "output_dtype": "int8", "y_scale": 0.02500000037252903},
    ],
}


@pytest.mark.parametrize(
    "handler_class,expected_result",
    [
        (LegacySchemaHandler, LEGACY_EXPECTED_RESULTS),
        (V1SchemaHandler, V1_EXPECTED_RESULTS),
        (V2SchemaHandler, V2_EXPECTED_RESULTS),
    ],
)
def test_schema_handler(
    handler_class: type[EncodingSchemaHandler],
    expected_result: dict[str, Any],
    mock_quantization_logs: dict[str, QuantParametersDict],
    mock_tensor_sets: dict[str, set[str]],
) -> None:
    # GIVEN a schema handler
    handler = handler_class()

    # WHEN adding encodings and building the dictionary
    for name, encoding in mock_quantization_logs.items():
        is_param_encoding = name in mock_tensor_sets["parameters"]
        handler.add_encoding(name, encoding, is_param_encoding)

    encodings_dict = handler.build_encodings_dictionary()

    # THEN the structure should match the expectation
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
