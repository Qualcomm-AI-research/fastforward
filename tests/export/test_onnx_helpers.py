# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from unittest.mock import Mock

import onnxscript

from fastforward.export._onnx_helpers import (
    _fix_encoding_names,
    _fix_onnx_names,
    _fix_reshape_allowzero,
)


def test_fix_reshape_allowzero_removes_attribute() -> None:
    # GIVEN a model with Reshape nodes containing allowzero attribute
    model = Mock()
    reshape_node = Mock()
    reshape_node.op_type = "Reshape"
    reshape_node.attributes = {"allowzero": 1, "other_attr": "keep"}

    model.graph.__len__ = Mock(return_value=1)
    model.graph.node.return_value = reshape_node

    # WHEN _fix_reshape_allowzero is called
    _fix_reshape_allowzero(model)

    # THEN allowzero attribute should be removed from Reshape nodes only"""
    assert "allowzero" not in reshape_node.attributes
    assert "other_attr" in reshape_node.attributes


def test_fix_encoding_names_adds_prefix() -> None:
    # GIVEN an encodings dictionary with tensor names
    encodings = {
        "input": {"scale": 0.1, "offset": 128},
        "conv1.weight": {"scale": 0.05, "offset": 0},
    }
    prefix = "ff_test123"

    # WHEN _fix_encoding_names is called with a prefix
    result = _fix_encoding_names(encodings, prefix)

    # THEN all keys should have the prefix and _0 suffix"""
    expected_keys = {"ff_test123_input_0", "ff_test123_conv1.weight_0"}
    assert set(result.keys()) == expected_keys
    assert result["ff_test123_input_0"] == encodings["input"]
    assert result["ff_test123_conv1.weight_0"] == encodings["conv1.weight"]


def _create_named_mock(name: str) -> Mock:
    mock = Mock()
    mock.name = name
    return mock


def test_fix_onnx_names() -> None:
    # GIVEN a mock onnxscript model
    model = Mock(spec=onnxscript.ir.Model)

    node = _create_named_mock("conv1")
    node.outputs = [_create_named_mock("conv1_output"), _create_named_mock("relu1_output")]
    node.inputs = [_create_named_mock("input_tensor"), _create_named_mock("conv1.weight")]

    model.graph = Mock(
        initializers=Mock(
            values=Mock(
                return_value=[_create_named_mock("conv1.weight"), _create_named_mock("conv1.bias")]
            )
        ),
        _nodes=[node],
        inputs=[_create_named_mock("input_tensor")],
        outputs=[_create_named_mock("relu1_output")],
    )

    # WHEN altering the initializers/inputs/outputs/node names
    _fix_onnx_names(model, "ff_test123")

    # THEN the appropriate new names should be applied
    assert model.graph.initializers.values()[0].name == "ff_test123_conv1.weight_0"
    assert model.graph._nodes[0].name == "ff_test123_conv1_0"
    assert model.graph._nodes[0].outputs[0].name == "ff_test123_conv1_output_0"
    assert model.graph._nodes[0].outputs[1].name == "ff_test123_relu1_output_0"
    assert model.graph.outputs[0].name == "ff_test123_relu1_output_0"
    # THEN the input names should remain unchanged
    assert model.graph.inputs[0].name == "input_tensor"
