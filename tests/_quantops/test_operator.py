# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import pytest

from fastforward._quantops import symtypes
from fastforward._quantops.operator import Operator


def test_from_spec_simple_function() -> None:
    """Test parsing a simple function specification."""
    # GIVEN a simple function spec with two quantized tensor parameters and return type
    spec = "add(x: QuantizedTensor, y: QuantizedTensor) -> QuantizedTensor"
    fallback = "torch.add"

    # WHEN parsing the spec into an Operator
    op = Operator.from_spec(spec, fallback)

    # THEN the operator should have correct identifier, parameters, and return type
    assert op.identifier == "add"
    assert len(op.parameters) == 2
    assert op.parameters[0].name == "x"
    assert op.parameters[0].param_type == symtypes.QuantizedTensor
    assert op.parameters[1].name == "y"
    assert op.parameters[1].param_type == symtypes.QuantizedTensor
    assert op.return_type == symtypes.QuantizedTensor
    assert op.metadata is not None
    assert op.metadata.fallback == fallback


def test_from_spec_function_with_defaults() -> None:
    """Test parsing function with default parameter values."""
    # GIVEN a function spec with parameters that have default values
    spec = "conv2d(input: QuantizedTensor, weight: QuantizedTensor, bias: Tensor = None, stride: int = 1)"
    fallback = "torch.nn.functional.conv2d"

    # WHEN parsing the spec into an Operator
    op = Operator.from_spec(spec, fallback)

    # THEN the operator should correctly capture default values for parameters
    assert op.identifier == "conv2d"
    assert len(op.parameters) == 4
    assert op.parameters[0].name == "input"
    assert op.parameters[0].default_value is None
    assert op.parameters[2].name == "bias"
    assert op.parameters[2].default_value == "None"
    assert op.parameters[3].name == "stride"
    assert op.parameters[3].default_value == "1"


def test_from_spec_no_return_type() -> None:
    """Test parsing function without return type annotation."""
    # GIVEN a function spec without a return type annotation
    spec = "relu_(x: QuantizedTensor)"
    fallback = "torch.relu_"

    # WHEN parsing the spec into an Operator
    op = Operator.from_spec(spec, fallback)

    # THEN the operator should have None as return type
    assert op.identifier == "relu_"
    assert len(op.parameters) == 1
    assert op.return_type is None
    assert op.metadata is not None
    assert op.metadata.fallback == fallback


def test_from_spec_with_metadata() -> None:
    """Test parsing function with additional metadata."""
    # GIVEN a function spec and additional metadata parameters
    spec = "matmul(a: QuantizedTensor, b: QuantizedTensor) -> QuantizedTensor"
    fallback = "torch.matmul"
    cast_output = "float32"
    line_number = 42

    # WHEN parsing the spec with extra metadata
    op = Operator.from_spec(spec, fallback, cast_output=cast_output, line_number=line_number)

    # THEN the operator metadata should include the additional parameters
    assert op.identifier == "matmul"
    assert op.metadata is not None
    assert op.metadata.fallback == fallback
    assert op.metadata.cast_output == cast_output
    assert op.metadata.line_number == line_number


def test_from_spec_string_default_value() -> None:
    """Test parsing function with string default values."""
    # GIVEN a function spec with string and numeric default values
    spec = (
        'pad(input: QuantizedTensor, pad: tuple[int], mode: str = "constant", value: float = 0.0)'
    )
    fallback = "torch.nn.functional.pad"

    # WHEN parsing the spec into an Operator
    op = Operator.from_spec(spec, fallback)

    # THEN the operator should correctly parse and store string default values
    assert op.identifier == "pad"
    assert len(op.parameters) == 4
    assert op.parameters[2].name == "mode"
    assert op.parameters[2].default_value == "constant"
    assert op.parameters[3].name == "value"
    assert op.parameters[3].default_value == "0.0"


def test_from_spec_mixed_parameter_types() -> None:
    """Test parsing function with different parameter types (positional, keyword-only)."""
    # GIVEN a function spec with both positional and keyword-only parameters
    spec = "complex_op(x: QuantizedTensor, y: Tensor, *, alpha: float = 1.0, beta: int = 2)"
    fallback = "torch.complex_op"

    # WHEN parsing the spec into an Operator
    op = Operator.from_spec(spec, fallback)

    # THEN the operator should include all parameter types in the parameters list
    assert op.identifier == "complex_op"
    assert len(op.parameters) == 4
    param_names = [p.name for p in op.parameters]
    assert "x" in param_names
    assert "y" in param_names
    assert "alpha" in param_names
    assert "beta" in param_names


def test_from_spec_invalid_syntax() -> None:
    """Test that invalid function syntax raises ValueError."""
    # GIVEN an invalid function specification string
    spec = "invalid syntax here"
    fallback = "torch.something"

    # WHEN attempting to parse the invalid spec
    # THEN a ValueError should be raised with appropriate message
    with pytest.raises(ValueError, match="is not a valid operator spec"):
        Operator.from_spec(spec, fallback)


def test_from_spec_missing_parameter_annotation() -> None:
    """Test that missing parameter annotations raise ValueError."""
    # GIVEN a function spec where one parameter lacks type annotation
    spec = "bad_func(x: QuantizedTensor, y)"  # y missing annotation
    fallback = "torch.bad_func"

    # WHEN attempting to parse the spec with missing annotation
    # THEN a ValueError should be raised requiring all parameters to have annotations
    with pytest.raises(ValueError, match="All parameters must have a valid annotation"):
        Operator.from_spec(spec, fallback)


def test_from_spec_empty_parameters() -> None:
    """Test parsing function with no parameters."""
    # GIVEN a function spec with no parameters but with return type
    spec = "get_device() -> str"
    fallback = "torch.cuda.current_device"

    # WHEN parsing the spec into an Operator
    op = Operator.from_spec(spec, fallback)

    # THEN the operator should have empty parameters list but valid return type
    assert op.identifier == "get_device"
    assert len(op.parameters) == 0
    assert op.return_type == symtypes.String
    assert op.metadata is not None
    assert op.metadata.fallback == fallback
