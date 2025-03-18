# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import pytest

from fastforward._quantops import operator as ops
from fastforward._quantops import spec_parser, symtypes


def test_parse_schema_parameters() -> None:
    # Extra or missing spaces in schema are intentional and part of the test
    schema = (
        "function_name    ("
        "param1: Quantized =   123,    "
        "param2: Optional[Quantized] =None, "
        "   param3: int | float,"
        "_param_4: float = 4.0,"
        "param5: Generic[int, float] "
        ")     ->     Optional[Tensor]"
    )
    operator = spec_parser.parse_schema(schema)

    expected_parameters = (
        ops.Parameter(symtypes.QuantizedTensor, "param1", "123"),
        ops.Parameter(symtypes.Optional[symtypes.QuantizedTensor], "param2", "None"),
        ops.Parameter(symtypes.Union[symtypes.Int, symtypes.Float], "param3", None),
        ops.Parameter(symtypes.Float, "_param_4", "4.0"),
        ops.Parameter(
            symtypes.GenericType("Generic", (symtypes.Int, symtypes.Float)),
            "param5",
            None,
        ),
    )

    assert operator.parameters == expected_parameters


def test_parse_schema_return_type() -> None:
    # Extra or missing spaces in schema are intentional and part of the test
    schema = (
        "function_name    ("
        "param1: Quantized =   123,    "
        "param2: Optional[Quantized] =None, "
        "   param3: int | float,"
        "_param_4: float = 4.0,"
        "param5: Generic[int, float] "
        ")     ->     Optional[Tensor]"
    )
    operator = spec_parser.parse_schema(schema)
    assert operator.return_type == symtypes.Optional[symtypes.Tensor]


def test_parse_schema_expected_failures() -> None:
    with pytest.raises(spec_parser.ParseError):
        spec_parser.parse_schema("function_name(Tensor param) Tensor?")
    with pytest.raises(spec_parser.ParseError):
        spec_parser.parse_schema("function_name(Tensor param Quantized other) -> Tensor?")
    with pytest.raises(spec_parser.ParseError):
        spec_parser.parse_schema("1func(Tensor param) -> Tensor?")
    with pytest.raises(spec_parser.ParseError):
        spec_parser.parse_schema("func() -> Tensor")
    with pytest.raises(spec_parser.ParseError):
        spec_parser.parse_schema("function_name(Tensor param == default) -> Tensor?")
    with pytest.raises(spec_parser.ParseError):
        spec_parser.parse_schema("function_name(int) -> Tensor?")
    with pytest.raises(spec_parser.ParseError):
        spec_parser.parse_schema("function_name(no_type) -> Tensor?")
