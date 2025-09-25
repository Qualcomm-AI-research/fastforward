# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#
import pathlib
import tempfile

from unittest.mock import Mock

import libcst

from fastforward._quantops import operator, symtypes
from fastforward._quantops.generate import (
    _ModuleGenerator,
    _parameter,
    _ParameterList,
    _simple_if,
    generate,
)
from libcst import matchers as m


class MockWriter:
    """Mock writer for testing."""

    def __init__(self) -> None:
        self.written_data: list[str] = []

    def write(self, data: str) -> None:
        self.written_data.append(data)


def test_module_generator_append_import() -> None:
    """Test adding imports to module generator."""
    # GIVEN: A module generator
    gen = _ModuleGenerator(pathlib.Path("test.py"))

    # WHEN: Adding imports
    gen.append_import("torch")
    gen.append_import("Tensor", "torch")

    # THEN: The imports should appear in the generated code
    code = gen.code
    assert "import torch" in code
    assert "from torch import Tensor" in code


def test_module_generator_append_raw() -> None:
    """Test adding raw source code."""
    # GIVEN: A module generator
    gen = _ModuleGenerator(pathlib.Path("test.py"))

    # WHEN: Adding raw source code
    gen.append_raw("x = 42")

    # THEN: The raw code should appear in the generated module
    assert "x = 42" in gen.code


def test_module_generator_append_op() -> None:
    """Test adding operator function."""
    # GIVEN: A module generator and a function definition
    gen = _ModuleGenerator(pathlib.Path("test.py"))
    func_def = libcst.parse_statement("def test_op(): pass")
    assert isinstance(func_def, libcst.FunctionDef)

    # WHEN: Adding the operator function
    gen.append_op(func_def)

    # THEN: The function should appear in code
    code = gen.code
    assert "def test_op():" in code

    # THEN: The function should appear in the __all__ list
    # Find __all__ assignment using matchers
    tree = libcst.parse_module(code)
    all_assignment = m.extractall(
        tree,
        m.SimpleStatementLine(
            body=[
                m.Assign(
                    targets=[m.AssignTarget(target=m.Name("__all__"))],
                    value=m.SaveMatchedNode(m.List(), "all_list"),
                )
            ]
        ),
    )

    assert len(all_assignment) == 1, "__all__ assignment not found"
    all_list = all_assignment[0]["all_list"]
    assert isinstance(all_list, libcst.List)

    all_items = [
        elem.value.value.strip("\"'")
        for elem in all_list.elements
        if isinstance(elem.value, libcst.SimpleString)
    ]
    assert "test_op" in all_items


def test_parameter_list_single_param() -> None:
    """Test single parameter."""
    # GIVEN: A parameter list with one parameter
    params = _ParameterList()
    params.append("x", "value")

    # WHEN: Converting to string
    result = str(params)

    # THEN: Should return formatted parameter
    assert result == "x=value"


def test_parameter_list_multiple_params() -> None:
    """Test multiple parameters."""
    # GIVEN: A parameter list with multiple parameters
    params = _ParameterList()
    params.append("x", "1")
    params.append("y", "2")

    # WHEN: Converting to string
    result = str(params)

    # THEN: Should return comma-separated parameters
    assert result == "x=1, y=2"


def test_parameter_with_default() -> None:
    """Test parameter creation with default value."""
    # GIVEN: An operator parameter with default value
    param = operator.Parameter(symtypes.Int, "count", "10")

    # WHEN: Converting to LibCST parameter
    result = _parameter(param)

    # THEN: Should have name, annotation, and default value
    assert result.name.value == "count"
    assert result.default is not None
    assert m.matches(result.default, m.Integer(value="10"))


def test_parameter_without_default() -> None:
    """Test parameter creation without default value."""
    # GIVEN: An operator parameter without default value
    param = operator.Parameter(symtypes.Int, "value", None)

    # WHEN: Converting to LibCST parameter
    result = _parameter(param)

    # THEN: Should have name and annotation but no default
    assert result.name.value == "value"
    assert result.default is None


def test_simple_if_basic() -> None:
    """Test simple if statement creation."""
    # GIVEN: A condition and body statement
    condition = "x > 0"
    body = "return x"

    # WHEN: Creating a simple if statement
    if_stmt = _simple_if(condition, body)

    # THEN: Should create proper if statement structure
    assert isinstance(if_stmt, libcst.If)
    assert isinstance(if_stmt.body, libcst.IndentedBlock)
    assert if_stmt.orelse is None


def test_simple_if_with_else() -> None:
    """Test simple if statement with else clause."""
    # GIVEN: A condition, body, and else statement
    condition = "x > 0"
    body = "return x"
    else_stmt = "return 0"

    # WHEN: Creating a simple if statement with else
    if_stmt = _simple_if(condition, body, else_stmt)

    # THEN: Should create if statement with else clause
    assert isinstance(if_stmt, libcst.If)
    assert isinstance(if_stmt.orelse, libcst.Else)


def test_generate_creates_files() -> None:
    """Test that generate function creates both files."""
    # GIVEN: An empty operator table and temporary directory
    operators = Mock()
    operators.operators.return_value = []

    with tempfile.TemporaryDirectory() as temp_dir:
        source = pathlib.Path("test_source.py")
        destination = pathlib.Path(temp_dir)

        # WHEN: Calling generate function
        generate(operators, source, destination)

        # THEN: Both fallback.py and operators.py should be created
        assert (destination / "fallback.py").exists()
        assert (destination / "operators.py").exists()


def test_generate_includes_copyright() -> None:
    """Test that generated files include copyright header."""
    # GIVEN: An empty operator table and temporary directory
    operators = Mock()
    operators.operators.return_value = []

    with tempfile.TemporaryDirectory() as temp_dir:
        source = pathlib.Path("test_source.py")
        destination = pathlib.Path(temp_dir)

        # WHEN: Calling generate function
        generate(operators, source, destination)

        # THEN: Generated files should start with copyright header
        fallback_content = (destination / "fallback.py").read_text()
        operators_content = (destination / "operators.py").read_text()

        assert fallback_content.startswith("# Copyright (c) Qualcomm Technologies")
        assert operators_content.startswith("# Copyright (c) Qualcomm Technologies")
