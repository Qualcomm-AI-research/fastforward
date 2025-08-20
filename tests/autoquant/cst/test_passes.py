# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from collections.abc import Sequence

import libcst as libcst
import pytest

from fastforward._autoquant.autoquant import codeformat_with_defaults
from fastforward._autoquant.cst import nodes, passes
from fastforward._autoquant.cst.filter import filter_nodes_by_type
from fastforward.testing.string import assert_strings_match_verbose, dedent_strip

_STATEMENT_SUITE_TO_BLOCK_IN = """
x = 10; y = 20; z = x + y
if z > 25: print(f"z is greater than 25: {z}"); print("!")
else: print(f"z is not greater than 25: {z}")

def some_function():
    x = 10; y = 11;
    return x + y
"""

_STATEMENT_SUITE_TO_BLOCK_OUT = """
x = 10
y = 20
z = x + y
if z > 25:
    print(f"z is greater than 25: {z}")
    print("!")
else:
    print(f"z is not greater than 25: {z}")

def some_function():
    x = 10
    y = 11
    return x + y
"""


def test_statement_suite_to_indented_block() -> None:
    """Verifies simple statement suite is replaced by indented block."""
    # GIVEN code with non-simple statements, and its reference simplified version
    input, expected = dedent_strip(_STATEMENT_SUITE_TO_BLOCK_IN, _STATEMENT_SUITE_TO_BLOCK_OUT)

    # WHEN we visit the code with ConvertSemicolonJoinedStatements
    transformer = passes.ConvertSemicolonJoinedStatements()

    # THEN the input code transforms as expected
    assert_input_transforms_as_expected(input, transformer, expected)


_ASSIGNMENT_IN = """
a: int
x: int = 10
x += 20
y = z = x
"""


def test_mark_assignment() -> None:
    """Verifies GeneralAssignment does not interfere with codegen."""
    # GIVEN different types of assignments
    input = dedent_strip(_ASSIGNMENT_IN)[0]

    # WHEN we wrap them into GeneralAssignments
    transformer = passes.WrapAssignments()

    # THEN the generated code is identical to the input
    assert_input_transforms_as_expected(input, transformer, input)


_ISOLATE_REPLACEMENT_CANDIDATES_IN = """
def some_function(a, b, c, d):
    v = a + b * c
    return (v + d) / b

# ensure that isolation works at the module level
r = a + b * c
"""

_ISOLATE_REPLACEMENT_CANDIDATES_OUT = """
def some_function(a, b, c, d):
    _tmp_1 = b * c
    v = a + _tmp_1
    _tmp_2 = (v + d)
    return _tmp_2 / b

# ensure that isolation works at the module level
_tmp_3 = b * c
r = a + _tmp_3
"""


def test_isolate_replacement_candidates() -> None:
    # GIVEN statements with compound expressions
    input, expected = dedent_strip(
        _ISOLATE_REPLACEMENT_CANDIDATES_IN, _ISOLATE_REPLACEMENT_CANDIDATES_OUT
    )

    # WHEN we isolate candidates after wrapping assignments
    transformer_mark_candidates = passes.MarkReplacementCandidates()
    transformer_isolate_candidates = passes.IsolateReplacementCandidates()

    # THEN the input code is transformed such that each statement contains one
    # expression at most
    assert_input_transforms_as_expected(
        input,
        [
            transformer_mark_candidates,
            transformer_isolate_candidates,
        ],
        expected,
    )


_EXTENDED_MARK_REPLACEMENT_CANDIDATES_IN = """
import torch
from typing import TypeVar

_T = TypeVar("_T", torch.Tensor)

def do_something(a_: _T, b_: _T) -> _T:
    return a_ + b_

def some_function(a: list[int]):
    b = [1,2,3]
    c = [4, 5, 6]
    d = a + b
    e = b + c
    f = do_something(d, e)
    
    g = 5
    h = -g
    
    a_tensor = torch.tensor([1,2,3])
    b_tensor = torch.tensor([4,5,6])
    c_tensor = a_tensor + b_tensor
    d_tensor = do_something(b_tensor, c_tensor)
    e_tensor = -a_tensor
    
    return d, e, c_tensor
"""


@pytest.mark.slow
def test_extended_mark_replacement_candidates() -> None:
    # GIVEN: A code snippet that contains tensor operations
    (input,) = dedent_strip(_EXTENDED_MARK_REPLACEMENT_CANDIDATES_IN)
    cst = libcst.parse_module(input)
    wrapped_cst = libcst.MetadataWrapper(cst, unsafe_skip_copy=True)

    # WHEN: We apply the ExtendedMarkReplacementCandidates transformer
    transformer_mark_candidates = passes.ExtendedMarkReplacementCandidates()
    cst = wrapped_cst.visit(transformer_mark_candidates)

    # THEN: The transformer should identify the expected tensor operations as
    # replacement candidates and not the non-tensor operations.
    expected_replacement_candidates = {
        "a_tensor + b_tensor",
        "do_something(b_tensor, c_tensor)",
        "-a_tensor",
    }

    codegen_module = libcst.Module([])
    for funcdef in filter_nodes_by_type(cst, libcst.FunctionDef):
        if funcdef.name.value != "some_function":
            continue

        for node in filter_nodes_by_type(funcdef, nodes.ReplacementCandidate):
            code_str = codegen_module.code_for_node(node)
            assert code_str in expected_replacement_candidates
            expected_replacement_candidates.remove(code_str)

    assert len(expected_replacement_candidates) == 0, "expected to be empty"


def assert_input_transforms_as_expected(
    input_module: str,
    transformers: libcst.CSTTransformer | Sequence[libcst.CSTTransformer],
    output_module: str,
) -> None:
    """Verifies the module transforms as expected."""
    module = libcst.parse_module(input_module)

    if isinstance(transformers, libcst.CSTTransformer):
        transformers = [transformers]
    for transformer in transformers:
        module = module.visit(transformer)

    transformed = module.code
    assert_strings_match_verbose(transformed, output_module)


FOLD_TMPS_EXAMPLE_1, FOLD_TMPS_EXPECTED_1 = (
    """
    def test_function():
        tmp_1 = 42
        x = tmp_1
        return x
    """,
    """
    def test_function():
        x = 42
        return x
    """,
)

FOLD_TMPS_EXAMPLE_2, FOLD_TMPS_EXPECTED_2 = (
    """
    def test_function():
        tmp_1 = 42
        x = tmp_1
        y = tmp_1  # Multiple accesses should prevent folding
        return x + y
    """,
    """
    def test_function():
        tmp_1 = 42
        x = tmp_1
        y = tmp_1  # Multiple accesses should prevent folding
        return x + y
    """,
)

FOLD_TMPS_EXAMPLE_3, FOLD_TMPS_EXPECTED_3 = (
    """
    def test_function():
        tmp_1 = 42
        tmp_1 = 43  # Multiple assignments should prevent folding
        x = tmp_1
        return x
    """,
    """
    def test_function():
        tmp_1 = 42
        tmp_1 = 43  # Multiple assignments should prevent folding
        x = tmp_1
        return x
    """,
)

FOLD_TMPS_EXAMPLE_4, FOLD_TMPS_EXPECTED_4 = (
    """
    def test_function():
        a[0] = 42  # Not a simple name target
        x = a[0]
        return x
    """,
    """
    def test_function():
        a[0] = 42  # Not a simple name target
        x = a[0]
        return x
    """,
)

FOLD_TMPS_EXAMPLE_5, FOLD_TMPS_EXPECTED_5 = (
    """
    def test_function():
        regular_var = 42  # Not a tmp_X variable
        x = regular_var
        return x
    """,
    """
    def test_function():
        regular_var = 42  # Not a tmp_X variable
        x = regular_var
        return x
    """,
)

FOLD_TMPS_EXAMPLE_6, FOLD_TMPS_EXPECTED_6 = (
    """
    def test_function():
        tmp_1 = 40 + 2
        x = tmp_1 * 3
        return x
    """,
    """
    def test_function():
        x = (40 + 2) * 3
        return x
    """,
)

FOLD_TMPS_EXAMPLE_7, FOLD_TMPS_EXPECTED_7 = (
    """
    def outer_function():
        tmp_1 = 42

        def inner_function():
            tmp_2 = 43
            y = tmp_2
            return y

        x = tmp_1
        return x + inner_function()
    """,
    """
    def outer_function():

        def inner_function():
            y = 43
            return y

        x = 42
        return x + inner_function()
    """,
)


@pytest.mark.parametrize(
    "code,expected",
    [
        (FOLD_TMPS_EXAMPLE_1, FOLD_TMPS_EXPECTED_1),
        (FOLD_TMPS_EXAMPLE_2, FOLD_TMPS_EXPECTED_2),
        (FOLD_TMPS_EXAMPLE_3, FOLD_TMPS_EXPECTED_3),
        (FOLD_TMPS_EXAMPLE_4, FOLD_TMPS_EXPECTED_4),
        (FOLD_TMPS_EXAMPLE_5, FOLD_TMPS_EXPECTED_5),
        (FOLD_TMPS_EXAMPLE_6, FOLD_TMPS_EXPECTED_6),
        (FOLD_TMPS_EXAMPLE_7, FOLD_TMPS_EXPECTED_7),
    ],
    ids=[f"case-{i}" for i in range(1, 8)],
)
def test_fold_temporaries(code: str, expected: str) -> None:
    # GIVEN a python function
    module = libcst.parse_module(dedent_strip(code)[0])
    module = module.visit(passes.WrapAssignments())

    # WHEN the transformer is applied
    wrapper = libcst.MetadataWrapper(module)
    transformed = wrapper.visit(passes.FoldSimpleTemporaries())

    # THEN the temporary variable should be folded into its usage
    actual = codeformat_with_defaults(transformed.code)
    expected = codeformat_with_defaults(dedent_strip(expected)[0])
    assert_strings_match_verbose(actual, expected)
