# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from collections.abc import Sequence

import libcst as libcst

from fastforward.autoquant.cst import passes

from tests.utils.string import assert_strings_match_verbose, dedent_strip

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
