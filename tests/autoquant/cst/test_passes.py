import difflib
import textwrap

import libcst as libcst

from fastforward.autoquant.cst import passes

_STATEMENT_SUITE_TO_BLOCK_IN = """
x = 10; y = 20; z = x + y
if z > 25: print(f"z is greater than 25: {z}"); print("!")
else: print(f"z is not greater than 25: {z}")
"""

_STATEMENT_SUITE_TO_BLOCK_OUT = """
x = 10; y = 20; z = x + y
if z > 25:
    print(f"z is greater than 25: {z}")
    print("!")
else:
    print(f"z is not greater than 25: {z}")
"""


def test_statement_suite_to_indented_block() -> None:
    """Verifies simple statement suite is replaced by indented block."""
    # GIVEN code with non-simple statements, and its reference simplified version
    input, expected = map(
        textwrap.dedent, (_STATEMENT_SUITE_TO_BLOCK_IN, _STATEMENT_SUITE_TO_BLOCK_OUT)
    )

    # WHEN we visit the code with SimpleStatementSuiteToIndentedBlock
    transformer = passes.SimpleStatementSuiteToIndentedBlock()

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
    input = textwrap.dedent(_ASSIGNMENT_IN)

    # WHEN we wrap them into GeneralAssignments
    transformer = passes.WrapAssignments()

    # THEN the generated code is identical to the input
    assert_input_transforms_as_expected(input, transformer, input)


def assert_input_transforms_as_expected(
    input_module: str, transformer: libcst.CSTTransformer, output_module: str
) -> None:
    """Verifies the module transforms as expected."""
    module = libcst.parse_module(input_module)
    transformed = module.visit(transformer).code
    if not transformed == output_module:
        output = "\n".join(
            difflib.unified_diff(output_module.splitlines(), transformed.splitlines())
        )
        raise RuntimeError(f"Transformed module does not match expected output:\n{output}.")
