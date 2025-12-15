# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import libcst
import pytest

from fastforward._autoquant.cst.pattern import PatternRule, _PatternRuleTransformer


def test_simple_replacement() -> None:
    """Test basic pattern matching and replacement."""
    # GIVEN: A rule that swaps operands in addition
    rule = PatternRule.from_str("ant + bat", "cat * dog")
    code = libcst.parse_expression("ant + bat")

    # WHEN: The rule is applied to the code
    result = rule.apply(code)

    # THEN: The operands should be swapped
    assert _as_expr_module(result).code == "cat * dog\n"


def test_capture_and_replace() -> None:
    """Test capturing variables and using them in replacement."""
    # GIVEN: A rule that captures two operands and swaps them with different operator
    rule = PatternRule.from_str("{x} + {y}", "{y} * {x}")
    code = libcst.parse_expression("ant + bat")

    # WHEN: The rule is applied
    result = rule.apply(code)

    # THEN: The captured variables should be used in the replacement
    assert _as_expr_module(result).code == "bat * ant\n"


def test_no_match_returns_original() -> None:
    """Test that non-matching patterns return the original node."""
    # GIVEN: A rule that matches addition but code has multiplication
    rule = PatternRule.from_str("x + y", "y + x")
    code = libcst.parse_expression("a * b")

    # WHEN: The rule is applied
    result = rule.apply(code)

    # THEN: The original code should be returned unchanged
    assert result == code


def test_function_call_pattern() -> None:
    """Test matching and replacing function calls."""
    # GIVEN: A rule that transforms function calls to method calls
    rule = PatternRule.from_str("{func}({arg})", "{func}.new({arg})")
    code = libcst.parse_expression("foo(bar)")

    # WHEN: The rule is applied
    result = rule.apply(code)

    # THEN: The function call should be transformed to a method call
    assert _as_expr_module(result).code == "foo.new(bar)\n"


def test_statement_to_expression_wrapping() -> None:
    """Test that expressions are wrapped in statements when needed."""
    # GIVEN: A rule applied to a statement node
    rule = PatternRule.from_str("{x} + {y}", "{x} * {y}")
    code = libcst.parse_statement("a + b")

    # WHEN: The rule is applied
    result = rule.apply(code, recursive=True)

    # THEN: The result should be wrapped as a statement
    assert isinstance(result, libcst.SimpleStatementLine)
    assert libcst.Module([result]).code == "a * b\n"


def test_multiple_captures() -> None:
    """Test pattern with multiple captures."""
    # GIVEN: A rule that reverses three operands
    rule = PatternRule.from_str("{a} + {b} + {c}", "{c} + {b} + {a}")
    code = libcst.parse_expression("x + y + z")

    # WHEN: The rule is applied
    result = rule.apply(code)

    # THEN: All three captures should be used in reversed order
    assert _as_expr_module(result).code == "z + y + x\n"


def test_invalid_pattern_raises_error() -> None:
    """Test that invalid patterns raise ValueError."""
    # GIVEN: An invalid pattern with unmatched brace
    # WHEN: Creating a rule from the invalid pattern
    # THEN: A ValueError should be raised
    with pytest.raises(ValueError):
        PatternRule.from_str("invalid {", "replacement")


def test_invalid_identifier_raises_error() -> None:
    """Test that non-identifier captures raise ValueError."""
    # GIVEN: A pattern with an invalid identifier (starts with number)
    # WHEN: Creating a rule from the pattern
    # THEN: A ValueError should be raised
    with pytest.raises(ValueError, match="not a valid identifier"):
        PatternRule.from_str("{123invalid}", "replacement")


def test_single_rule_transformation() -> None:
    """Test transformer with a single rule."""
    # GIVEN: A transformer with a rule that changes addition to subtraction
    rule = PatternRule.from_str("{x} + {y}", "{x} - {y}")
    transformer = _PatternRuleTransformer([rule])
    code = libcst.parse_module("a + b\nc + d\n")

    # WHEN: The transformer is applied to the module
    result = code.visit(transformer)

    # THEN: All additions should be changed to subtractions
    assert result.code == "a - b\nc - d\n"


def test_multiple_rules_transformation() -> None:
    """Test transformer with multiple rules applied in sequence."""
    # GIVEN: A transformer with two rules that chain transformations
    rule1 = PatternRule.from_str("{x} * {y}", "{y} * {x}")
    rule2 = PatternRule.from_str("{x} + {y}", "{x} * {y}")
    transformer = _PatternRuleTransformer([rule1, rule2])
    code = libcst.parse_module("a + b\n")

    # WHEN: The transformer is applied
    result = code.visit(transformer)

    # THEN: Both rules should be applied in sequence
    assert result.code == "b * a\n"


def test_multiple_rules_with_cycle() -> None:
    """Test transformer with multiple rules applied in sequence."""
    # GIVEN: A transformer with three rules that represent a cycle
    rule1 = PatternRule.from_str("{x} * {y}", "{y} + {x}")
    rule2 = PatternRule.from_str("{x} + {y}", "{x} - {y}")
    rule3 = PatternRule.from_str("{x} - {y}", "{x} * {y}")
    transformer = _PatternRuleTransformer([rule1, rule2, rule3])
    code = libcst.parse_module("ant + bat\n")

    # WHEN: The transformer is applied
    result = code.visit(transformer)

    # THEN: The cycle should not result in an infinite loop
    assert result.code == "bat * ant\n"


def test_no_match_leaves_code_unchanged() -> None:
    """Test that non-matching rules don't change the code."""
    # GIVEN: A transformer with a rule that doesn't match the code
    rule = PatternRule.from_str("func()", "other_func()")
    transformer = _PatternRuleTransformer([rule])
    code = libcst.parse_module("x + y\n")

    # WHEN: The transformer is applied
    result = code.visit(transformer)

    # THEN: The code should remain unchanged
    assert result.code == "x + y\n"


def _as_expr_module(node: libcst.CSTNode) -> libcst.Module:
    assert isinstance(node, libcst.BaseExpression)
    return libcst.Module([libcst.SimpleStatementLine([libcst.Expr(node)])])
