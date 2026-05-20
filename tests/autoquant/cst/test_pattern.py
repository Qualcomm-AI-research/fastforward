# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from unittest.mock import MagicMock, patch

import libcst
import pytest
import syrupy

from fastforward._autoquant.cst.pattern import PatternRule, PatternRuleTransformer


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
    result = code.visit(PatternRuleTransformer([rule]))

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
    transformer = PatternRuleTransformer([rule])
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
    transformer = PatternRuleTransformer([rule1, rule2])
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
    transformer = PatternRuleTransformer([rule1, rule2, rule3])
    code = libcst.parse_module("ant + bat\n")

    # WHEN: The transformer is applied
    result = code.visit(transformer)

    # THEN: The cycle should not result in an infinite loop
    assert result.code == "bat * ant\n"


def test_no_match_leaves_code_unchanged() -> None:
    """Test that non-matching rules don't change the code."""
    # GIVEN: A transformer with a rule that doesn't match the code
    rule = PatternRule.from_str("func()", "other_func()")
    transformer = PatternRuleTransformer([rule])
    code = libcst.parse_module("x + y\n")

    # WHEN: The transformer is applied
    result = code.visit(transformer)

    # THEN: The code should remain unchanged
    assert result.code == "x + y\n"


def test_small_statement_replacement_does_not_create_nested_statement_line() -> None:
    """Replacing a small statement with expression returns `Expr`, not `SimpleStatementLine`."""
    # GIVEN: A rule that replaces an assignment with an expression, applied to a small statement
    rule = PatternRule.from_str("a = b", "b + 1")
    statement_line = libcst.parse_statement("a = b")
    assert isinstance(statement_line, libcst.SimpleStatementLine)
    small_stmt = statement_line.body[0]

    # WHEN: The rule is applied to the small statement directly
    result = rule.apply(small_stmt)

    # THEN: The result is an `Expr` node, not a `SimpleStatementLine`, so it
    # can be placed back into a `SimpleStatementLine.body` without nesting
    assert isinstance(result, libcst.Expr)
    rebuilt_line = libcst.SimpleStatementLine(body=(result,))
    assert libcst.Module([rebuilt_line]).code == "b + 1\n"


def test_pattern_rule_transformer_handles_semicolon_statements() -> None:
    """Transformer should not crash when traversing semicolon-joined statements."""
    # GIVEN: A module with semicolon-joined statements and a rule that matches within them
    source = "x = 1; y = 2\nif x: print(x); print(y)\n"
    module = libcst.parse_module(source)
    rule = PatternRule.from_str("{a} = 1", "{a} = 100")

    # WHEN: The transformer traverses the module
    transformed = module.visit(PatternRuleTransformer([rule]))

    # THEN: The rule is applied correctly inside the semicolon-joined line
    assert isinstance(transformed, libcst.Module)
    assert transformed.code == "x = 100; y = 2\nif x: print(x); print(y)\n"


@patch("fastforward._autoquant.cst.pattern._to_code", side_effect=TypeError("boom"))
def test_pattern_rule_transformer_fallback_key_on_codegen_error(_mock: MagicMock) -> None:
    """Cycle dedup should continue if `_to_code` raises for a visited statement node."""
    # GIVEN: A module with `_to_code` patched to always raise
    source = "x = 1\n"
    module = libcst.parse_module(source)
    rule = PatternRule.from_str("{a}", "{a}")

    # WHEN: The transformer is applied with the broken serializer
    transformed = module.visit(PatternRuleTransformer([rule]))

    # THEN: No exception is raised and the module is unchanged
    assert isinstance(transformed, libcst.Module)
    assert transformed.code == source


def test_pattern_rule_transformer_replacement_with_multiple_statements_semicolon_and_newline() -> (
    None
):
    """Semicolon-joined replacements are supported, newline-separated replacements are rejected."""
    # GIVEN: A module with a single assignment
    source = "a = 1\n"
    module = libcst.parse_module(source)
    semicolon_rule = PatternRule.from_str("a = 1", "b = 2; c = 3")

    # WHEN: The transformer is applied with a semicolon-joined replacement
    transformed = module.visit(PatternRuleTransformer([semicolon_rule]))

    # THEN: The replacement is accepted and the semicolon is preserved in output
    assert isinstance(transformed, libcst.Module)
    assert transformed.code == "b = 2; c = 3\n"

    # WHEN: A newline-separated replacement is constructed
    # THEN: A ValueError is raised at rule construction time
    with pytest.raises(ValueError, match="Pattern must be a single statement or expression"):
        PatternRule.from_str("a = 1", "b = 2\nc = 3")


def test_pattern_rule_type_annotation_whitespace_robust(
    snapshot: syrupy.assertion.SnapshotAssertion,
) -> None:
    """Rule matching is robust to whitespace variations in the source."""
    # GIVEN: A module having different whitespaces around two similar assignments
    module_source = """
class Model:
    self.property = 3

    def __init__(self, property):
        self.property=property

    def double(self):
        self.property =    self.property * 2
    """

    # WHEN: The PatternRule is applied to add typing for those assignments
    module = libcst.parse_module(module_source)
    rule = PatternRule.from_str("self.property = {value}", "self.property: int = {value}")
    transformed = module.visit(PatternRuleTransformer([rule], module_qualified_name="module"))

    # THEN: The rule is applied correctly everywhere and whitespaces are ignored
    assert isinstance(transformed, libcst.Module)
    assert transformed.code == snapshot


@pytest.mark.parametrize(
    "on",
    [
        None,
        "module",
        "module.Model",
        "module.Model.__init__",
        "Module.__init__",
        "__init__",
        "Module",
    ],
)
def test_pattern_rule_applied_on_fully_qualified_name(
    on: str | None, snapshot: syrupy.assertion.SnapshotAssertion
) -> None:
    """Rule are applied only over correct match over fully-qualified name and children."""
    # GIVEN: A module with a single assignment
    module_source = """
class Model:
    self.property = 3

    def __init__(self, property):
        self.property=property

    def double(self):
        self.property =    self.property * 2
    """

    # WHEN: The PatternRule is applied to the source `on` specific qualifiers
    module = libcst.parse_module(module_source)
    type_annotation_rule = PatternRule.from_str(
        pattern="self.property = {value}", replacement="self.property: int = {value}", on=on
    )
    transformed = module.visit(
        PatternRuleTransformer(
            [type_annotation_rule],
            module_qualified_name="module",
        )
    )

    # THEN: The rule is only applied to part of the source selected by the
    #       fully-qualified name (and recursively to all children)
    assert isinstance(transformed, libcst.Module)
    assert transformed.code == snapshot


def test_pattern_rule_applied_on_fully_qualified_function_name() -> None:
    source = """\
def decode():
    my_var = 1
    return my_var

def foo():
    my_var = 1
    return my_var
"""
    expected = """\
def decode():
    my_var: int = 1
    return my_var

def foo():
    my_var = 1
    return my_var
"""

    # GIVEN A rule scoped to `package.module_a.decode` only
    rule = PatternRule.from_str(
        pattern="my_var = {value}",
        replacement="my_var: int = {value}",
        on="package.module_a.decode",
    )

    # WHEN The module is transformed
    transformed = libcst.parse_module(source).visit(
        PatternRuleTransformer([rule], module_qualified_name="package.module_a")
    )
    print()
    print(transformed.code)
    # THEN Only `decode` is annotated; `foo` is left unchanged
    assert isinstance(transformed, libcst.Module)
    assert transformed.code == expected


def test_pattern_rule_robust_type_annotation_multiple_classes(
    snapshot: syrupy.assertion.SnapshotAssertion,
) -> None:
    """Rule with `on=` only applies inside the named scope correctly when scopes are nested."""
    # GIVEN: A module with a single assignment
    source = """
class ModelInt:
    def __init__(self, property):
        self.property = property

class ModelFloat:
    def __init__(self, property):
        self.property = property
    
    class SubModelFloat:
        def __init__(self, property):
            self.property = property

class ModelDuck:
    def __init__(self, property):
        self.property = property
    
    class SubModelDuckInt:
        def __init__(self, property):
            self.property = property
        
        def foo(self, property):
            self.property = property
    """

    # test should pass even if we specify to replace only inside the `Model` class
    module = libcst.parse_module(source)
    rule_int = PatternRule.from_str(
        pattern="self.property = {value}",
        replacement="self.property: int = {value}",
        on="module.ModelInt",
    )
    rule_float = PatternRule.from_str(
        pattern="self.property = {value}",
        replacement="self.property: float = {value}",
        on="module.ModelFloat",
    )
    rule_sub_duck = PatternRule.from_str(
        pattern="self.property = {value}",
        replacement="self.property: int = {value}",
        on="module.ModelDuck.SubModelDuckInt.foo",
    )

    # WHEN: The transformer is applied
    transformed = module.visit(
        PatternRuleTransformer(
            [rule_int, rule_float, rule_sub_duck], module_qualified_name="module"
        )
    )

    # THEN: The replacement is accepted and the type annotation is appleid everywhere in the code
    assert isinstance(transformed, libcst.Module)
    assert transformed.code == snapshot


class MyFooModule:
    def __init__(self, qualified_name: str) -> None:
        self.qualified_name: str = qualified_name

    @property
    def source(self) -> str:
        return """\
def foo():
    my_var = 1
    return my_var
"""

    @property
    def target_source(self) -> str:
        return """\
def foo():
    my_var: int = 1
    return my_var
"""


def test_pattern_rule_on_disambiguates_same_name_across_modules() -> None:
    """`on` targets a symbol in one module without touching an identically named one elsewhere.

    Two modules `module_a` and `module_b` each define an identical `def foo()`.
    A rule scoped to `package.module_a` should annotate only
    `module_a.foo`'s `my_var` and leave `module_b.foo` untouched.
    """
    # GIVEN: Two separate modules that each define an identically named `foo`
    module_a = MyFooModule(qualified_name="package.module_a")
    module_b = MyFooModule(qualified_name="package.module_b")

    # GIVEN: A rule intended to apply only to the `foo` function  defined in module_b
    rule = PatternRule.from_str(
        pattern="my_var = {value}",
        replacement="my_var: int = {value}",
        on="package.module_a.foo",
    )

    # WHEN: Each module is transformed using the same rule that should only transform `module_a`
    transformed_foo_a = libcst.parse_module(module_a.source).visit(
        PatternRuleTransformer([rule], module_qualified_name=module_a.qualified_name)
    )
    transformed_foo_b = libcst.parse_module(module_b.source).visit(
        PatternRuleTransformer([rule], module_qualified_name=module_b.qualified_name)
    )
    assert isinstance(transformed_foo_a, libcst.Module)
    assert isinstance(transformed_foo_b, libcst.Module)

    # THEN: Only `module_a`'s function is annotated and `module_b` is left unchanged
    assert transformed_foo_a.code == module_a.target_source
    assert transformed_foo_b.code == module_b.source


def test_pattern_rule_on_not_fully_qualified_name_is_not_applied() -> None:
    # GIVEN: A module that define a function named `foo`
    module = MyFooModule(qualified_name="package.module")

    # GIVEN: A rule intended to apply to the `foo` function but
    #        using the middle-part of the fully-qualified name
    rule = PatternRule.from_str(
        pattern="my_var = {value}",
        replacement="my_var: int = {value}",
        on="module.foo",
    )

    # WHEN: The module is transformed using the rule
    module_transformed_source = libcst.parse_module(module.source).visit(
        PatternRuleTransformer([rule], module_qualified_name=module.qualified_name)
    )

    # THEN: `foo` is left unchanged
    assert module_transformed_source.code == module.source


def test_pattern_rule_on_package_is_applied_to_children_modules() -> None:
    # GIVEN: A module that define a function named `foo`
    module = MyFooModule(qualified_name="project.package.sub_package.module")

    # GIVEN: A rule intended to apply to the `foo` function but
    #        using the middle-part of the fully-qualified name
    rule = PatternRule.from_str(
        pattern="my_var = {value}",
        replacement="my_var: int = {value}",
        on="project.package",
    )

    # WHEN: The module is transformed using the rule
    module_transformed_source = libcst.parse_module(module.source).visit(
        PatternRuleTransformer([rule], module_qualified_name=module.qualified_name)
    )

    # THEN: `foo` is left unchanged
    assert module_transformed_source.code == module.target_source


def test_pattern_rule_on_module_is_applied_to_functions() -> None:
    # GIVEN: A module that define a function named `foo`
    module = MyFooModule(qualified_name="project.package.sub_package.module")

    # GIVEN: A rule intended to apply to the `foo` function but
    #        using the middle-part of the fully-qualified name
    rule = PatternRule.from_str(
        pattern="my_var = {value}",
        replacement="my_var: int = {value}",
        on="project.package.sub_package.module",
    )

    # WHEN: The module is transformed using the rule
    module_transformed_source = libcst.parse_module(module.source).visit(
        PatternRuleTransformer([rule], module_qualified_name=module.qualified_name)
    )

    # THEN: `foo` is left unchanged
    assert module_transformed_source.code == module.target_source


def _as_expr_module(node: libcst.CSTNode) -> libcst.Module:
    assert isinstance(node, libcst.BaseExpression)
    return libcst.Module([libcst.SimpleStatementLine([libcst.Expr(node)])])
