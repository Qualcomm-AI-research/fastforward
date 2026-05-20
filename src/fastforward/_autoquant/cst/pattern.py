# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import dataclasses
import string

from typing import Any, Sequence, Union, overload

import libcst
import libcst.matchers as m

from libcst import CSTNode, FlattenSentinel, RemovalSentinel
from typing_extensions import Self, override

PATTERN_PREFIX = "__FF__CAPTURE__PATTERN_"


@dataclasses.dataclass
class PatternRule:
    """A rule that matches a pattern in a CST node and replaces it with a replacement.

    Attributes:
        pattern: A matcher node that defines the pattern to match.
        replacement: A CST node that will replace the matched pattern.
        on: Optional dotted scope name restricting where the rule applies.
            The scope name is matched as a contiguous segment of the current
            fully-qualified scope, which (when running via the autoquant
            preprocessing flow) includes the module name. This means `on` should
            be an fully-qualified path (e.g. `my_package.my_module.MyClass.foo`)
            used to disambiguate identically named symbols across modules.
            The rule applies inside that scope and any of its nested scopes.
            `None` means apply everywhere. NB: `on` is only honored when the rule runs
            through `PatternRuleTransformer`, a direct `apply(cst)`
            call has no scope context and therefore ignores `on`.
    """

    pattern: m.BaseMatcherNode
    replacement: libcst.CSTNode
    on: str | None = None

    def apply(self, cst: libcst.CSTNode) -> libcst.CSTNode:
        """Apply the pattern rule to a CST node.

        Args:
            cst: The CST node to apply the rule to.

        Returns:
            The transformed CST node if the pattern matches, otherwise the original node.
        """
        captures = m.extract(cst, self.pattern)
        if captures is None and isinstance(cst, libcst.SimpleStatementLine) and len(cst.body) == 1:
            # Allow small-statement patterns (e.g., "a = 1") to match a full
            # statement line so multi-small-statement replacements can be
            # returned as a valid `SimpleStatementLine`.
            captures = m.extract(cst.body[0], self.pattern)

        if captures is None:
            return cst

        for key, capture in captures.items():
            if not isinstance(capture, libcst.CSTNode):
                msg = f"Capture '{key}' is used multiple times. This is not supported."
                raise ValueError(msg)

        replacement = self.replacement.visit(_CaptureTransformer(captures))  # type: ignore[arg-type]
        assert isinstance(replacement, libcst.CSTNode)

        if isinstance(cst, libcst.BaseSmallStatement):
            match replacement:
                case libcst.BaseExpression():
                    replacement = libcst.Expr(replacement)
                case libcst.SimpleStatementLine(body=body) if len(body) == 1:
                    replacement = body[0]
                case libcst.BaseStatement():
                    # A small-statement slot cannot hold a full statement node.
                    # Defer this rewrite to the enclosing `SimpleStatementLine`
                    # where applicable.
                    return cst
        elif isinstance(cst, libcst.BaseStatement):
            match replacement:
                case libcst.BaseExpression():
                    replacement = libcst.SimpleStatementLine([libcst.Expr(replacement)])
                case libcst.BaseSmallStatement():
                    replacement = libcst.SimpleStatementLine([replacement])

        return replacement

    @classmethod
    def from_str(cls, pattern: str, replacement: str, on: str | None = None) -> Self:
        """Create a PatternRule from string representations of pattern and replacement.

        Args:
            pattern: A string representation of the pattern to match.
            replacement: A string representation of the replacement to use.
            on: Optional dotted scope name (e.g. `my_package.my_module.MyClass.foo`)
                restricting where the rule applies. `None` to apply everywhere.

        Returns:
            A new PatternRule instance with the parsed pattern and replacement.
        """
        return cls(
            pattern=_convert_to_matcher(_parse_pattern(pattern)),
            replacement=_parse_pattern(replacement),
            on=on,
        )


def _to_code(node: libcst.CSTNode) -> str:
    if isinstance(node, libcst.Annotation):
        # Annotation._codegen_impl requires an extra argument, so wrap it in an
        # AnnAssign (e.g., "p: int") and strip the "p: " prefix to get just the annotation
        code_repr = _to_code(libcst.AnnAssign(target=libcst.Name("p"), annotation=node))
        return code_repr[3:]
    return libcst.Module([]).code_for_node(node)


def _applicable_in_module(on: str | None, module_qualified_name: str) -> bool:
    if on is None or not module_qualified_name:
        return True
    if "." not in on:
        return True
    return on.startswith(module_qualified_name) or module_qualified_name.startswith(on)


class PatternRuleTransformer(libcst.CSTTransformer):
    def __init__(
        self,
        rules: Sequence[PatternRule],
        module_qualified_name: str = "",
    ):
        self._scope_stack: list[str] = []
        self._rules = tuple(
            rule for rule in rules if _applicable_in_module(rule.on, module_qualified_name)
        )
        self._module_qualified_name = module_qualified_name

    @override
    def visit_ClassDef(self, node: libcst.ClassDef) -> bool | None:
        self._scope_stack.append(node.name.value)
        return None

    @override
    def leave_ClassDef(
        self, original_node: libcst.ClassDef, updated_node: libcst.ClassDef
    ) -> libcst.ClassDef:
        self._scope_stack.pop()
        return updated_node

    @override
    def visit_FunctionDef(self, node: libcst.FunctionDef) -> bool | None:
        self._scope_stack.append(node.name.value)
        return None

    @override
    def leave_FunctionDef(
        self, original_node: libcst.FunctionDef, updated_node: libcst.FunctionDef
    ) -> libcst.FunctionDef:
        self._scope_stack.pop()
        return updated_node

    def _current_scope_parts(self) -> list[str]:
        parts: list[str] = []
        if self._module_qualified_name:
            parts.extend(self._module_qualified_name.split("."))
        parts.extend(self._scope_stack)
        return parts

    def _scope_matches(self, on: str | None) -> bool:
        if on is None:
            return True
        on_parts = on.split(".")
        scope_parts = self._current_scope_parts()
        return scope_parts[: len(on_parts)] == on_parts

    @override
    def on_leave(  # type: ignore[override]
        self, original_node: CSTNode, updated_node: CSTNode
    ) -> Union[CSTNode, RemovalSentinel, FlattenSentinel[CSTNode]]:
        updated_node = super().on_leave(original_node, updated_node)  # type: ignore[assignment]
        return self._apply_rules(updated_node)

    @staticmethod
    def _node_key(node: libcst.CSTNode) -> str:
        """Return a stable key for cycle detection without hard-failing on codegen edge cases."""
        try:
            return _to_code(node)
        except Exception:
            return f"{type(node).__name__}:{repr(node)}"

    def _apply_rules(self, node: CSTNode) -> CSTNode:
        idx = 0
        current_node = node
        seen_nodes = {self._node_key(current_node)}
        while idx < len(self._rules):
            rule = self._rules[idx]
            if not self._scope_matches(rule.on):
                idx += 1
                continue
            updated_node = rule.apply(current_node)
            if updated_node is current_node:
                # continue loop early without extra checks
                idx += 1
            elif (code_repr := self._node_key(updated_node)) not in seen_nodes:
                idx = 0
                seen_nodes.add(code_repr)
                current_node = updated_node
            else:
                idx += 1
        return current_node


def _get_matcher_for_node(
    node: libcst.CSTNode | Any,
) -> m.BaseMatcherNode | m.DoNotCareSentinel | None:
    match node:
        case (
            libcst.SimpleWhitespace()
            | libcst.TrailingWhitespace()
            | libcst.ParenthesizedWhitespace()
            | libcst.MaybeSentinel()
            | libcst.EmptyLine()
            | libcst.Comment()
        ):
            return m.DoNotCare()

        case libcst.Name(value) if value.startswith(PATTERN_PREFIX):
            name = value.removeprefix(PATTERN_PREFIX)
            return m.SaveMatchedNode(m.DoNotCare(), name)

        case libcst.CSTNode():
            try:
                # libcst matcher names match CST node names, so we can get the matcher by name.
                return getattr(m, type(node).__name__)  # type: ignore[no-any-return]
            except AttributeError:
                return m.DoNotCare()

        case _:
            return None


@overload
def _convert_to_matcher_impl(node: libcst.CSTNode) -> m.BaseMatcherNode | m.DoNotCareSentinel: ...


@overload
def _convert_to_matcher_impl(node: Any) -> Any: ...


def _convert_to_matcher_impl(node: Any) -> Any:

    if not (Matcher := _get_matcher_for_node(node)):
        return node

    matcher_fields = set(getattr(Matcher, "__dataclass_fields__", {}).keys())
    matcher_fields &= set(getattr(node, "__dataclass_fields__", {}).keys())

    fields = {}
    for field in matcher_fields:
        value = getattr(node, field)

        match value:
            case [*_]:
                value = tuple(_convert_to_matcher_impl(v) for v in value)
            case _:
                value = _convert_to_matcher_impl(value)

        fields[field] = value

    ignore_fields = ["footer", "header", "leading_lines"]
    for field in ignore_fields:
        if field in fields:
            del fields[field]

    if callable(Matcher):
        Matcher = Matcher(**fields)  # type: ignore[unreachable]
    return Matcher


@overload
def _convert_to_matcher(node: libcst.CSTNode) -> m.BaseMatcherNode: ...


@overload
def _convert_to_matcher(node: Any) -> Any: ...


def _convert_to_matcher(node: Any) -> Any:
    """Converts a CST node to a matcher node.

    Args:
        node: A libcst.CSTNode to convert to a matcher

    Returns:
        A matcher node corresponding to the input CST node
    """
    matcher = _convert_to_matcher_impl(node)
    if isinstance(matcher, m.DoNotCareSentinel):
        matcher = m.OneOf(matcher)
    return matcher


def _parse_pattern(pattern: str) -> libcst.CSTNode:
    named_captures = []
    for line in string.Formatter().parse(pattern):
        _, ident, *_ = line
        if ident is None:
            continue
        if not ident.isidentifier():
            msg = f"Only identifiers can be used in patterns, '{ident}' is not a valid identifier"
            raise ValueError(msg)
        named_captures.append(ident)

    cst: libcst.CSTNode
    source = pattern.format(**{name: f"{PATTERN_PREFIX}{name}" for name in named_captures})
    try:
        cst = libcst.parse_statement(source)
    except libcst.ParserSyntaxError:
        raise ValueError("Parse failure. Pattern must be a single statement or expression.")
    if isinstance(cst, libcst.SimpleStatementLine) and len(cst.body) == 1:
        cst = cst.body[0]
    if isinstance(cst, libcst.Expr):
        cst = cst.value

    return cst


class _CaptureTransformer(libcst.CSTTransformer):
    def __init__(self, captures: dict[str, libcst.CSTNode]) -> None:
        super().__init__()
        self._captures = captures

    def leave_Name(self, original_node: libcst.Name, updated_node: libcst.Name) -> libcst.CSTNode:  # type: ignore[override]
        value = original_node.value
        if not value.startswith(PATTERN_PREFIX):
            return updated_node
        key = value.removeprefix(PATTERN_PREFIX)
        try:
            return self._captures[key]
        except KeyError:
            msg = f"'{key}' is not captured from input."
            raise KeyError(msg)
