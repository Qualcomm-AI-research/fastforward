# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


from enum import Enum, auto
from typing import Iterable, Iterator, NoReturn

import libcst
import libcst.helpers
import libcst.matchers as m

from typing_extensions import override

from fastforward._autoquant.cst import nodes
from fastforward._autoquant.cst.node_creation import create_quantize_statement
from fastforward._autoquant.cst.passes import ConvertSemicolonJoinedStatements
from fastforward._autoquant.cst.quantizer_analysis.unimplemented import NotImplementedMixin

from .annotator import QuantizationAnnotation, QuantizationAnnotationProvider


def _create_inline_quantize_statement(
    expr: libcst.BaseExpression, quantizer_name: nodes.QuantizerReference
) -> libcst.Call:
    """Create a quantize statement.

    Args:
        expr: The expression to be quantized.
        quantizer_name: The name of the quantizer.

    Returns:
        The quantize statement.
    """
    quantize_statement = libcst.helpers.parse_template_expression(
        "self.{quantizer_name}({expr})",
        expr=expr,
        quantizer_name=quantizer_name,
    )
    assert isinstance(quantize_statement, libcst.Call)
    return quantize_statement


def _expr_to_named_expr(name: str, expr: libcst.BaseExpression) -> libcst.BaseExpression:
    name_ = libcst.Name(value=name)
    return libcst.helpers.parse_template_expression("({name_} := {expr})", name_=name_, expr=expr)


class ExprOccurrence(Enum):
    UNIQUE = auto()
    PRIMARY = auto()
    SECONDARY = auto()


class _InlineQuantization:
    """Replaces expressions to be quantized inline with their quantized counterparts.

    Roughly, we replace
    1. `[x for x in y]` with `[quantizer_x(x) for x in y]`
    2. and 3 combined. `[x, x for x in y]` with `[(__ff0:=quantizer_x(x)), __ff0 for x in y]`

    These three cases align with the values of ExprOccurrence.
    """

    def __init__(self) -> None:
        self._node_annotations: dict[libcst.CSTNode, QuantizationAnnotation] = {}
        self._occurrence: dict[QuantizationAnnotation, ExprOccurrence] = {}
        self._named_expr_tracker: dict[QuantizationAnnotation, int] = {}

    def add_annotations(self, annotations: Iterable[QuantizationAnnotation]) -> None:
        """Keeps track of annotations that should be replaced."""
        for annotation in annotations:
            assert isinstance(annotation, QuantizationAnnotation)
            self.add_annotation(annotation)

    def add_annotation(self, annotation: QuantizationAnnotation) -> None:
        """Keeps track of a single annotation that should be replaced."""
        if len(annotation.uses) == 0:
            raise ValueError("Expected a non-zero amount of annotations.")
        elif len(annotation.uses) == 1:
            self._occurrence[annotation] = ExprOccurrence.UNIQUE
        else:
            self._named_expr_tracker[annotation] = len(self._named_expr_tracker)
            self._occurrence[annotation] = ExprOccurrence.PRIMARY

        for use in annotation.uses:
            self._node_annotations[use] = annotation

    def maybe_quantize_expression(
        self, original_node: libcst.CSTNode, updated_node: libcst.CSTNode
    ) -> libcst.CSTNode:
        """Replaces (inline) an expression to be quantized."""
        if not isinstance(updated_node, libcst.BaseExpression):
            return updated_node
        if (annotation := self._node_annotations.get(original_node)) is None:
            return updated_node

        usage = self._occurrence[annotation]
        quantized_statement = _create_inline_quantize_statement(
            expr=updated_node, quantizer_name=nodes.QuantizerReference(annotation.target)
        )

        if usage is ExprOccurrence.UNIQUE:
            return quantized_statement

        varname = f"__ff{self._named_expr_tracker[annotation]}"
        if usage is ExprOccurrence.PRIMARY:
            self._occurrence[annotation] = ExprOccurrence.SECONDARY
            return _expr_to_named_expr(name=varname, expr=quantized_statement)

        return libcst.Name(value=varname)


class QuantizerFunctionTransformer(NotImplementedMixin, ConvertSemicolonJoinedStatements):
    """Quantizes assignments, input parameters, context variables and loop variables.

    Note: We inherit from ConvertSemicolonJoinedStatements as this class may merge several
    statements into a single line, introducing semicolons in the generated code.
    """

    METADATA_DEPENDENCIES = (QuantizationAnnotationProvider,)

    def __init__(self) -> None:
        super().__init__()
        self._inline_quantization = _InlineQuantization()

    def _get_annotations(
        self,
        original_node: libcst.CSTNodeT,
    ) -> Iterator[QuantizationAnnotation] | set[QuantizationAnnotation] | None:
        try:
            return self.get_metadata(QuantizationAnnotationProvider, original_node)
        except KeyError:
            return None

    def leave_FunctionDef(
        self, original_node: libcst.FunctionDef, updated_node: libcst.FunctionDef
    ) -> libcst.FunctionDef:
        statements = []
        for original_param, updated_param in zip(
            original_node.params.params,
            updated_node.params.params,
        ):
            if not original_param.name.deep_equals(updated_param.name):
                _fail_due_to_mismatch()
            if self._get_annotations(original_param) is None:
                continue
            statement = create_quantize_statement(
                name=updated_param.name.value,
                quantizer_name=nodes.QuantizerReference(updated_param.name.value),
            )
            statements.append(statement)

        # Keep docstring in its original place
        if _function_has_docstring(updated_node):
            updated_body = [updated_node.body.body[0], *statements, *updated_node.body.body[1:]]
        else:
            updated_body = [*statements, *updated_node.body.body]
        return updated_node.with_changes(body=updated_node.body.with_changes(body=updated_body))

    def leave_GeneralAssignment(
        self, original_node: nodes.GeneralAssignment, updated_node: nodes.GeneralAssignment
    ) -> nodes.GeneralAssignment | libcst.FlattenSentinel[libcst.BaseStatement]:
        if original_node.original != updated_node.original:
            _fail_due_to_mismatch()
        if (annotations := self._get_annotations(original_node)) is None:
            return updated_node

        statements = []
        for annotation in annotations:
            target = annotation.target
            statement = create_quantize_statement(
                name=target, quantizer_name=nodes.QuantizerReference(target)
            )
            statements.append(statement.body[0])

        return libcst.FlattenSentinel((updated_node, *statements))  # type:ignore[arg-type]

    def leave_For(self, original_node: libcst.For, updated_node: libcst.For) -> libcst.For:
        if (annotations := self._get_annotations(original_node.target)) is None:
            return updated_node

        statements = []
        for annotation in annotations:
            statement = create_quantize_statement(
                name=annotation.target,
                quantizer_name=nodes.QuantizerReference(annotation.target),
            )
            statements.append(statement)

        return updated_node.with_changes(
            body=updated_node.body.with_changes(
                body=[
                    *statements,
                    *updated_node.body.body,
                ]
            )
        )

    def leave_With(self, original_node: libcst.With, updated_node: libcst.With) -> libcst.With:
        statements = []

        for original_item, updated_item in zip(original_node.items, updated_node.items):
            if (original_item.asname is None) + (updated_item.asname is None) < 2:
                if original_item.asname is not None and updated_item.asname is not None:
                    if not original_item.asname.deep_equals(updated_item.asname):
                        _fail_due_to_mismatch()
                else:
                    _fail_due_to_mismatch()

            if self._get_annotations(original_item) is None:
                continue

            match updated_item:
                case libcst.WithItem(
                    asname=libcst.AsName(
                        name=libcst.Name(value),
                    )
                ):
                    statement = create_quantize_statement(
                        name=value,
                        quantizer_name=nodes.QuantizerReference(value),
                    )
                case _:
                    self.warn_not_implemented(
                        f"Unexpected QuantizationAnnotation of type {type(updated_item)}"
                    )
            statements.append(statement)
        updated_body = updated_node.body.with_changes(body=(*statements, *updated_node.body.body))
        return updated_node.with_changes(body=updated_body)

    def visit_ListComp(self, _node: libcst.ListComp) -> bool:
        return False

    def visit_SetComp(self, _node: libcst.SetComp) -> bool:
        return False

    def visit_GeneratorExp(self, _node: libcst.GeneratorExp) -> bool:
        return False

    def visit_DictComp(self, _node: libcst.DictComp) -> bool:
        return False

    def leave_ListComp(
        self, _original_node: libcst.ListComp, updated_node: libcst.ListComp
    ) -> libcst.ListComp:
        # Note that we visit here and not in the visit_ListComp, as we wish to visit in
        # the order following the semantics of evaluations. Quantization annotations are on the
        # `for_in`, so we need to get that info before descending into `elt`.
        for_in = updated_node.for_in.visit(self)
        elt = updated_node.elt.visit(self)
        return updated_node.with_changes(for_in=for_in, elt=elt)

    def leave_SetComp(
        self, _original_node: libcst.SetComp, updated_node: libcst.SetComp
    ) -> libcst.SetComp:
        for_in = updated_node.for_in.visit(self)
        elt = updated_node.elt.visit(self)
        return updated_node.with_changes(for_in=for_in, elt=elt)

    def leave_GeneratorExp(
        self, _original_node: libcst.GeneratorExp, updated_node: libcst.GeneratorExp
    ) -> libcst.GeneratorExp:
        for_in = updated_node.for_in.visit(self)
        elt = updated_node.elt.visit(self)
        return updated_node.with_changes(for_in=for_in, elt=elt)

    def leave_DictComp(
        self, _original_node: libcst.DictComp, updated_node: libcst.DictComp
    ) -> libcst.DictComp:
        for_in = updated_node.for_in.visit(self)
        key = updated_node.key.visit(self)
        value = updated_node.value.visit(self)
        return updated_node.with_changes(for_in=for_in, key=key, value=value)

    @override
    def visit_CompFor(
        self,
        original_node: libcst.CompFor,
    ) -> bool:
        if (annotations := self._get_annotations(original_node.target)) is None:
            return True
        self._inline_quantization.add_annotations(annotations)
        return True

    @override
    def on_leave(
        self, original_node: libcst.CSTNodeT, updated_node: libcst.CSTNodeT
    ) -> libcst.CSTNodeT | libcst.RemovalSentinel | libcst.FlattenSentinel[libcst.CSTNodeT]:
        """Looks for inline replacements of quantized expressions."""
        modified_updated_node = super().on_leave(original_node, updated_node)
        if not isinstance(modified_updated_node, libcst.CSTNode):
            return modified_updated_node
        return self._inline_quantization.maybe_quantize_expression(  # type: ignore[return-value]
            original_node, modified_updated_node
        )


def _function_has_docstring(node: libcst.FunctionDef) -> bool:
    return m.matches(
        node,
        m.FunctionDef(
            body=m.IndentedBlock(
                body=[
                    m.SimpleStatementLine(
                        body=[
                            m.Expr(value=m.SimpleString()),
                            m.ZeroOrMore(),
                        ]
                    ),
                    m.ZeroOrMore(),
                ]
            )
        ),
    )


def _fail_due_to_mismatch() -> NoReturn:
    raise ValueError("Inconsistent annotations for original and updated node")
