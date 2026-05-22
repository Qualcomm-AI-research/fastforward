# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


import logging

from enum import Enum, auto
from typing import Iterable, NoReturn, TypeVar, cast

import libcst
import libcst.helpers
import libcst.matchers as m

from typing_extensions import override

from fastforward._autoquant.cst import nodes
from fastforward._autoquant.cst.node_creation import create_quantize_statement
from fastforward._autoquant.cst.passes import ConvertSemicolonJoinedStatements
from fastforward._autoquant.cst.quantizer_analysis.unimplemented import NotImplementedMixin
from fastforward._autoquant.pybuilder import QuantizerReferenceCollection

from .annotator import QuantizationAnnotation, QuantizationAnnotationProvider

logger = logging.getLogger(__name__)


def _create_inline_quantize_statement(
    expr: libcst.BaseExpression, quantizer_name: libcst.BaseExpression
) -> libcst.Call:
    """Create a quantize statement.

    Args:
        expr: The expression to be quantized.
        quantizer_name: The name of the quantizer.

    Returns:
        The quantize statement.
    """
    quantize_statement = libcst.helpers.parse_template_expression(
        "{quantizer_name}({expr})",
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
    """Apply inline quantization with safe reuse for repeated expression uses.

    For each quantization annotation this helper rewrites expression occurrences
    directly in-place. A single occurrence becomes ``quantizer(expr)``.
    Repeated occurrences within the same expression use a walrus-bound temporary
    (``__ffN``) so the quantizer call runs once and later uses reference the
    bound value.

    Quantizer references are memoized by ``annotation.quantizer_id`` to keep a
    stable one-to-one mapping between data-flow annotation and generated
    quantizer symbol.
    """

    def __init__(self, quantizer_refs: QuantizerReferenceCollection) -> None:
        self._node_annotations: dict[libcst.CSTNode, QuantizationAnnotation] = {}
        self._occurrence: dict[QuantizationAnnotation, ExprOccurrence] = {}
        self._named_expr_tracker: dict[QuantizationAnnotation, int] = {}
        self._quantizer_refs = quantizer_refs
        self._quantizer_refs_by_id: dict[int, libcst.BaseExpression] = {}

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
        """Rewrite a node to inline quantization when it has an annotation.

        For repeated annotated uses, this method emits a walrus assignment on
        the primary occurrence and ``Name`` references on secondary occurrences.
        """
        if not isinstance(updated_node, libcst.BaseExpression):
            return updated_node
        if (annotation := self._node_annotations.get(original_node)) is None:
            return updated_node

        usage = self._occurrence[annotation]

        # _quantizer_refs.create_quantizer_expression has side effects. Only
        # call it when the result is actually used.
        def quant_statement() -> libcst.Call:
            quantizer_id = annotation.quantizer_id
            if quantizer_id is not None and quantizer_id in self._quantizer_refs_by_id:
                quantizer_ref = self._quantizer_refs_by_id[quantizer_id]
            else:
                quantizer_ref = self._quantizer_refs.create_quantizer_expression(annotation.target)
                if quantizer_id is not None:
                    self._quantizer_refs_by_id[quantizer_id] = quantizer_ref
            return _create_inline_quantize_statement(
                expr=updated_node,
                quantizer_name=quantizer_ref,
            )

        if usage is ExprOccurrence.UNIQUE:
            return quant_statement()

        varname = f"__ff{self._named_expr_tracker[annotation]}"
        if usage is ExprOccurrence.PRIMARY:
            self._occurrence[annotation] = ExprOccurrence.SECONDARY
            return _expr_to_named_expr(name=varname, expr=quant_statement())

        return libcst.Name(value=varname)


class QuantizerFunctionTransformer(NotImplementedMixin, ConvertSemicolonJoinedStatements):
    """Inject quantization statements/replacements for annotated function bodies.

    The transformer combines three mechanisms:
    - explicit quantize statements inserted at dependency-safe positions,
    - inline quantization for unsafe expression contexts where hoisting could
      cause load-before-assignment,
    - block-local deduplication of repeated quantizer calls.

    It inherits from ``ConvertSemicolonJoinedStatements`` because some rewrites
    can introduce semicolon-joined output that must be normalized.
    """

    METADATA_DEPENDENCIES = (QuantizationAnnotationProvider,)

    def __init__(
        self,
        quantizer_refs: QuantizerReferenceCollection,
    ) -> None:
        super().__init__()
        self._inline_quantization = _InlineQuantization(quantizer_refs)
        self._quantizer_refs = quantizer_refs
        self._expr_replacements: dict[libcst.CSTNode, libcst.CSTNode] = {}
        self._inline_quantize_replacements: dict[libcst.CSTNode, libcst.BaseExpression] = {}
        self._quantizer_refs_by_id: dict[int, libcst.BaseExpression] = {}

    def _get_annotations(
        self,
        original_node: libcst.CSTNode,
    ) -> list[QuantizationAnnotation] | None:
        if metadata := self.get_metadata(QuantizationAnnotationProvider, original_node, None):
            return list(sorted(metadata))
        return None

    @staticmethod
    def _should_inline_quantized_use(use: libcst.CSTNode) -> bool:
        """Return whether quantization of `use` must happen inline.

        Hoisting quantization of container/subscript expressions to the beginning
        of a block can read loop-scoped locals before they are assigned.
        """
        return isinstance(
            use,
            (
                libcst.Subscript,
                libcst.List,
                libcst.Tuple,
                libcst.Set,
                libcst.Dict,
            ),
        )

    def _quantizer_ref_for_annotation(
        self, annotation: QuantizationAnnotation
    ) -> libcst.BaseExpression:
        """Return a stable quantizer expression for an annotation.

        ``annotation.quantizer_id`` is used as cache key so separate rewrites
        that refer to the same data-flow annotation reuse the same quantizer
        symbol.
        """
        quantizer_id = annotation.quantizer_id
        if quantizer_id is not None and quantizer_id in self._quantizer_refs_by_id:
            return self._quantizer_refs_by_id[quantizer_id]

        quantizer_ref = self._quantizer_refs.create_quantizer_expression(annotation.target)
        if quantizer_id is not None:
            self._quantizer_refs_by_id[quantizer_id] = quantizer_ref
        return quantizer_ref

    @staticmethod
    def _node_code(node: libcst.CSTNode) -> str:
        return libcst.Module([]).code_for_node(node)

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
                target=updated_param.name.value,
                quantizer_ref=self._quantizer_refs.create_quantizer_expression(
                    updated_param.name.value
                ),
            )
            statements.append(statement)

        # Keep docstring in its original place
        updated_body = _insert_in_indented_block(updated_node.body, statements)
        updated_function = updated_node.with_changes(body=updated_body)

        # Run call dedup after all expression rewrites are in place so repeated
        # inline quantizer calls can be materialized into shared `_tmp_` values.
        deduped = updated_function.visit(_QuantizerCallDedup())
        if not isinstance(deduped, libcst.FunctionDef):
            _fail_due_to_mismatch()
        return deduped

    def visit_IndentedBlock(self, node: libcst.IndentedBlock) -> bool:
        # Record replacements for every annotation in the block.
        # Unsafe expression uses are inlined unless the annotation can be
        # materialized and reused via its target variable.
        if (annotations := self._get_annotations(node)) is not None:
            for annotation in annotations:
                quantizer_ref: libcst.BaseExpression | None = None
                for use in annotation.uses:
                    if isinstance(use, libcst.Name):
                        self._expr_replacements[use] = libcst.Name(annotation.target)
                    elif self._should_inline_quantized_use(use):
                        if quantizer_ref is None:
                            quantizer_ref = self._quantizer_ref_for_annotation(annotation)
                        self._inline_quantize_replacements[use] = quantizer_ref
                    else:
                        self._expr_replacements[use] = libcst.Name(annotation.target)

        return True

    def leave_IndentedBlock(
        self, original_node: libcst.IndentedBlock, updated_node: libcst.IndentedBlock
    ) -> libcst.IndentedBlock:
        if (annotations := self._get_annotations(original_node)) is None:
            # Reuse can still be improved in blocks where quantized calls were
            # emitted inline, even if the block itself carries no annotations.
            return _dedup_repeated_quantizer_calls(updated_node)

        # Insert quantizer call statements for annotations that are not replaced
        # inline. Placement is dependency-aware: for each injected statement we
        # place it after the first assignment of all local names loaded by the
        # statement's source expression.
        quantize_statements: list[tuple[libcst.SimpleStatementLine, libcst.BaseExpression]] = []
        for annotation in annotations:
            use_expr = cast(libcst.BaseExpression, annotation.uses[0])
            if self._should_inline_quantized_use(use_expr):
                continue

            statement = create_quantize_statement(
                target=annotation.target,
                source=use_expr,
                quantizer_ref=self._quantizer_ref_for_annotation(annotation),
            )
            quantize_statements.append((statement, use_expr))

        updated = _insert_dependency_aware_quantize_statements(updated_node, quantize_statements)
        return _dedup_repeated_quantizer_calls(updated)

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
                target=target,
                quantizer_ref=self._quantizer_refs.create_quantizer_expression(target),
            )
            statements.append(statement.body[0])

        return libcst.FlattenSentinel((updated_node, *statements))  # type:ignore[arg-type]

    def leave_For(self, original_node: libcst.For, updated_node: libcst.For) -> libcst.For:
        if (annotations := self._get_annotations(original_node.target)) is None:
            return updated_node

        statements = []
        for annotation in annotations:
            statement = create_quantize_statement(
                target=annotation.target,
                quantizer_ref=self._quantizer_refs.create_quantizer_expression(annotation.target),
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
                    statements.append(
                        create_quantize_statement(
                            target=value,
                            quantizer_ref=self._quantizer_refs.create_quantizer_expression(value),
                        )
                    )
                case _:
                    self.warn_not_implemented(
                        f"Unexpected QuantizationAnnotation of type {type(updated_item)}"
                    )
        updated_body = updated_node.body.with_changes(body=(*statements, *updated_node.body.body))
        return updated_node.with_changes(body=updated_body)

    def visit_ListComp(self, node: libcst.ListComp) -> bool:
        del node
        return False

    def visit_SetComp(self, node: libcst.SetComp) -> bool:
        del node
        return False

    def visit_GeneratorExp(self, node: libcst.GeneratorExp) -> bool:
        del node
        return False

    def visit_DictComp(self, node: libcst.DictComp) -> bool:
        del node
        return False

    def leave_ListComp(
        self, original_node: libcst.ListComp, updated_node: libcst.ListComp
    ) -> libcst.ListComp:
        del original_node
        # Note that we visit here and not in the visit_ListComp, as we wish to visit in
        # the order following the semantics of evaluations. Quantization annotations are on the
        # `for_in`, so we need to get that info before descending into `elt`.
        for_in = updated_node.for_in.visit(self)
        elt = updated_node.elt.visit(self)
        return updated_node.with_changes(for_in=for_in, elt=elt)

    def leave_SetComp(
        self, original_node: libcst.SetComp, updated_node: libcst.SetComp
    ) -> libcst.SetComp:
        del original_node
        for_in = updated_node.for_in.visit(self)
        elt = updated_node.elt.visit(self)
        return updated_node.with_changes(for_in=for_in, elt=elt)

    def leave_GeneratorExp(
        self, original_node: libcst.GeneratorExp, updated_node: libcst.GeneratorExp
    ) -> libcst.GeneratorExp:
        del original_node
        for_in = updated_node.for_in.visit(self)
        elt = updated_node.elt.visit(self)
        return updated_node.with_changes(for_in=for_in, elt=elt)

    def leave_DictComp(
        self, original_node: libcst.DictComp, updated_node: libcst.DictComp
    ) -> libcst.DictComp:
        del original_node
        for_in = updated_node.for_in.visit(self)
        key = updated_node.key.visit(self)
        value = updated_node.value.visit(self)
        return updated_node.with_changes(for_in=for_in, key=key, value=value)

    @override
    def visit_CompFor(
        self,
        node: libcst.CompFor,
    ) -> bool:
        if (annotations := self._get_annotations(node.target)) is None:
            return True
        self._inline_quantization.add_annotations(annotations)
        return True

    @override
    def on_leave(  # type: ignore[override]
        self, original_node: libcst.CSTNodeT, updated_node: libcst.CSTNodeT
    ) -> libcst.CSTNode | libcst.RemovalSentinel | libcst.FlattenSentinel[libcst.CSTNodeT]:
        """Looks for inline replacements of quantized expressions."""
        # If original node is included in _expr_replacements, use the replacement and don't
        # process further. The `_expr_replacements` map is updated `visit_IndentedBlock`.
        if original_node in self._expr_replacements:
            return self._expr_replacements[original_node]

        if original_node in self._inline_quantize_replacements:
            if not isinstance(updated_node, libcst.BaseExpression):
                _fail_due_to_mismatch()
            quantizer_ref = self._inline_quantize_replacements[original_node]
            return _create_inline_quantize_statement(updated_node, quantizer_ref)

        modified_updated_node = super().on_leave(original_node, updated_node)
        if not isinstance(modified_updated_node, libcst.CSTNode):
            return modified_updated_node

        return self._inline_quantization.maybe_quantize_expression(
            original_node, modified_updated_node
        )


_BaseSuiteT = TypeVar("_BaseSuiteT", libcst.BaseSuite, libcst.IndentedBlock)


def _has_docstring(node: _BaseSuiteT) -> bool:
    return m.matches(
        node,
        m.IndentedBlock(
            body=[
                m.SimpleStatementLine(body=[m.Expr(value=m.SimpleString()), m.ZeroOrMore()]),
                m.ZeroOrMore(),
            ]
        ),
    )


def _insert_in_indented_block(
    node: _BaseSuiteT,
    nodes: Iterable[libcst.SimpleStatementLine | libcst.BaseCompoundStatement],
) -> _BaseSuiteT:
    """Insert `statements` at the beginning of a `node`, preserving docstrings."""
    assert not isinstance(node, libcst.SimpleStatementSuite)
    if _has_docstring(node):
        updated_body = (node.body[0], *nodes, *node.body[1:])
    else:
        updated_body = (*nodes, *node.body)
    return node.with_changes(body=updated_body)


def _insert_dependency_aware_quantize_statements(
    node: libcst.IndentedBlock,
    statements: Iterable[tuple[libcst.SimpleStatementLine, libcst.BaseExpression]],
) -> libcst.IndentedBlock:
    """Insert the quantize statements of tensor-expressions immediately before their use in the source-expression.

    Each tuple contains the quantize statement to insert and the source
    expression used to derive dependencies. This statement must be placed
    after the last statement that could have a side effect on the tensor-expression
    but before it's use in the source-expression.
    Placing the hoisted `target = quantizer(source)` statement directly before
    the use therefore inherits that guarantee without any dataflow analysis.

    E.g.:

    Original Code:
    ```
        def forward(a, b):
            a = b.T
            c = foo(a + b)
    ```

    Autoquant code should be:
    ```
        def forward(a, b):
            a = b.T
            a_b = quantize_add(a, b)
            c = foo(a_b)
    ```

    And not:
    ```
        def forward(a, b):
            a_b = quantize_add(a, b)
            a = b.T
            c = foo(a_b)
    ```
    """
    orig_body = list(node.body)
    if not orig_body:
        return node

    docstring_offset = 1 if _has_docstring(node) else 0

    # 1. Find the new statements to insert with their position relative to
    #    the original body
    # --------------------------------------------------------------------------
    new_statements: list[tuple[int, libcst.SimpleStatementLine]] = []
    for statement, _source_expr in statements:
        target_name = _quantize_statement_target_name(statement)
        if target_name is None:
            # Defensive: the helper that builds these statements always emits
            # `Name = ...` targets, but if a future change breaks that, fall
            # back to appending so we don't crash mid-rewrite.
            insert_idx = len(orig_body)
        else:
            use_idx = _first_use_index_in_block(orig_body, target_name)
            if use_idx is None:
                insert_idx = len(orig_body)
            else:
                insert_idx = max(use_idx, docstring_offset)
        new_statements.append((insert_idx, statement))
    # --------------------------------------------------------------------------

    if not new_statements:
        return node

    # 2. Merge the new statements in the original body
    # --------------------------------------------------------------------------
    new_body: list[libcst.BaseStatement] = []

    # `sort` keeps original order among equal keys,
    # the loop below preserves that order in the new body.
    new_statements.sort(key=lambda idx_stmt: idx_stmt[0])
    new_stmt_iter = iter(new_statements)

    # Get the first "new-statement" to insert,
    # then loop over the original-body's statements.
    new_stmt: tuple[int, libcst.SimpleStatementLine] | None = next(new_stmt_iter, None)
    for orig_idx, orig_stmt in enumerate(orig_body):
        # Insert all the new-stmts that should come before the current orig_stmt
        while new_stmt is not None and new_stmt[0] <= orig_idx:
            new_body.append(new_stmt[1])
            new_stmt = next(new_stmt_iter, None)

        new_body.append(orig_stmt)

    # Insert all the remaining new-stmts that should be appended to the body
    while new_stmt is not None:
        new_body.append(new_stmt[1])
        new_stmt = next(new_stmt_iter, None)
    # --------------------------------------------------------------------------

    return node.with_changes(body=tuple(new_body))


def _quantize_statement_target_name(stmt: libcst.SimpleStatementLine) -> str | None:
    """Return the LHS name of a single ``target = ...`` quantize statement."""
    if len(stmt.body) != 1:
        return None
    inner = stmt.body[0]
    if not isinstance(inner, libcst.Assign) or len(inner.targets) != 1:
        return None
    target = inner.targets[0].target
    if isinstance(target, libcst.Name):
        return target.value
    return None


class _BlockScopeNameLoadFinder(libcst.CSTVisitor):
    """Find Names loaded within a statement, without crossing function/class scopes.

    Names appearing as assignment targets are excluded so writes don't count
    as loads.
    """

    def __init__(self, target_name: str) -> None:
        super().__init__()
        self._target_name = target_name
        self.found = False
        self._exclude_ids: set[int] = set()

    def _exclude_target(self, expr: libcst.BaseExpression) -> None:
        if isinstance(expr, libcst.Name):
            self._exclude_ids.add(id(expr))
        elif isinstance(expr, (libcst.Tuple, libcst.List)):
            for el in expr.elements:
                if isinstance(el, libcst.Element):
                    self._exclude_target(el.value)

    @override
    def visit_Assign(self, node: libcst.Assign) -> bool:
        for tgt in node.targets:
            self._exclude_target(tgt.target)
        return True

    @override
    def visit_AugAssign(self, node: libcst.AugAssign) -> bool:
        # Aug-assign reads the target before storing — keep it as a load.
        return True

    @override
    def visit_AnnAssign(self, node: libcst.AnnAssign) -> bool:
        self._exclude_target(node.target)
        return True

    def visit_GeneralAssignment(self, node: nodes.GeneralAssignment) -> bool:
        for tgt in node.targets:
            self._exclude_target(tgt)
        return True

    @override
    def visit_FunctionDef(self, node: libcst.FunctionDef) -> bool:
        del node
        return False

    @override
    def visit_ClassDef(self, node: libcst.ClassDef) -> bool:
        del node
        return False

    @override
    def visit_Lambda(self, node: libcst.Lambda) -> bool:
        del node
        return False

    @override
    def visit_Name(self, node: libcst.Name) -> bool:
        if node.value == self._target_name and id(node) not in self._exclude_ids:
            self.found = True
        return True


def _first_use_index_in_block(body: list[libcst.BaseStatement], target_name: str) -> int | None:
    """Return the first top-level statement index that loads `target_name`."""
    for idx, stmt in enumerate(body):
        finder = _BlockScopeNameLoadFinder(target_name)
        stmt.visit(finder)
        if finder.found:
            return idx
    return None


class _QuantizerCallDedup(libcst.CSTTransformer):
    """Deduplicate repeated quantizer calls inside a single indented block.

    The transformer looks for repeated calls of the shape
    ``self.quantizer*(...)`` and rewrites them to use a shared temporary
    variable with ``_tmp_`` prefix.

    The first occurrence determines the insertion point of the temporary
    assignment, and all occurrences are replaced with that temporary name.
    """

    def __init__(self) -> None:
        """Initialize per-transformer state for `_tmp_` name allocation."""
        self._created_temp_suffixes: set[int] = set()

    @staticmethod
    def _call_key(expr: libcst.BaseExpression) -> str | None:
        """Return a stable key for supported quantizer calls.

        During transformation, quantizer attributes may still carry temporary
        names before final name disambiguation. For ``QuantizerReference``
        attributes we key by ``refid`` to avoid merging distinct quantizers that
        currently share the same temporary textual name.
        """
        if not isinstance(expr, libcst.Call):
            return None
        if not isinstance(expr.func, libcst.Attribute):
            return None
        if not isinstance(expr.func.value, libcst.Name) or expr.func.value.value != "self":
            return None

        attr = expr.func.attr
        expr_code = libcst.Module([]).code_for_node(expr)
        if isinstance(attr, nodes.QuantizerReference):
            return f"quantizer_ref:{attr.refid}:{expr_code}"
        if not attr.value.startswith("quantizer"):
            return None

        return f"quantizer_name:{expr_code}"

    @staticmethod
    def _tmp_suffix(name: str) -> int | None:
        """Parse numeric suffix from `_tmp_<n>` names, or return ``None``."""
        if not name.startswith("_tmp_"):
            return None
        suffix = name[5:]
        if not suffix.isdigit():
            return None
        return int(suffix)

    def _next_tmp_name(self) -> str:
        """Allocate the next available `_tmp_<n>` variable name."""
        suffix = 1
        while suffix in self._created_temp_suffixes:
            suffix += 1
        self._created_temp_suffixes.add(suffix)
        return f"_tmp_{suffix}"

    @override
    def leave_IndentedBlock(
        self, original_node: libcst.IndentedBlock, updated_node: libcst.IndentedBlock
    ) -> libcst.IndentedBlock:
        """Rewrite a block so repeated quantizer calls are evaluated once.

        Steps:
        1. Collect existing `_tmp_` suffixes to avoid name collisions.
        2. Count quantizer call occurrences and record first occurrence index.
        3. For calls repeated more than once, allocate shared `_tmp_` names.
        4. Replace repeated call expressions with the corresponding temp name.
        5. Insert ``_tmp_`` assignments before the first statement using each
           repeated call.
        """
        del original_node

        body = list(updated_node.body)

        for stmt in body:

            class _TmpScan(libcst.CSTVisitor):
                def visit_Name(self, node: libcst.Name) -> bool:
                    suffix = _QuantizerCallDedup._tmp_suffix(node.value)
                    if suffix is not None:
                        self_suffixes.add(suffix)
                    return True

            self_suffixes = self._created_temp_suffixes
            stmt.visit(_TmpScan())

        first_stmt_idx: dict[str, int] = {}
        first_call_expr: dict[str, libcst.Call] = {}
        counts: dict[str, int] = {}

        for idx, stmt in enumerate(body):

            class _CallScan(libcst.CSTVisitor):
                # Nested suites/comprehensions are deduped independently when
                # their own block is visited. Do not mix scopes here.
                def visit_IndentedBlock(self, node: libcst.IndentedBlock) -> bool:
                    del node
                    return False

                def visit_ListComp(self, node: libcst.ListComp) -> bool:
                    del node
                    return False

                def visit_SetComp(self, node: libcst.SetComp) -> bool:
                    del node
                    return False

                def visit_DictComp(self, node: libcst.DictComp) -> bool:
                    del node
                    return False

                def visit_GeneratorExp(self, node: libcst.GeneratorExp) -> bool:
                    del node
                    return False

                def visit_Call(self, node: libcst.Call) -> bool:
                    key = _QuantizerCallDedup._call_key(node)
                    if key is None:
                        return True
                    counts[key] = counts.get(key, 0) + 1
                    if key not in first_stmt_idx:
                        first_stmt_idx[key] = idx
                        first_call_expr[key] = node
                    return True

            stmt.visit(_CallScan())

        repeated_keys = [k for k, c in counts.items() if c > 1]
        if not repeated_keys:
            return updated_node

        temp_by_call = {key: self._next_tmp_name() for key in repeated_keys}

        class _ReplaceCall(libcst.CSTTransformer):
            # Keep rewrites local to the current block; nested suites and
            # comprehensions are transformed on their own visit.
            def visit_IndentedBlock(self, node: libcst.IndentedBlock) -> bool:
                del node
                return False

            def visit_ListComp(self, node: libcst.ListComp) -> bool:
                del node
                return False

            def visit_SetComp(self, node: libcst.SetComp) -> bool:
                del node
                return False

            def visit_DictComp(self, node: libcst.DictComp) -> bool:
                del node
                return False

            def visit_GeneratorExp(self, node: libcst.GeneratorExp) -> bool:
                del node
                return False

            def leave_Call(
                self, original_node: libcst.Call, updated_node: libcst.Call
            ) -> libcst.BaseExpression:
                key = _QuantizerCallDedup._call_key(original_node)
                if key is not None and key in temp_by_call:
                    return libcst.Name(value=temp_by_call[key])
                return updated_node

        for i, stmt in enumerate(body):
            replaced = stmt.visit(_ReplaceCall())
            if not isinstance(replaced, libcst.BaseStatement):
                _fail_due_to_mismatch()
            body[i] = replaced

        inserts: list[tuple[int, libcst.SimpleStatementLine]] = []
        for key in repeated_keys:
            inserts.append((
                first_stmt_idx[key],
                libcst.SimpleStatementLine(
                    body=[
                        libcst.Assign(
                            targets=[
                                libcst.AssignTarget(target=libcst.Name(value=temp_by_call[key]))
                            ],
                            value=first_call_expr[key],
                        )
                    ]
                ),
            ))

        offset = 0
        for idx, stmt in sorted(inserts, key=lambda x: x[0]):
            body.insert(idx + offset, stmt)
            offset += 1

        return updated_node.with_changes(body=tuple(body))


def _dedup_repeated_quantizer_calls(node: libcst.IndentedBlock) -> libcst.IndentedBlock:
    """Apply `_QuantizerCallDedup` to one block and preserve type safety."""
    transformed = node.visit(_QuantizerCallDedup())
    if not isinstance(transformed, libcst.IndentedBlock):
        _fail_due_to_mismatch()
    return transformed


def _fail_due_to_mismatch() -> NoReturn:
    raise ValueError("Inconsistent annotations for original and updated node")
