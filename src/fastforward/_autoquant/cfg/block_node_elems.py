# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


from collections.abc import Iterator, Sequence
from typing import TypeAlias

import libcst

from typing_extensions import override

from ..cst import nodes
from ..cst.node_processing import NormalizedAssignment, normalize_assignments
from . import blocks

_ExprOrStatement: TypeAlias = libcst.SimpleStatementLine | libcst.BaseExpression
_AssignmentsReturnType: TypeAlias = Iterator[tuple[NormalizedAssignment, _ExprOrStatement]]


def assignments_in_block(block: blocks.Block) -> _AssignmentsReturnType:
    """Yield all assignments in `block`.

    Args:
        block: The block for which to yield all assignments.

    Returns:
        Iterator over `NormalizedAssignment`s in `block` and the statement the
        normalized assignment originated from.

    """
    for stmt_or_expr, node in extract_nodes_from_block(
        block, (nodes.GeneralAssignment, libcst.NamedExpr)
    ):
        match node:
            case nodes.GeneralAssignment():
                for normalized_assignment in normalize_assignments(node):
                    yield normalized_assignment, stmt_or_expr
            case libcst.NamedExpr:
                raise NotImplementedError("NamedExpr is not supported yet")


def _statements_and_expressions_in_block(block: blocks.Block) -> Iterator[_ExprOrStatement]:
    """Statements and expression in a block.

    Statements and expressions are yielded in order of execution.

    Arguments:
        block: The block for which statements and expressions are yielded.

    Returns:
        Iterator over statements and expression CST nodes in `block`.
    """
    yield from block.visit(_BlockStamementAndExpressions())


class _BlockStamementAndExpressions:
    """Block visitor that yields all assignments in visited block.

    Yields statements and expressions in order of execution.
    """

    def visit_FunctionBlock(self, _block: blocks.FunctionBlock) -> Iterator[_ExprOrStatement]:
        yield from ()

    def visit_IfBlock(self, block: blocks.IfBlock) -> Iterator[_ExprOrStatement]:
        yield block.test

    def visit_WhileBlock(self, block: blocks.WhileBlock) -> Iterator[_ExprOrStatement]:
        yield block.test

    def visit_ForBlock(self, block: blocks.ForBlock) -> Iterator[_ExprOrStatement]:
        yield block.iter
        # Note: target is yielded here. Outside of the context of a `ForBlock`
        #       or `libcst.For` it is not clear that this is an assignment.
        #       Assignments require special handling.
        yield block.target

    def visit_ExitBlock(self, _block: blocks.ExitBlock) -> Iterator[_ExprOrStatement]:
        yield from ()

    def visit_SimpleBlock(self, block: blocks.SimpleBlock) -> Iterator[_ExprOrStatement]:
        yield from block.statements

    def visit_WithBlock(self, block: blocks.WithBlock) -> Iterator[_ExprOrStatement]:
        yield from (with_item.item for with_item in block.items)

    def visit_MarkerBlock(self, _block: blocks.MarkerBlock) -> Iterator[_ExprOrStatement]:
        yield from ()


def extract_nodes_from_block(
    block: blocks.Block,
    node_types: Sequence[type[libcst.CSTNode]],
    include_subclasses: bool = False,
) -> Iterator[tuple[_ExprOrStatement, libcst.CSTNode]]:
    """Iterate over all CST nodes in `block` that are of `node_types`.

    Args:
        block: The CFG block to inspect.
        node_types: Sequence of CST node types to extract.
        include_subclasses: If `True`, also extract subclasses of elements in
            `node_types`.

    Returns:
        Iterator over CST Nodes that appear in `block` which type (or
        supertype) is included in `node_types`.
    """
    for stmt_or_expr in _statements_and_expressions_in_block(block):
        visitor = _ExtractionVisitor(node_types, include_subclasses=include_subclasses)
        stmt_or_expr.visit(visitor)
        for extract_node in visitor.extracted_nodes():
            yield stmt_or_expr, extract_node


class _ExtractionVisitor(libcst.CSTVisitor):
    """Visitor that implements logic for `extract_nodes_from_block` for each `Block`.

    After visiting a CST, `extracted_nodes` contains all nodes in the CST that match
    `node_types` following `include_subclasses`.

    Args:
        node_types: Sequence of CST node types to extract.
        include_subclasses: If `True`, also extract subclasses of elements in
            `node_types`.
    """

    def __init__(
        self, node_types: Sequence[type[libcst.CSTNode]], include_subclasses: bool
    ) -> None:
        self._node_types = tuple(node_types)
        self._extracted_nodes: list[libcst.CSTNode] = []
        self._include_subclasses = include_subclasses

    def _should_extract_node(self, node: libcst.CSTNode) -> bool:
        if self._include_subclasses:
            return isinstance(node, self._node_types)
        else:
            return type(node) in self._node_types

    @override
    def on_leave(self, original_node: libcst.CSTNode) -> None:
        if self._should_extract_node(original_node):
            self._extracted_nodes.append(original_node)
        super().on_leave(original_node)

    def extracted_nodes(self) -> Iterator[libcst.CSTNode]:
        for node in self._extracted_nodes:
            yield node
