# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from collections.abc import Iterator

from ..cst import nodes
from ..cst.node_processing import NormalizedAssignment, normalize_assignments
from . import blocks


def assignments_in_block(block: blocks.Block) -> Iterator[NormalizedAssignment]:
    """Yield all assignments in `block`.

    Args:
        block: The block for which to yield all assignments.

    Returns:
        Iterator over `NormalizedAssignment`s in `block`.
    """
    yield from block.visit(_AssignmentVisitor())


class _AssignmentVisitor:
    """Block visitor that yields all assignments in visited block."""

    def visit_FunctionBlock(self, _block: blocks.FunctionBlock) -> Iterator[NormalizedAssignment]:
        yield from ()

    def visit_IfBlock(self, _block: blocks.IfBlock) -> Iterator[NormalizedAssignment]:
        yield from ()

    def visit_ExitBlock(self, _block: blocks.ExitBlock) -> Iterator[NormalizedAssignment]:
        # Requires support for walrus operator. See #103
        yield from ()

    def visit_SimpleBlock(self, block: blocks.SimpleBlock) -> Iterator[NormalizedAssignment]:
        # This implementation does not take assignment expressions into
        # account. See #103.
        for statement_line in block.statements:
            for statement in statement_line.body:
                if isinstance(statement, nodes.GeneralAssignment):
                    yield from normalize_assignments(statement)
