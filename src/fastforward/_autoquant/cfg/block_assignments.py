# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from collections.abc import Iterator

from ..cst import nodes
from ..cst.node_processing import NormalizedAssignment, normalize_assignments
from . import block_node_elems, blocks


def assignments_in_block(block: blocks.Block) -> Iterator[NormalizedAssignment]:
    """Yield all assignments in `block`.

    Args:
        block: The block for which to yield all assignments.

    Returns:
        Iterator over `NormalizedAssignment`s in `block`.
    """
    for assignment, _ in block_node_elems.extract_nodes_from_block(
        block, (nodes.GeneralAssignment,)
    ):
        assert isinstance(assignment, nodes.GeneralAssignment)
        yield from normalize_assignments(assignment)
