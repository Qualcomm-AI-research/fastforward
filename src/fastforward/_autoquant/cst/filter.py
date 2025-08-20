# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import collections

from collections.abc import Iterator

import libcst

from typing_extensions import overload


@overload
def filter_nodes_by_type(
    tree: libcst.CSTNode,
    needle: type[libcst.CSTNodeT],
) -> Iterator[libcst.CSTNodeT]: ...


@overload
def filter_nodes_by_type(
    tree: libcst.CSTNode,
    needle: tuple[type[libcst.CSTNode], ...],
) -> Iterator[libcst.CSTNode]: ...


def filter_nodes_by_type(
    tree: libcst.CSTNode,
    needle: type[libcst.CSTNodeT] | tuple[type[libcst.CSTNode], ...],
) -> Iterator[libcst.CSTNodeT] | Iterator[libcst.CSTNode]:
    """Find all nodes in `tree` that are of type `needle`.

    The tree is traversed in a breadth first order. Nodes are yielded in order
    of appearance in this order.
    """
    seen_nodes: set[libcst.CSTNode] = {tree}
    frontier_nodes: collections.deque[libcst.CSTNode] = collections.deque([tree])

    while frontier_nodes:
        node = frontier_nodes.popleft()
        if isinstance(node, needle):
            yield node
        for child in node.children:
            if child in seen_nodes:
                continue
            seen_nodes.add(child)
            frontier_nodes.append(child)
