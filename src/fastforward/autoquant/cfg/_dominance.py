# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import collections

from collections.abc import Sequence
from typing import NewType, Protocol, TypeVar

from . import blocks

OrderIndex = NewType("OrderIndex", int)


def infer_immediate_dominators(root: blocks.Block) -> None:
    predecessors = infer_predecessors(root)
    block_order = {block: OrderIndex(idx) for idx, block in enumerate(root.blocks())}

    # Set entry block to its own immediate dominator for the duration
    # of this function. This removes None-checks from the algorithm.
    root.immediate_dominator = root

    try:
        _assign_immediate_dominators(
            root, parents, block_order, _DominatorAccessor(), reverse_traversal=True
        )
    finally:
        root.immediate_dominator = None

    children = infer_successors(root)
    block_order = {block: OrderIndex(idx) for idx, block in enumerate(root.blocks(reverse=False))}
    exit_block = next(root.blocks(reverse=False))
    exit_block.immediate_post_dominator = exit_block
    try:
        _assign_immediate_dominators(
            root=root,
            parents=children,
            block_order=block_order,
            dominator_accessor=_PostDominatorAccessor(),
            reverse_traversal=False,
        )
    finally:
        exit_block.immediate_post_dominator = None


def infer_predecessors(root: blocks.Block) -> dict[blocks.Block, list[blocks.Block]]:
    predecessors: dict[blocks.Block, list[blocks.Block]] = collections.defaultdict(list)
    for member in root.blocks():
        for _, child in member.named_children():
            parents[child].append(member)
    return parents


def infer_successors(root: blocks.Block) -> dict[blocks.Block, list[blocks.Block]]:
    predecessors: dict[blocks.Block, list[blocks.Block]] = collections.defaultdict(list)
    for member in root.blocks():
        for _, child in member.named_children():
            predecessors[member].append(child)
    return predecessors


def _assign_immediate_dominators(
    root: blocks.Block,
    predecessors: dict[blocks.Block, list[blocks.Block]],
    block_order: dict[blocks.Block, OrderIndex],
    dominator_accessor: "_DominatorAccessorProtocol",
    reverse_traversal: bool = True,
) -> None:
    changed = True
    while changed:
        iter = root.blocks(reverse=reverse_traversal)
        _ = next(iter)  # skip the function/entry block
        for block in iter:
            precs = predecessors[block]
            idom = _processed_predecessor(precs, dominator_accessor)
            for prec in precs:
                if prec is idom:
                    continue
                if dominator_accessor.get(prec):
                    idom = _most_immediate_common_dominator(
                        parent, idom, block_order, dominator_accessor
                    )
            changed = idom != dominator_accessor.get(block)
            dominator_accessor.set(block, idom)


def _most_immediate_common_dominator(
    block1: blocks.Block,
    block2: blocks.Block,
    block_order: dict[blocks.Block, OrderIndex],
    dominator_accessor: "_DominatorAccessorProtocol",
) -> blocks.Block:
    finger1 = block1
    finger2 = block2

    while finger1 != finger2:
        fidx1 = block_order[finger1]
        fidx2 = block_order[finger2]
        while fidx2 < fidx1:
            finger1 = _not_none(dominator_accessor.get(finger1))
            fidx1 = block_order[finger1]
        while fidx1 < fidx2:
            finger2 = _not_none(dominator_accessor.get(finger2))
            fidx2 = block_order[finger2]

    return finger1


_T = TypeVar("_T")


def _processed_parent(
    parents: Sequence[blocks.Block],
    dominator_accessor: "_DominatorAccessorProtocol",
) -> blocks.Block:
    """
    Returns any parent that has an immediate dominator set.

    NB: during dominance assignment, this may not be the actual
        immediate dominator
    """
    for prec in parents:
        if dominator_accessor.get(prec):
            return prec
    raise RuntimeError("Block has no processed parents")


class _DominatorAccessorProtocol(Protocol):
    def get(self, block: blocks.Block) -> blocks.Block | None: ...
    def set(self, block: blocks.Block, dominator: blocks.Block) -> None: ...


class _DominatorAccessor:
    def get(self, block: blocks.Block) -> blocks.Block | None:
        return block.immediate_dominator

    def set(self, block: blocks.Block, dominator: blocks.Block) -> None:
        block.immediate_dominator = dominator


class _PostDominatorAccessor:
    def get(self, block: blocks.Block) -> blocks.Block | None:
        return block.immediate_post_dominator

    def set(self, block: blocks.Block, dominator: blocks.Block) -> None:
        block.immediate_post_dominator = dominator
