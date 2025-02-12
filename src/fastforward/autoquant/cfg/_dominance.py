# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import collections

from collections.abc import Sequence
from typing import NewType, Protocol, TypeVar

from . import blocks

OrderIndex = NewType("OrderIndex", int)


def set_immediate_dominators(root: blocks.Block) -> None:
    """Infer immediate dominators and post-dominators in CFG.

    A dominator of a block A is a block B which is guaranteed to be on every
    path between block A and the entry node of the CFG. Similarly, a
    post-dominator of block A is a block B which is guaranteed to be on every
    path between block A and the exit block. The immediate (post-)dominator of
    block A is the unique block B that strictly dominates block A but does not
    strictly dominate any other block C that also dominates A. A block always
    dominates itself and all dominators of a block A that dominates block B are
    also dominators
    of B.

    This function infers the immediate (post-)dominators of each block in the
    CFG and assigns these to the `immediate_dominator` and
    `immediate_post_dominator` of each block respectively. The set of
    (post-)dominators for each block can be obtained by traversing the
    immediate dominator edges.

    The `immediate_dominator` of the entry block and the
    `immediate_post_dominator` of the exit block are `None` after this function
    terminates, indicating that the block has no immediate (post-)dominator.
    The set of dominators for this block, however, contains the block itself.

    This implementation follows the method described in [1]

    [1] "A Simple, Fast Dominance Algorithm" by Cooper, Harvey, Kennedy,
        https://www.cs.tufts.edu/~nr/cs257/archive/keith-cooper/dom14.pdf

    Args:
        root: The root block of the graph.
    """
    parents = infer_parents(root)
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

    # Do the same thing as for dominators, but in reverse. I.e., use the exit
    # block of the graph as the root element and children instead of
    # parents. This will infer the immediate post-dominators.
    children = infer_children(root)
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


def infer_parents(root: blocks.Block) -> dict[blocks.Block, list[blocks.Block]]:
    """Infer parents for each block in CFG given by `root`.

    Args:
        root: The root block of the CFG.

    Returns:
        A dictionary mapping a block to a list of blocks that are its parents.
    """
    parents: dict[blocks.Block, list[blocks.Block]] = collections.defaultdict(list)
    for member in root.blocks():
        for _, child in member.named_children():
            parents[child].append(member)
    return parents


def infer_children(root: blocks.Block) -> dict[blocks.Block, list[blocks.Block]]:
    """Infer children for each block in CFG given by `root`.

    Note that the children are always stored on the `block` directly. This
    function merely produces a dictionary that maps blocks to their children
    for use in methods that require such mapping.

    Args:
        root: The root block of the CFG.

    Returns:
        A dictionary mapping a block to a list of blocks that are its children.
    """
    children: dict[blocks.Block, list[blocks.Block]] = collections.defaultdict(list)
    for member in root.blocks():
        for _, child in member.named_children():
            children[member].append(child)
    return children


def _assign_immediate_dominators(
    root: blocks.Block,
    parents: dict[blocks.Block, list[blocks.Block]],
    block_order: dict[blocks.Block, OrderIndex],
    dominator_accessor: "_DominatorAccessorProtocol",
    reverse_traversal: bool,
) -> None:
    """Infer dominator for each block in CFG given by `root`.

    This function implements an iterative algorithm for inferring dominators.
    After convergence the dominator is set correctly, however, during the
    runtime of this function, a dominator may be assigned that is not the
    immediate dominator.

    This function follows the method introduced in [1].

    [1] "A Simple, Fast Dominance Algorithm" by Cooper, Harvey, Kennedy,
        https://www.cs.tufts.edu/~nr/cs257/archive/keith-cooper/dom14.pdf

    Args:
        root: the root block of the CFG for which to infer the immediate
            dominators of.
        parents: A mapping from block to a list of parents.
        block_order: Order for blocks in the CFG. Parents are expected to
            appear before their children in the order.
        dominator_accessor: Accessor that determines which dominator is
            retrieved and set.
        reverse_traversal: Boolean indicating if blocks are iterated over in
            reverse order.

    Note:
        The arguments of this function are written from an 'immediate
        dominator' inference perspective. However, by passing the appropriate
        `parents`, `block_order`, `dominator_accessor` and
        `reverse_traversal` it can also be used to infer 'immediate
        post-dominators'. See the usage in `set_immediate_dominators` for an
        example.
    """
    changed = True

    # Loop until convergence
    while changed:
        block_iter = root.blocks(reverse=reverse_traversal)
        _ = next(block_iter)  # skip the function/entry block
        for block in block_iter:
            block_parents = parents[block]

            # Set immediate dominator to any parent that has its dominator
            # field set.
            idom = _processed_parent(block_parents, dominator_accessor)
            for parent in block_parents:
                if parent is idom:
                    continue
                if dominator_accessor.get(parent):
                    # Update immediate dominator to the immediate dominator of
                    # both the current immediate dominator and the parent.
                    # This can be idom, parent, or another block.
                    idom = _most_immediate_common_dominator(
                        parent, idom, block_order, dominator_accessor
                    )
            # Update `changed` if immediate dominator for `block` is updated in
            # this iteration.
            changed = idom != dominator_accessor.get(block)
            dominator_accessor.set(block, idom)


def _most_immediate_common_dominator(
    block1: blocks.Block,
    block2: blocks.Block,
    block_order: dict[blocks.Block, OrderIndex],
    dominator_accessor: "_DominatorAccessorProtocol",
) -> blocks.Block:
    """Find lowest common dominator for `block1` and `block2`.

    The dominator selection is given by `dominator_accessor`, making this
    function usable for both normal and post-dominators.

    Args:
        block1: the first block.
        block2: the second block.
        block_order: An ordering of all blocks in the graph of which `block1`
            and `block2` are member of. This ordering determines the closeness
            of the blocks to possible dominators. It is assumed that parents
            appear before their children in the block order, for any definition
            of parents/children.
        dominator_accessor: the accessor that is used to obtain the dominator
            for a given block. This can, for example, be an accessor that retrieves
            the dominator or post-dominator of a given block.

    Returns:
        A block B that dominates `block1` and `block2` such that there is no
        other block that dominates `block1` and `block2` but not B.
    """
    finger1 = block1
    finger2 = block2

    def _not_none(__obj: _T | None) -> _T:
        if __obj is None:
            raise RuntimeError(f"{__obj} is expected to be not None")
        return __obj

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
    """Returns any parent that has an immediate dominator set.

    NB: during dominance assignment, this may not be the actual
        immediate dominator
    """
    for prec in parents:
        if dominator_accessor.get(prec):
            return prec
    raise RuntimeError("Block has no processed parents")


class _DominatorAccessorProtocol(Protocol):
    """Protocol for dominator accessors.

    A dominator accessor returns a specific type of dominator, e.g., immediate
    dominator or immediate post-dominator, such that algorithm implementation
    gen be generic over a specific type of dominator.
    """

    def get(self, block: blocks.Block) -> blocks.Block | None:
        """Return the dominator of `block`.

        Args:
            block: The block to obtain the dominator from.

        Returns:
            The block that is set as `block`s dominator.
        """

    def set(self, block: blocks.Block, dominator: blocks.Block) -> None:
        """Set the dominator of `block` to `dominator`.

        Args:
            block: The block to set the dominator of.
            dominator: The block that should be set as `block`s dominator.
        """


class _DominatorAccessor:
    """A dominator accessor for  immediate dominators."""

    def get(self, block: blocks.Block) -> blocks.Block | None:
        return block.immediate_dominator

    def set(self, block: blocks.Block, dominator: blocks.Block) -> None:
        block.immediate_dominator = dominator


class _PostDominatorAccessor:
    """A dominator accessor for  immediate post-dominators."""

    def get(self, block: blocks.Block) -> blocks.Block | None:
        return block.immediate_post_dominator

    def set(self, block: blocks.Block, dominator: blocks.Block) -> None:
        block.immediate_post_dominator = dominator
