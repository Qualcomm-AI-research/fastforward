# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import functools

from typing import overload

import libcst

from ..cst import node_manipulation
from ..cst.validation import ensure_type
from . import blocks
from .exceptions import CFGReconstructionError


def reconstruct(block: blocks.FunctionBlock) -> libcst.FunctionDef:
    """Convert a `FunctionBlock` into a CST.

    Args:
        block: The `FunctionBlock` to convert to a CST.

    Returns:
        CST that represents same code as the CFG with `block` as root.
    """
    wrappers = block.wrappers
    if len(wrappers) != 1 or not isinstance(wrappers[0], libcst.FunctionDef):
        raise CFGReconstructionError(
            "Expected a single FunctionDef node associated with the function block"
        )

    body_node = block.body.visit(_BlockReconstructor())
    node = wrappers[0].with_changes(body=body_node)
    assert isinstance(node, libcst.FunctionDef)
    return node


class _BlockReconstructor:
    """Visitor used for constructing CSTs from CFGs.

    This visitor cannot visit `FunctionBlock`s. Instead, for a given CFG with a
    `FunctionBlock` root it should be applied to its `body` attribute.

    Each visit function returns an IndentedBlock and reconstructs the CST for all
    blocks that are dominated by the input block.
    """

    def visit_FunctionBlock(self, _block: blocks.FunctionBlock) -> libcst.IndentedBlock:
        # FunctionBlock is the only block that the we don't reconstruct as an
        # IndentedBlock and is dealt with outside of the BlockConcstructor.
        # If we encounter a `FunctionBlock` as part of the CFG, it is considered
        # an error.
        raise RuntimeError(
            f"Encountered 'FunctionBlock' as a non root-node of a CFG or {type(self).__name__} "
            + "was passed to 'FunctionBlock.visit'. Both are not allowed."
        )

    def visit_ExitBlock(self, _block: blocks.ExitBlock) -> libcst.IndentedBlock:
        """Construct a CST from an `ExitBlock`.

        Since the exit block does not contain any logic this function simply
        returns an empty `IndentedBlock`.

        Args:
            block: `Block` to convert into a CST.

        Returns:
                CST representation of `block` with `IndentedBlock` as root node.
        """
        return libcst.IndentedBlock(body=())

    def visit_IfBlock(self, block: blocks.IfBlock) -> libcst.IndentedBlock:
        """Convert an `IfBlock` into an `IndentedBlock`.

        Args:
            block: `Block` to convert into a CST.

        Returns:
                CST representation of `block` with `IndentedBlock` as root node.
        """
        if block.false is None:
            raise ValueError("Encountered IfBlock with dangling 'else' edge")

        true_branch = block.true.visit(self)
        false_branch: libcst.If | libcst.IndentedBlock | None = None
        if block.immediate_post_dominator is not block.false:
            false_branch = block.false.visit(self)
            false_branch = node_manipulation.unwrap_single_statement(false_branch, libcst.If)
            false_branch = _apply_wrappers(false_branch, block.false)  # type: ignore[assignment]
            assert isinstance(false_branch, (libcst.If, libcst.Else)), type(false_branch).__name__

        if_node = block.wrappers[0].with_changes(
            body=true_branch, orelse=false_branch, test=block.test
        )
        if_node = ensure_type(if_node, libcst.If, CFGReconstructionError)
        node = libcst.IndentedBlock(body=(if_node,))

        return self._process_tail(block, node)

    def visit_ForBlock(self, block: blocks.ForBlock) -> libcst.IndentedBlock:
        """Convert a `ForBlock` into an `IndentedBlock`.

        Args:
            block: `Block` to convert into a CST.

        Returns:
                CST representation of `block` with `IndentedBlock` as root node.
        """
        body = block.body.visit(self)
        for_node = block.wrappers[0].with_changes(body=body, target=block.target, iter=block.iter)
        for_node = ensure_type(for_node, libcst.For, CFGReconstructionError)
        node = libcst.IndentedBlock(body=(for_node,))

        return self._process_tail(block, node)

    def visit_WhileBlock(self, block: blocks.WhileBlock) -> libcst.IndentedBlock:
        """Convert a `WhileBlock` into an `IndentedBlock`.

        Args:
            block: `Block` to convert into a CST.

        Returns:
                CST representation of `block` with `IndentedBlock` as root node.
        """
        body = block.body.visit(self)
        while_node = block.wrappers[0].with_changes(body=body, test=block.test)
        while_node = ensure_type(while_node, libcst.While, CFGReconstructionError)
        node = libcst.IndentedBlock(body=(while_node,))

        return self._process_tail(block, node)

    def visit_WithBlock(self, block: blocks.WithBlock) -> libcst.IndentedBlock:
        """Convert a `WithBlock` into an `IndentedBlock`.

        Args:
            block: `Block` to convert into a CST.

        Returns:
                CST representation of `block` with `IndentedBlock` as root node.
        """
        body = block.body.visit(self)
        with_node = block.wrappers[0].with_changes(body=body, items=block.items)
        with_node = ensure_type(with_node, libcst.With, CFGReconstructionError)
        node = libcst.IndentedBlock(body=(with_node,))

        if with_terminator := _find_with_terminator_block(block):
            return self._process_tail(with_terminator, node)
        else:
            msg = (
                "Inconsistent CFG. CFG contains a 'WithBlock' without a "
                + "corresponding terminator 'MarkerBlock'"
            )
            raise CFGReconstructionError(msg)

    def visit_MarkerBlock(self, block: blocks.MarkerBlock) -> libcst.IndentedBlock:
        """Skip marker block and continue reconstruction."""
        if (next_block := block.next_block) is not None:
            return next_block.visit(self)
        msg = "Encountered a dangling MarkerBlock"
        raise CFGReconstructionError(msg)

    def visit_SimpleBlock(self, block: blocks.SimpleBlock) -> libcst.IndentedBlock:
        """Convert a `SimpleBlock` into an `IndentedBlock`.

        Args:
            block: `Block` to convert into a CST.

        Returns:
                CST representation of `block` with `IndentedBlock` as root node.
        """
        node = block.wrappers[0].with_changes(body=block.statements)
        if not isinstance(node, libcst.IndentedBlock):
            raise TypeError(
                f"Expected 'root' wrapper of SimpleBlock to be IndentedBlock, got {type(node).__name__}"
            )

        # Check if block.next is dominated by block. If true, merge CSTs from block
        # and block.next.
        return self._process_tail(block, node)

    def _process_tail(
        self, block: blocks.Block, node: libcst.IndentedBlock
    ) -> libcst.IndentedBlock:
        """Process the 'tail' of `block` and merge with node if appropriate.

        Consider a `Block` `B` and let its immediate post-dominator be `P`. `A` is
        considered to have a "tail" iff `A` is the immediate post-dominator of `P`.
        In this case, control flow can only reach `P` through `A`. In terms of CST
        reconstruction, this means that the `IndentedBlock` associated with `A` and
        `P` can be merged into a single `IndentedBlock`.

        This function merges `node` (the `IndentedBlock` associated with `block`,
        i.e., `A` in the description above) and the associated (to be constructed)
        `IndentedBlock` associated with `blocks`s immediate post-dominator if the
        requirements described above hold. Otherwise, this function returns `node`.

        Args:
            block: the `Block` to evaluate.
            node: the `IndentedBlock` associated with `block`.

        Returns:
            `IndentedBlock` that is either `node` or `node` merged with the
            `IndentedBlock` constructed from `block.immediate_post_dominator`.
        """
        if block.immediate_post_dominator and _has_tail(block):
            tail = block.immediate_post_dominator.visit(self)
            node = node_manipulation.combine_indented_blocks(node, tail)
        return node


def _find_with_terminator_block(
    block: blocks.Block, *, _depth: int = 0
) -> blocks.MarkerBlock | None:
    with_depth = _depth
    match block:
        case blocks.WithBlock():
            with_depth += 1
        case blocks.MarkerBlock(marker=blocks.MarkerType.WithBlockTerminator):
            with_depth -= 1
            if with_depth == 0:
                return block

    for child in block.children():
        if terminator := _find_with_terminator_block(child, _depth=with_depth):
            return terminator

    return None


def _has_tail(block: blocks.Block) -> bool:
    """Checks if `block` has a 'tail' block.

    A tail block should be processed to be part of the same `IndentedBlock` as
    that associated with `block`.

    For example, consider the following code:

    ```python
    if a > b:
        print("a is bigger than b")
    print("end program")
    ```

    In this example, the final print statement should be in the same
    `IndentedBlock` as the if statement. It can be identified because it will
    have the `Block` corresponding to the if-statement as its dominator. From
    the if-statement block, the block corresponding to the block of the last
    print statement will be its immediate post-dominator. We can use this
    information to identify if there are further blocks that should be
    processed or if blocks that can be reached from `block` are part of another
    sub-tree of the CST.

    Args:
        block: the `Block` to evaluate.

    Returns:
        boolean indicating if immediate post-dominator should be processed as
        part of CST sub-tree associated with `block`.
    """
    if block.immediate_post_dominator is None:
        raise RuntimeError(
            f"Encountered {type(block).__name__} with dangling immediate_post_dominator"
        )
    post_dom = block.immediate_post_dominator

    # Syntactic block boundaries are not explicitly represented in the CFG.
    # However, the notion of an ending block is important for CST
    # reconstruction (e.g., for `with` statements). To support this, the CFG
    # includes `MarkerBlock` nodes that signify the end of such blocks. A
    # `MarkerBlock` is considered a terminator if `is_terminator` evaluates to
    # True, typically when it post-dominates a block. In such cases, the block
    # has no tail in terms of CFG-to-CST reconstruction, and therefore this
    # function will return False.

    is_terminating_marker = isinstance(post_dom, blocks.MarkerBlock) and post_dom.is_terminator
    return post_dom.immediate_dominator is block and not is_terminating_marker


@overload
def _apply_wrappers(node: None, block: blocks.Block) -> None: ...


@overload
def _apply_wrappers(node: libcst.CSTNode, block: blocks.Block) -> libcst.CSTNode: ...


def _apply_wrappers(node: libcst.CSTNode | None, block: blocks.Block) -> libcst.CSTNode | None:
    if node is None:
        return None
    for wrapper in block.wrappers[1:]:
        node = apply_node_wrapper(wrapper, block, node)
    return node


# Wrapper handlers


@functools.singledispatch
def apply_node_wrapper(
    _wrapper: libcst.CSTNode,
    _block: blocks.Block,
    _node: libcst.CSTNode,
    /,
) -> libcst.CSTNode:
    """Dispatcher for wrapper application functions.

    Wrappers on `Blocks` are CST nodes that may carry extra information that is
    not contained in the block itself. During the deconstruction process, these
    wrappers are applied again. This is a general function for applying
    wrappers. Node specific functions can be registered using
    `apply_node_wrapper.register`.

    New specializations for wrapper applications can be registered using the
    `apply_node_wrapper.register` decorator.

    Args:
        _wrapper: The CST node to apply or unwrap.
        _block: The block of which the wrapper is stored.
        _node: The CST node to which to apply the wrapper to.

    Returns:
        Processed and 'unwrapped' CST Node.

    Note:
        `Apply` is used in a very general sense. I.e., it simply means: take a
        wrapper node and a reconstructed node and update the reconstructed node
        or create a new node such that the relevant information from the
        wrapper node is contained in the reconstructed CST. The actual
        implementation is heavily wrapper, block, and node dependent.
    """
    raise RuntimeError(
        f"There is no known handler for applying a {type(_wrapper).__name__}. "
        + "You can register one using apply_node_wrapper.register."
    )


@apply_node_wrapper.register
def apply_else_wrapper(
    wrapper: libcst.Else,
    block: blocks.SimpleBlock,
    node: libcst.CSTNode,
    /,
) -> libcst.Else | libcst.If:
    """Turn `node` into an `Else` node.

    If node is an `If` node, wrap it in an `IndentedBlock` before further
    processing. This ensures that an else branch is present in the produced CST
    if it was in the original.

    Raises an error if `node` is not an `IndentedBlock` or `If`.

    Args:
        wrapper: The CST node to apply or unwrap.
        block: The block of which the wrapper is stored.
        node: The CST node to which to apply the wrapper to.

    Returns:
        Processed and 'unwrapped' CST Node.
    """
    if isinstance(node, libcst.If):
        node = libcst.IndentedBlock(body=(node,))
    if isinstance(node, libcst.IndentedBlock):
        return wrapper.with_changes(body=node)

    raise ValueError(
        f"Cannot unwrap Else node for {type(block).__name__} that produces {type(node).__name__}"
    )


@apply_node_wrapper.register
def apply_indented_block_wrapper(
    wrapper: libcst.IndentedBlock,
    _block: blocks.SimpleBlock,
    node: libcst.CSTNode,
    /,
) -> libcst.CSTNode:
    """Apply an `IndentedBlock` wrapper.

    If `node` is an `IndentedBlock`, update the metadata (but not body) on
    `node` such that the information from `wrapper` is available on `node`.
    Otherwise, create a new `IndentedBlock` from `wrapper` where body equals
    `node`.

    Args:
        wrapper: The CST node to apply or unwrap.
        _block: The block of which `wrapper` is stored.
        node: The CST node to which to apply `wrapper` to.

    Returns:
        Processed and 'unwrapped' CST Node.
    """
    match node:
        case libcst.IndentedBlock():
            return node_manipulation.flatten_indented_blocks(wrapper, node)
        case libcst.BaseCompoundStatement() | libcst.SimpleStatementLine():
            return wrapper.with_changes(body=(node,))
        case _:
            return node
