# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import functools
import logging

from typing import overload

import libcst

from ..cst import node_manipulation
from ..cst.validation import ensure_type
from . import blocks
from .exceptions import CFGReconstructionError

logger = logging.getLogger(__name__)


def reconstruct(block: blocks.FunctionBlock) -> libcst.FunctionDef:
    """
    Convert a `FunctionBlock` into a CST.

    Args:
        block: The `FunctionBlock` to convert to a CST.

    Returns:
        CST that represents same code as the CFG with `block` as root.
    """
    logger.debug("start reconstruct CFG")
    wrappers = block.wrappers
    if len(wrappers) != 1 or not isinstance(wrappers[0], libcst.FunctionDef):
        raise CFGReconstructionError(
            "Expected function block to have exactly one FunctionDef associated node"
        )

    node = wrappers[0].with_changes(body=reconstruct_block(block.body))
    assert isinstance(node, libcst.FunctionDef)
    return node


@functools.singledispatch
def reconstruct_block(block: blocks.Block) -> libcst.IndentedBlock:
    """
    Convert `block` into an `IndentedBlock`.

    Specializations for this function that accept a subclass of `blocks.Block`
    can be registered using `reconstruct_block.register`. Each of these
    functions is expected to accept a single block as input and construct a CST
    of all the blocks dominated by the input block. The constructed CST root
    element must be an `IndentedBlock`.

    Args:
        block: `Block` to convert into a CST.

    Returns:
            CST representation of `block` with `IndentedBlock` as root node.
    """
    raise TypeError(
        f"No reconstruction function is registered for {type(block).__name__}. "
        + "Please file an issue when you run into this error."
    )


@reconstruct_block.register
def reconstruct_SimpleBlock(block: blocks.SimpleBlock) -> libcst.IndentedBlock:
    """
    Convert a `SimpleBlock` into an `IndentedBlock`.

    A `SimpleBlock` has a list of statements and a 'root' wrapper that must be
    an `IndentedBlock`. The CST is reconstructed by replacing the wrapper's
    body with the block's statements.

    If the block pointed to by `next_block` is dominated by `block` then the
    `IndentedBlock` created from `block` is merged with the `IndentedBlock`
    created from `block.next_block`.

    Args:
        block: `Block` to convert into a CST.

    Returns:
            CST representation of `block` with `IndentedBlock` as root node.
    """
    logger.debug("Reconstruct SimpleBlock")
    node = block.wrappers[0].with_changes(body=block.statements)
    if not isinstance(node, libcst.IndentedBlock):
        raise TypeError(
            f"Expected 'root' wrapper of SimpleBlock to be IndentedBlock, got {type(node).__name__}"
        )

    # Check if block.next_block is dominated by block. If true, merge CSTs from block
    # and block.next_block.
    return _process_tail(block, node)


@reconstruct_block.register
def reconstruct_IfBlock(block: blocks.IfBlock) -> libcst.IndentedBlock:
    """
    Convert an `IfBlock` into an `IndentedBlock`.

    An `IfBlock` contains a test expression and references blocks for the true
    and false branch. The body of the `libcst.If` is made up of the
    `IndentedBlock` that is created by converting the `true` block to and
    `IndentedBlock`.

    If `block.false` is equal to the post-dominator of block, there is no
    specific control flow for the else branch, hence, else branch is created.
    Otherwise, the else branch is created from the block pointed to by
    `block.false`. In the case where the `IndentedBlock` of the `false` branch
    only contains a single `libcst.If` node, the `libcst.If` is used directly
    as an elif branch, if no `libcst.Else` wrapper node is present.

    If the immediate post-dominator of `block` is dominated by `block`, its
    reconstructed `IndentedBlock` is merged with that of `block`.

    Args:
        block: `Block` to convert into a CST.

    Returns:
            CST representation of `block` with `IndentedBlock` as root node.
    """
    logger.debug("Reconstruct IfBlock")
    if block.false is None:
        raise ValueError("Encountered IfBlock withing dangling 'else' edge")

    true_branch = reconstruct_block(block.true)
    false_branch: libcst.If | libcst.IndentedBlock | None = None
    if block.immediate_post_dominator is not block.false:
        logger.debug(f"Reconstruct False branch from {type(block.false).__name__}")
        false_branch = reconstruct_block(block.false)
        false_branch = node_manipulation.unwrap_single_statement(false_branch, libcst.If)
        false_branch = _apply_wrappers(false_branch, block.false)  # type: ignore[assignment]
        assert isinstance(false_branch, (libcst.If, libcst.Else)), type(false_branch).__name__

    if_node = block.wrappers[0].with_changes(body=true_branch, orelse=false_branch, test=block.test)
    if_node = ensure_type(if_node, libcst.If, CFGReconstructionError)
    node = libcst.IndentedBlock(body=(if_node,))

    return _process_tail(block, node)


@reconstruct_block.register
def reconstruct_ExitBlock(_block: blocks.ExitBlock) -> libcst.IndentedBlock:
    """
    Construct a CST from an `ExitBlock`.

    Since the exit block does not contain any logic this function simply
    returns an empty `IndentedBlock`.

    Args:
        block: `Block` to convert into a CST.

    Returns:
            CST representation of `block` with `IndentedBlock` as root node.
    """
    logger.debug("Reconstruct ExitBlock")
    return libcst.IndentedBlock(body=())


def _process_tail(block: blocks.Block, node: libcst.IndentedBlock) -> libcst.IndentedBlock:
    """
    Merge `IndentedBlock` associated with `block` if the immediate
    post-dominator of `block` is dominated by `block`.

    Args:
        block: the `Block` to evaluate.
        node: the `IndentedBlock` associated with `block`.

    Returns:
        `IndentedBlock` that is either `node` or `node` merged with the
        `IndentedBlock` constructed from `block.immediate_post_dominator`.
    """
    if block.immediate_post_dominator and _has_tail(block):
        tail = reconstruct_block(block.immediate_post_dominator)
        node = node_manipulation.combine_indented_blocks(node, tail)
    return node


def _has_tail(block: blocks.Block) -> bool:
    """
    Checks if `block` has a 'tail' block that should be processed as part of
    the same `IndentedBlock`.

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
    return block.immediate_post_dominator.immediate_dominator is block


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
    """
    Wrappers on `Blocks` are CST nodes that may carry extra information that is
    not contained in the block itself. During the deconstruction process, these
    wrappers are applied again. This is a general function for applying
    wrappers. Node specific functions can be registered using
    `apply_node_wrapper.register`

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
    """
    Turn `node` into an `Else` node.

    If node is an `If` node. Wrap it in an `IndentedBlock` before further
    processing. This ensures that an else branch is present in the produced CST
    if it was in the original.

    Raises an error if `node` is not an `IndentedBlock` or `If`.

    Args:
        _wrapper: The CST node to apply or unwrap.
        _block: The block of which the wrapper is stored.
        _node: The CST node to which to apply the wrapper to.

    Returns:
        Processed and 'unwrapped' CST Node.
    """
    logger.debug(f"apply Else wrapper on {type(node).__name__}")

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
    """
    Apply an `IndentedBlock` wrapper.

    If `node` is an `IndentedBlock`, update the metadata (but not body) on
    `node` such that the information from `wrapper` is available on `node`.
    Otherwise, create a new `IndentedBlock` from `wrapper` where body equals
    `node`.

    Args:
        _wrapper: The CST node to apply or unwrap.
        _block: The block of which the wrapper is stored.
        _node: The CST node to which to apply the wrapper to.

    Returns:
        Processed and 'unwrapped' CST Node.
    """
    logger.debug("apply IndentedBlock wrapper")
    match node:
        case libcst.IndentedBlock():
            return node_manipulation.flatten_indented_blocks(wrapper, node)
        case libcst.BaseCompoundStatement() | libcst.SimpleStatementLine():
            return wrapper.with_changes(body=(node,))
        case _:
            return node
