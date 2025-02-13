# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import functools

from collections.abc import Iterator, Sequence
from typing import cast

import libcst

from ..cst.validation import ensure_type
from . import _dominance, blocks
from .exceptions import CFGConstructionError


def construct(node: libcst.FunctionDef) -> blocks.FunctionBlock:
    """Construct a CFG from a `FunctionDef` CST node.

    Currently, CFG creation is only supported for functions.

    Args:
        node: Function node to turn into a CFG.

    Returns:
        CFG that represents same function as node.
    """
    cfg = _block_from_FunctionDef(node)
    _dominance.set_immediate_dominators(cfg)
    return cfg


@functools.singledispatch
def _block_from_CSTNode(node: libcst.CSTNode) -> blocks.Block:
    """Convert CST with `node` as root to a CFG.

    Specialization for this functions that accept a subclass `libcst.CSTNode`
    can be registered using `_block_from_CSTNode.register`. Each of this
    functions is expected to accept a single `libcst.CSTNode` as input and
    construct a CFG for the entire sub-tree.

    Args:
        node: `CSTNode` to convert to a CFG.

    Returns:
        A CFG block. This block may be a partial CFG, i.e., some of its edges
        may be dangling.
    """
    raise CFGConstructionError(
        f"Cannot convert {type(node).__name__} to CFG. A conversion implementation is missing. "
        + "Please file an issue if you run into this error."
    )


@_block_from_CSTNode.register
def _block_from_FunctionDef(node: libcst.FunctionDef) -> blocks.FunctionBlock:
    """Convert a `FunctionDef` to a CFG.

    The returned `FunctionBlock` contains the `FunctionDef` node and points to
    a CFG block that represents the function implementation.

    Args:
        node: A `FunctionDef` CST node that is converted into a CFG.

    Returns:
        `FunctionBlock` that represents the same function as `node`. The CFG
        returned from this function is 'complete', i.e., non of its edges are
        dangling.
    """
    body_node = ensure_type(node.body, libcst.IndentedBlock, CFGConstructionError)
    body = _block_from_IndentedBlock(body_node)

    block = blocks.FunctionBlock(body=body)
    block.set_tail(blocks.ExitBlock())
    block.push_wrapper(node)
    return block


@_block_from_CSTNode.register
def _block_from_IndentedBlock(node: libcst.IndentedBlock) -> blocks.Block:
    """Convert an `IndentedBlock` to a CFG.

    The returned `Block` may be any subclass of `Block` and depends on the
    statements in `node.body`.

    Args:
        node: An `IndentedBlock` CST node that is converted into a CFG.

    Returns:
        `Block` that represents the same block of code as `node`. The returned
        block may be partial, i.e., some of its edges may be dangling.
    """
    entry: blocks.Block | None = None
    previous: blocks.Block | None = None

    def _block_from_partial_node(node: libcst.IndentedBlock) -> blocks.Block:
        block: blocks.Block
        match node.body[0]:  # Partial blocks have at least 1 body member
            case libcst.SimpleStatementLine():
                # When the first element is a `SimpleStatementLine`, from
                # `_split_indented_block` we can be sure all elements are
                # `SimpleStatementLine` and the block does not contain branches
                statements = cast(Sequence[libcst.SimpleStatementLine], node.body)
                block = blocks.SimpleBlock(statements=statements, next_block=None)
            case _:
                # If the first element is not a `SimpleStatementLine`, the partial block
                # can, by construction, only have one element.
                assert len(node.body) == 1
                block = _block_from_CSTNode(node.body[0])

        block.push_wrapper(node)
        return block

    for partial_node in _split_indented_block(node):
        block = _block_from_partial_node(partial_node)
        entry = entry or block
        _set_tails(previous, block)
        previous = block

    if entry is None:
        raise CFGConstructionError("Encountered an empty IndentedBlock")

    return entry


@_block_from_CSTNode.register
def _block_from_If(node: libcst.If) -> blocks.IfBlock:
    """Convert `libcst.If` to a CFG.

    The returned `IfBlock` contains the test expression. The contents of the
    true and false branches are represented by other blocks that are pointed to
    by the `true` and `false` attribute.

    Args:
        node: An `If` CST node that is converted in a CFG.

    Returns:
        `IfBlock` that represents the if statement of `node`. The returned
        block may be partial, i.e., the `false` edge may be dangling.
    """
    true_node = ensure_type(node.body, libcst.IndentedBlock, CFGConstructionError)
    true_block = _block_from_IndentedBlock(true_node)
    false_block: blocks.Block | None = None
    if false_node := node.orelse:
        false_block = _block_from_CSTNode(false_node)
    block = blocks.IfBlock(test=node.test, true=true_block, false=false_block)
    block.push_wrapper(node)
    return block


@_block_from_CSTNode.register
def _block_from_Else(node: libcst.Else) -> blocks.Block:
    """Convert an `Else` node to a CFG.

    An else branch is not represented by a specific block in the CFG. It is
    identified by being the `false` member on an `IfBlock`. As such, it can be
    any type of block. In order to maintain the information from `node` it is
    pushed as a wrapper on the block. This information can be retrieved during
    reconstruction.

    Args:
        node: An `Else` CST node that is converted in a CFG.

    Returns:
        `Block` that represents the else branch of `node`. The block
        specifically represents `node.body` The returned block may be partial,
        i.e., some edges may be dangling.
    """
    block = _block_from_CSTNode(node.body)
    block.push_wrapper(node)
    return block


def _split_indented_block(node: libcst.IndentedBlock) -> Iterator[libcst.IndentedBlock]:
    """Split `IndentedBlock` into multiple `IndentedBlock`s.

    Split a single `IndentedBlock` into one or more `IndentedBlock`s that form
    the content of `node`. All returned blocks contain either no branching or a
    single compound statement that branches.

    Note that any of the yielded blocks may still contain IndentedBlocks
    as successors. These are not processed directly (i.e., recursively) in this
    function, but may be broken down in subsequent calls.

    Args:
        node: `IndentedBlock` to break up into one or more smaller `IndentedBlock`s.

    Returns:
        An iterator of `IndentedBlock`s. It is a guaranteed that each yielded
        `IndentedBlock` contains either a single statement or multiple
        statements that or not `CompoundStatement`s.
    """
    assert len(node.body) > 0, "IndentedBlock cannot be empty"
    partial_blocks: list[list[libcst.BaseStatement]] = [[]]

    for line in node.body:
        match line:
            case libcst.SimpleStatementLine():
                partial_blocks[-1].append(line)
            case _:
                if len(partial_blocks[-1]) > 0:
                    partial_blocks.append([line])
                else:
                    partial_blocks[-1].append(line)
                partial_blocks.append([])
    partial_blocks = partial_blocks[:-1] if len(partial_blocks[-1]) == 0 else partial_blocks

    if len(partial_blocks) == 1:
        yield node
        return

    # Yield first block. Keep any whitespace/comment prefixes
    yield node.with_changes(
        body=tuple(partial_blocks[0]),
        footer=(),
    )

    # Yield all but first and last blocks. Remove all prefix and trailing
    # whitespace/comment info
    for block in partial_blocks[1:-1]:
        yield node.with_changes(
            body=block,
            header=libcst.TrailingWhitespace(),
            footer=(),
        )

    # Yield last block. Keep trailing whitespace/comments
    yield node.with_changes(
        body=tuple(partial_blocks[-1]),
        header=libcst.TrailingWhitespace(),
    )


def _set_tails(block: blocks.Block | None, tail: blocks.Block) -> None:
    """Helper function that sets tails of `block` if `block` is not `None`."""
    if block:
        block.set_tail(tail)
