# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import functools

from collections.abc import Iterator, Sequence
from typing import cast

import libcst

from . import _dominance, blocks
from .exceptions import CFGConstructionError
from .validation import ensure_type


def construct(node: libcst.FunctionDef) -> blocks.FunctionBlock:
    cfg = _block_from_FunctionDef(node)
    _dominance.infer_immediate_dominators(cfg)
    return cfg


@functools.singledispatch
def _block_from_CSTNode(node: libcst.CSTNode) -> blocks.Block:
    raise CFGConstructionError(
        f"Cannot convert {type(node).__name__} to CFG. A conversion implementation is missing. "
        + "Please file an issue if you run into this error."
    )


@_block_from_CSTNode.register
def _block_from_FunctionDef(node: libcst.FunctionDef) -> blocks.FunctionBlock:
    body_node = ensure_type(node.body, libcst.IndentedBlock, CFGConstructionError)
    body = _block_from_IndentedBlock(body_node)

    block = blocks.FunctionBlock(body=body)
    block.set_tail(blocks.ExitBlock())
    block.push_wrapper(node)
    return block


@_block_from_CSTNode.register
def _block_from_IndentedBlock(node: libcst.IndentedBlock) -> blocks.Block:
    entry: blocks.Block | None = None
    previous: blocks.Block | None = None

    def _block_from_partial_node(node: libcst.IndentedBlock) -> blocks.Block:
        block: blocks.Block
        match node.body[0]:  # Partial blocks have at least 1 body member
            case libcst.SimpleStatementLine():
                # When the first element is a `SimpleStatementLine`, from
                # `_split_indented_block` we can be sure all elements are
                # `SimpleStatementLine` and the block does not contain branches
                statements = cast(list[libcst.SimpleStatementLine], node.body)
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
    block = _block_from_CSTNode(node.body)
    block.push_wrapper(node)
    return block


def _split_indented_block(node: libcst.IndentedBlock) -> Iterator[libcst.IndentedBlock]:
    """
    Split a single `IndentedBlock` into one or more `IndentedBlock`s that form
    the content of CFG blocks. All returned blocks contain either no branching
    or a single compound statement that branches.
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
    if block:
        block.set_tail(tail)
