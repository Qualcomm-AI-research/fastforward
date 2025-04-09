# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import TypeVar

import libcst


def combine_indented_blocks(
    block1: libcst.IndentedBlock,
    block2: libcst.IndentedBlock,
) -> libcst.IndentedBlock:
    """Combine two indented blocks into a single indented block.

    `IndentedBlock`s are combined such that the statements of block1 are
    followed by the statements in block2 and any leading or trailing comments
    or whitespace is maintained.

    Args:
        block1: The first block to merge.
        block2: The second block to merge.

    Returns:
        A single `IndentedBlock` that maintains the information in both
        `block1` and `block2`.
    """
    midway_lines = (
        *block1.footer,
        libcst.EmptyLine(
            whitespace=block2.header.whitespace,
            comment=block2.header.comment,
            newline=block2.header.newline,
        ),
    )

    body = list(block2.body)
    if len(body) == 0:
        return block1

    root_elem: libcst.SimpleStatementLine | libcst.BaseCompoundStatement = body[0]  # type: ignore[assignment]
    root_elem = root_elem.with_changes(
        leading_lines=(*midway_lines, *root_elem.leading_lines),
    )
    return block1.with_changes(
        body=tuple(block1.body) + tuple(body),
        footer=block2.footer,
    )


def flatten_indented_blocks(
    block1: libcst.IndentedBlock,
    block2: libcst.IndentedBlock,
) -> libcst.IndentedBlock:
    """Discard the body of `block1` and combine its header and footer with `block2`.

    Combines two `IndentedBlock`s in a way that the statements/body of `block2`
    are maintained and the comments, trailing whitespace and leading whitespace
    from both blocks are maintained.

    Args:
        block1: The first block to merge.
        block2: The second block to merge.

    Returns:
        A single `IndentedBlock` that maintains the information in both
        `block1` and `block2` but only the statements/body of `block2`.
    """
    return block2.with_changes(
        header=_merge_trailing_whitespace(block1.header, block2.header),
        footer=tuple(block2.footer) + tuple(block1.footer),
    )


def _merge_trailing_whitespace(
    whitespace1: libcst.TrailingWhitespace,
    whitespace2: libcst.TrailingWhitespace,
) -> libcst.TrailingWhitespace:
    comments: list[str] = []
    if comment := whitespace1.comment:
        comments.append(comment.value)
    if comment := whitespace2.comment:
        comments.append(comment.value)

    if len(comments) == 0:
        return whitespace1

    return whitespace1.with_changes(comment=libcst.Comment(value=" ".join(comments)))


_CompoundT = TypeVar("_CompoundT", bound=libcst.BaseCompoundStatement)


def unwrap_single_statement(
    node: libcst.IndentedBlock,
    StatementType: type[_CompoundT] = libcst.BaseCompoundStatement,  # type: ignore[assignment]
) -> _CompoundT | libcst.IndentedBlock:
    """Unwrap an `IndentedBlock` if it only contains a single `StatementType`.

    If `node.body` only contains a single statement of type `StatementType`, return only the first member of `node.body`.
    Otherwise, return `node` unchanged.

    Args:
        node: the IndentedBlock to unwrap if it satisfies the statement test.
        StatementType: The statement type to test for.

    Returns:
        `node` if `node.body` has two or more elements or the first element is
        not of type `StatementType`. Otherwise `node.body[0]`.
    """
    if len(node.body) == 1 and isinstance(node.body[0], StatementType):
        return node.body[0]
    return node
