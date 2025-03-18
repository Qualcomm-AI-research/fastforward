# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import inspect
import unittest.mock

from fastforward.autoquant.cfg import blocks


def test_block_visitor() -> None:
    """Test the contract between `BlockVisitor`  and `Block`.

    This test is included to flag any block that does not satisfy the
    contract. It assumes that a (non-initialized) block instance can be
    created using `Block.__new__`. As long as that holds, this test is valid.
    """
    for BlockType in blocks.Block.__subclasses__():
        if inspect.isabstract(BlockType):
            continue
        visitor_mock = unittest.mock.Mock()

        # GIVEN an instance of a non-abstract Block.
        block = BlockType.__new__(BlockType)  # type: ignore[type-abstract]

        # WHEN the block is visisted.
        block.visit(visitor_mock)

        # THEN the appropriate visit method on visitor must be called.
        visit_method = getattr(visitor_mock, f"visit_{BlockType.__name__}")
        visit_method.assert_called()
