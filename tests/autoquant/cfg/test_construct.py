# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import functools
import textwrap

import libcst
import pytest

from fastforward._autoquant.cfg import blocks, construct
from fastforward._autoquant.cfg.exceptions import CFGConstructionNotImplemented
from fastforward.testing.string import dedent_strip

(_src1,) = dedent_strip("""
    def example(a):
        b = 200
        c = 300
        if a > 10:
            out = a
        elif a < 5:
            out = a / 2
        else:
            out = 10
        return out
        # a trailing comment
""")

(_src2,) = dedent_strip("""
    def example(a):
        b = 200
        c = 300
        out = 0
        while a > 10:
            a -= 1
            b = b / a
            c = c * b
            out += b * c
        for i in range(a):
            out += b
        return out
""")

_funcdef1: libcst.FunctionDef = libcst.parse_module(_src1).body[0]  # type: ignore[assignment]
_funcdef2: libcst.FunctionDef = libcst.parse_module(_src2).body[0]  # type: ignore[assignment]


def test_cfg_construction_1() -> None:
    # Given an expected CFG structure
    tail = _TestSimpleBlock(next_block=_TestExitBlock())
    expected_graph_structure = _TestFunctionBlock(
        body=_TestSimpleBlock(
            next_block=_TestIfBlock(
                true=_TestSimpleBlock(next_block=tail),
                false=_TestIfBlock(
                    true=_TestSimpleBlock(next_block=tail),
                    false=_TestSimpleBlock(next_block=tail),
                ),
            ),
        ),
    )

    # When a CFG is created that matches the expected CFG structure
    cfg = construct(_funcdef1)

    # Then the CFG structure must match the expected CFG structure.
    expected_graph_structure.assert_cfg_structure(cfg)


def test_cfg_construction_2() -> None:
    # Given an expected CFG structure
    exit_block = _TestExitBlock()
    for_block = _TestForBlock(
        body=_TestSimpleBlock(), next_block=_TestSimpleBlock(next_block=exit_block)
    )
    for_block.set_edge("body.next_block", for_block)

    while_block = _TestWhileBlock(body=_TestSimpleBlock(), next_block=for_block)
    while_block.set_edge("body.next_block", while_block)

    expected_graph_structure = _TestFunctionBlock(body=_TestSimpleBlock(next_block=while_block))

    # When a CFG is created that matches the expected CFG structure
    cfg = construct(_funcdef2)

    # Then the CFG structure must match the expected CFG structure.
    expected_graph_structure.assert_cfg_structure(cfg)


class _TestBlock:
    """A test utility to assert the structure of a Control Flow Graph.

    `_TestBlock`s can be used to create a graph structure.
    `assert_cfg_structure` will then assert that the given CFG has the same
    structure in terms of edges and node/block types.

    Args:
        BlockType: The expected block type for a CFG node.
        edges: Named edges that must be a member of the tested CFG.
    """

    def __init__(self, BlockType: type[blocks.Block], **edges: "_TestBlock") -> None:
        self._expected_type = BlockType
        self._expected_edges: dict[str, _TestBlock] = edges

    def assert_cfg_structure(
        self, block: blocks.Block, *, path: str = "", seen: set["_TestBlock"] | None = None
    ) -> None:
        """Assert that the given CFG `block` has the same structure as the `_TestBlock` graph.

        The CFG given by `block` must match the `BlockType`s of each
        `_TestBlock` and all edges must be present.

        Args:
            block: The CFG to test.
            path: A string to identify a given block. Used for error reporting.
            seen: Set of seen blocks that break recursive loop.
        """
        name = f"root.{path}" if path else "root block"
        seen = seen or set()

        if type(block) is not self._expected_type:
            raise AssertionError(
                f"Expected '{name}' to be a {self._expected_type.__name__} but got {type(block).__name__}"
            )
        for child_name, test_block in self._expected_edges.items():
            if test_block in seen:
                continue
            seen.add(test_block)

            if not hasattr(block, child_name):
                raise AssertionError(f"Expected '{name}' to have a child '{child_name}'")

            child_path = f"{path}.{child_name}" if path else child_name
            child_block: blocks.Block = getattr(block, child_name)
            test_block.assert_cfg_structure(child_block, path=child_path, seen=seen)

    def set_edge(self, name: str, target: "_TestBlock") -> None:
        *edges, tail = name.split(".")
        block = self
        for edge in edges:
            block = block._expected_edges[edge]
        block._expected_edges[tail] = target

    def __getattr__(self, key: str) -> "_TestBlock":
        return self._expected_edges[key]


# Some helpers to create test graphs more succinctly.
_TestSimpleBlock = functools.partial(_TestBlock, blocks.SimpleBlock)
_TestIfBlock = functools.partial(_TestBlock, blocks.IfBlock)
_TestFunctionBlock = functools.partial(_TestBlock, blocks.FunctionBlock)
_TestExitBlock = functools.partial(_TestBlock, blocks.ExitBlock)
_TestWhileBlock = functools.partial(_TestBlock, blocks.WhileBlock)
_TestForBlock = functools.partial(_TestBlock, blocks.ForBlock)


@pytest.mark.parametrize(
    "code",
    [
        """
        for _ in range(10):
            pass
        else:
            pass
        """,
        """
        while True:
            pass
        else:
            pass
        """,
        """
        async for _ in range(10):
            pass
        """,
    ],
)
def test_construction_notimplemented_failure(code: str) -> None:
    src = f"def func():{textwrap.indent(textwrap.dedent(code), '    ')}"
    funcdef = libcst.parse_statement(src)
    assert isinstance(funcdef, libcst.FunctionDef)
    with pytest.raises(CFGConstructionNotImplemented):
        construct(funcdef)
