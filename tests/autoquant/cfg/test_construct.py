# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import functools

import libcst

from fastforward.autoquant.cfg import blocks, construct

from tests.utils.string import dedent_strip

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

_funcdef1: libcst.FunctionDef = libcst.parse_module(_src1).body[0]  # type: ignore[assignment]


def test_cfg_construction():
    cfg = construct(_funcdef1)

    tail = _TestSimpleBlock(next=_TestExitBlock())
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

    expected_graph_structure.assert_cfg_structure(cfg)


class _TestBlock:
    def __init__(self, BlockType: type[blocks.Block], **edges: "_TestBlock") -> None:
        self._expected_type = BlockType
        self._expected_edges: dict[str, _TestBlock] = edges

    def assert_cfg_structure(self, block: blocks.Block, *, path: str = "") -> None:
        name = f"root.{path}" or "root block"

        if type(block) is not self._expected_type:
            raise AssertionError(
                f"Expected '{name}' to be a {self._expected_type.__name__} but got {type(block).__name__}"
            )
        for child_name, test_block in self._expected_edges.items():
            if not hasattr(block, child_name):
                raise AssertionError(f"Expected '{name}' to have a child '{child_name}'")

            child_path = f"{path}.{child_name}" if path else child_name
            child_block: blocks.Block = getattr(block, child_name)
            test_block.assert_cfg_structure(child_block, path=child_path)


_TestSimpleBlock = functools.partial(_TestBlock, blocks.SimpleBlock)
_TestIfBlock = functools.partial(_TestBlock, blocks.IfBlock)
_TestFunctionBlock = functools.partial(_TestBlock, blocks.FunctionBlock)
_TestExitBlock = functools.partial(_TestBlock, blocks.ExitBlock)
