# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import itertools
import textwrap

import libcst
import pytest

from fastforward.autoquant.cfg import _construct, blocks

from tests.utils.string import dedent_strip

(_src1,) = dedent_strip("""
    def example(a):
        b = 200
        c = 300
        if a > 10:
            out = a
        else:
            out = 10
        return out
        # a trailing comment
""")

_funcdef1 = libcst.parse_module(_src1).body[0]


@pytest.mark.parametrize("funcdef", (_funcdef1,))
def test_break_indented_block(funcdef: libcst.FunctionDef) -> None:
    assert isinstance(funcdef.body, libcst.IndentedBlock)
    blocks = list(_construct._break_indented_block(funcdef.body))
    assert len(blocks) == 3
    assert len(blocks[0].body) == 2
    assert len(blocks[1].body) == 1
    assert len(blocks[2].body) == 1

    assert blocks[0].header.deep_equals(funcdef.body.header)
    for line_block, line_body in zip(blocks[2].footer, funcdef.body.footer):
        assert line_block.deep_equals(line_body)


class TestNode:
    _expected: type[blocks.BaseBlock]
    _children: dict[str, "TestNode"]

    def __init__(self, expected_block_type: type[blocks.BaseBlock]) -> None:
        object.__setattr__(self, "_expected", expected_block_type)
        object.__setattr__(self, "_children", {})

    def __getattr__(self, name: str) -> "TestNode":
        return self._children[name]

    def __setattr__(self, name: str, node: "TestNode") -> None:
        self._children[name] = node

    def validate(self, block: blocks.BaseBlock, prefix="") -> None:
        assert isinstance(
            block, self._expected
        ), f"Expected {self._expected.__name__} got {type(block).__name__} for Entry{prefix}"
        for name, testnode in self._children.items():
            child: blocks.BlockEdge | None = getattr(block, name)
            assert child is not None
            assert (child_block := child.block()) is not None
            testnode.validate(child_block, prefix=f"{prefix}.{name}")


@pytest.mark.parametrize("funcdef", (_funcdef1,))
def test_construct(funcdef: libcst.FunctionDef) -> None:
    cfg = _construct.construct(funcdef)

    entry = TestNode(blocks.EntryBlock)
    entry.next = TestNode(blocks.BaseBlock)
    entry.next.next = TestNode(blocks.BaseBlock)
    entry.next.next.next = (if_block := TestNode(blocks.IfBlock))
    if_block.true = TestNode(blocks.BaseBlock)
    if_block.false = TestNode(blocks.BaseBlock)
    if_block.true.next = TestNode(blocks.BaseBlock)
    if_block.false.next = TestNode(blocks.BaseBlock)
    if_block.true.next.next = TestNode(blocks.ExitBlock)
    if_block.false.next.next = TestNode(blocks.ExitBlock)

    entry.validate(cfg.entry)


def test_construct_reconstruct_if():
    parts = [
        "\n    elif b:\n        second = 1",
        "\n    else:\n        third = 1",
        "\n    fourth = 1",
    ]

    # This will create all options for control flow of an if statement:
    # ifelse branch, else branch, block after if statement. Each of these
    # may be included or not included. The resulting 8 samples are tested
    # for reconstruction.
    def create_test_src():
        src = textwrap.dedent("def func():\n    if a:\n        first = 1")
        for include_elif, include_else, include_next in itertools.product(*[(True, False)] * 3):
            out = src
            if include_elif:
                out += parts[0]
            if include_else:
                out += parts[1]
            if include_next:
                out += parts[2]

            yield out

    for out in create_test_src():
        cst = libcst.parse_module(out)
        cfg = _construct.construct(cst.body[0])  # type: ignore[arg-type]
        cst = cst.with_changes(body=[cfg.entry.reconstruct()])
        assert out == cst.code
