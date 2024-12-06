# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import itertools
import sys

from collections.abc import Sequence
from typing import TextIO

import libcst

from libcst._nodes.internal import CodegenState

from . import blocks


class _GraphvizPrinter:
    def __init__(self) -> None:
        self._node_identifiers: set[int] = set()
        self._edge_identifiers: set[tuple[int, int]] = set()
        self._nodes: list[str] = []
        self._edges: list[str] = []

    def add_node(self, identifier: int, content: str) -> None:
        self._node_identifiers.add(identifier)
        self._nodes.append(
            f'{identifier}[shape=box nojustify=true label="{content}\\l" fontsize=10]'
        )

    def add_edge(self, src: int, dst: int, label: str | None = None) -> None:
        edge_identifier = (src, dst)

        if edge_identifier in self._edge_identifiers:
            return
        self._edge_identifiers.add(edge_identifier)

        attributes = ""
        if label:
            attributes = f'[label="{label}" fontsize=8]'
        self._edges.append(f"{src} -> {dst}{attributes}")

    def add_dominator_edges(self, block: blocks.Block) -> None:
        if block.immediate_dominator:
            self._edges.append(f'{id(block)} -> {id(block.immediate_dominator)} [color="red"]')
        if block.immediate_post_dominator:
            self._edges.append(
                f'{id(block)} -> {id(block.immediate_post_dominator)} [color="darkorchid3"]'
            )

    def add_block(self, block: blocks.Block) -> None:
        if id(block) in self._node_identifiers:
            return

        if add_fn := getattr(self, f"add_{type(block).__name__}"):
            add_fn(block)
            self.add_dominator_edges(block)
        else:
            raise NotImplementedError(
                f"{type(block).__name__} is not supported by {type(self).__name__}"
            )

    def add_FunctionBlock(self, block: blocks.FunctionBlock) -> None:
        self.add_node(id(block), "ENTRY")
        self.add_edge(id(block), id(block.body))

    def add_SimpleBlock(self, block: blocks.SimpleBlock) -> None:
        self.add_node(id(block), _codegen(block.statements))
        if next := block.next:
            self.add_edge(id(block), id(next))

    def add_IfBlock(self, block: blocks.IfBlock) -> None:
        test_src = _codegen([block.test])
        self.add_node(id(block), f"if {test_src}")

        self.add_edge(id(block), id(block.true), "true")

        if false := block.false:
            self.add_edge(id(block), id(false), "false")

    def add_ExitBlock(self, block: blocks.IfBlock) -> None:
        self.add_node(id(block), "EXIT")

    def write(self, writer: TextIO) -> None:
        _ = writer.write("digraph G{\n")
        for line in itertools.chain(self._nodes, self._edges):
            _ = writer.write(line)
            _ = writer.write("\n")
        _ = writer.write("}\n")


def dump_graphviz(root: blocks.Block, writer: TextIO = sys.stdout) -> None:
    printer = _GraphvizPrinter()
    for block in root.blocks():
        printer.add_block(block)

    printer.write(writer)


def _codegen(nodes: Sequence[libcst.CSTNode | str]) -> str:
    codegen_state = CodegenState("  ", "\n")
    for node in nodes:
        match node:
            case str():
                codegen_state.add_token(node)
            case libcst.CSTNode():
                node._codegen(codegen_state)

    codegen_txt = "".join(codegen_state.tokens)
    return codegen_txt.strip().replace("\n", "\\l").replace('"', '\\"')
