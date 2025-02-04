# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import subprocess

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

import libcst as libcst

from libcst.display.graphviz import dump_graphviz
from typing_extensions import override


@contextmanager
# pylint: disable-next=invalid-name # for similarity with TemporaryDirectory
def TemporaryPath() -> Iterator[Path]:
    """Provides a temporary directory as a Path object."""
    with TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


def render_tree(
    tree: libcst.CSTNode,
    filename: str = "outfile.png",
    show_defaults: bool = False,
    show_syntax: bool = False,
    show_whitespace: bool = False,
) -> None:
    """Renders tree as graphviz image.

    Assumes the system package `graphviz` is installed.
    """
    text = dump_graphviz(
        tree,
        show_defaults=show_defaults,
        show_whitespace=show_whitespace,
        show_syntax=show_syntax,
    )
    with TemporaryPath() as tmp_dir:
        tmp_file = tmp_dir / "outfile.dot"
        tmp_file.write_text(text)
        subprocess.run(["dot", "-Tpng", str(tmp_file), "-o", filename], check=True)


class TreeFlattener(libcst.CSTVisitor):
    def __init__(self) -> None:
        self.fqns: list[str] = []
        self.stack: list[str] = []

    @override
    def on_visit(self, original: libcst.CSTNode) -> bool:
        name = original.name.value if hasattr(original, "name") else type(original).__name__
        self.stack.append(name)
        self.fqns.append(".".join(self.stack))
        return True

    @override
    def on_leave(self, original: libcst.CSTNode) -> None:
        del original
        self.stack.pop()


def compute_graph_diff(tree1: libcst.CSTNode, tree2: libcst.CSTNode) -> set[str]:
    """Finds "names" of nodes that are on one but not on the other tree.

    Known limitations:
    - Does not provide FQN
    - Ignores node values
    """
    flattener_actual = TreeFlattener()
    flattener_expected = TreeFlattener()
    tree1.visit(flattener_expected)
    tree2.visit(flattener_actual)

    return set(flattener_expected.fqns).symmetric_difference(flattener_actual.fqns)
