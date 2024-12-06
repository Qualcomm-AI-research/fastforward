# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import abc
import dataclasses

from collections.abc import Iterator, Sequence
from types import UnionType
from typing import TypeVar, Union, get_args, get_origin, get_type_hints

import libcst

from typing_extensions import override

# Block = ForwardRef("Block")


@dataclasses.dataclass(eq=False)
class Block(abc.ABC):
    _wrappers: list[libcst.CSTNode] = dataclasses.field(init=False, default_factory=list)
    immediate_dominator: "Block | None" = dataclasses.field(init=False, default=None)
    immediate_post_dominator: "Block | None" = dataclasses.field(
        init=False, default=None, repr=False
    )

    @abc.abstractmethod
    def set_tail(self, tail: "Block") -> None:
        """
        Set a tail block on self and/or its children
        """

    @property
    def wrappers(self) -> list[libcst.CSTNode]:
        return self._wrappers[:]

    def push_wrapper(self, node: libcst.CSTNode) -> None:
        self._wrappers.append(node)

    def named_children(self) -> Iterator[tuple[str, "Block"]]:
        for name, annotation in get_type_hints(type(self)).items():
            # Skip all fields that are defined in the base `Block`
            if name in Block.__dataclass_fields__:
                continue
            if _is_block_annotation(annotation):
                if child := getattr(self, name):
                    yield name, child

    def blocks(
        self,
        *,
        reverse: bool = True,
    ) -> Iterator["Block"]:
        """
        Yield all blocks in sub-graph with self as root.

        The blocks are yielded in either reverse post-order or post-order
        depending on `reverse`.

        Args:
            reverse: yield blocks in reverse post order if True, other yield
                blocks in post order.
        """
        visited: set["Block"] = set()

        def dfs(block: "Block") -> Iterator["Block"]:
            """Depth first search."""
            visited.add(block)
            for _, child in block.named_children():
                if child not in visited:
                    yield from dfs(child)
            yield block

        block_iter = dfs(self)
        if reverse:
            block_iter = reversed(list(block_iter))

        yield from block_iter

    def is_dominated_by(self, other: "Block") -> bool:
        """
        True if `other` dominates `self`, False otherwise.

        Args:
            other: `Block` to test for dominance.
        """
        return _dominates(other, self)

    def dominates(self, other: "Block") -> bool:
        """
        True if `self` dominates `other`, False otherwise.

        Args:
            other: `Block` to test for dominance.
        """
        return _dominates(self, other)


BlockT = TypeVar("BlockT", bound=Block)


@dataclasses.dataclass(eq=False)
class SimpleBlock(Block):
    next: Block | None
    statements: Sequence[libcst.SimpleStatementLine]

    @override
    def set_tail(self, tail: Block) -> None:
        if tail is self:
            return

        if next_block := self.next_block:
            next_block.set_tail(tail)
        else:
            self.next_block = tail


@dataclasses.dataclass(eq=False)
class BranchingBlock(Block, abc.ABC):
    pass


@dataclasses.dataclass(eq=False)
class IfBlock(BranchingBlock):
    test: libcst.BaseExpression
    true: Block
    false: Block | None

    @override
    def set_tail(self, tail: Block) -> None:
        self.true.set_tail(tail)
        if false_branch := self.false:
            false_branch.set_tail(tail)
        else:
            self.false = tail


@dataclasses.dataclass(eq=False)
class FunctionBlock(Block):
    body: Block

    @override
    def set_tail(self, tail: Block) -> None:
        self.body.set_tail(tail)


@dataclasses.dataclass(eq=False)
class ExitBlock(Block):
    @override
    def set_tail(self, tail: Block) -> None:
        pass


def _dominates(maybe_dom: Block, block: Block) -> bool:
    """True if `maybe_dom` dominates `block`, False otherwise"""
    dom: Block | None = block
    while dom is not None:
        if maybe_dom is dom:
            return True
        dom = block.immediate_dominator
    return False


def _is_block_annotation(cls: type) -> bool:
    if origin := get_origin(cls):
        if origin is Union or origin is UnionType:
            for arg in get_args(cls):
                if _is_block_annotation(arg):
                    return True
        return False
    else:
        return issubclass(cls, Block)
