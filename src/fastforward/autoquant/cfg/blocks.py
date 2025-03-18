# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import abc
import dataclasses
import itertools

from collections.abc import Iterator, Sequence
from types import UnionType
from typing import Protocol, TypeVar, Union, get_args, get_origin, get_type_hints

import libcst

from typing_extensions import override


# Use eq=False to ensure that the dataclass uses object.__hash__ which defaults
# to the instance id. This way `Block`s are hashable and each block is unique.
@dataclasses.dataclass(eq=False)
class Block(abc.ABC):
    """Base class for blocks of a Control Flow Graph.

    Subclasses of `Block` implement specific types of control flow. Depending
    on the type of control flow, it may have zero or more outgoing edges.

    To retain information from the CST  from which a CFG was constructed, a
    `Block` maintains a list of `wrappers` accessible via `Block.wrappers`.
    Each `wrapper` is a CST node that may contain information used to
    reconstruct a CST from a CFG.

    The information from these wrappers should strictly not relate to control
    flow. For example, an `IfBlock` may store a `libcst.If` as a wrapper. The
    test expression on this node should not be used; instead, the test
    expression on the `IfBlock` itself should be used for analysis and
    reconstruction.

    In contrast, the `libcst.If` node may also store information on surrounding
    comments. Since these do not affect control flow, they can be used directly
    from the CST node. Note that none of the above is strictly enforced, so
    care should be taken to avoid 'leaking' information from the CST node that
    is also present on the `Block` during reconstruction.
    """

    _wrappers: list[libcst.CSTNode] = dataclasses.field(init=False, default_factory=list)
    immediate_dominator: "Block | None" = dataclasses.field(init=False, default=None)
    immediate_post_dominator: "Block | None" = dataclasses.field(
        init=False, default=None, repr=False
    )

    @abc.abstractmethod
    def set_tail(self, tail: "Block") -> None:
        """Set a tail block on self and/or its children.

        Here, tail means a CFG subgraph, i.e., a CFG that contains one or
        more blocks. This method updates all the dangling edges in this graph
        to point to `tail`. Hence, the only remaining dangling edges will be
        those in `tail`.

        Args:
            tail: The block to which to set all dangling edges in this block
                and successors.
        """

    @property
    def wrappers(self) -> list[libcst.CSTNode]:
        """A list of stored wrappers on this block."""
        return self._wrappers[:]

    def push_wrapper(self, node: libcst.CSTNode) -> None:
        """Add a wrapper node.

        A wrapper node can be used to store more CST context on a block. Please
        see the docstring on `Block` for more information.
        """
        self._wrappers.append(node)

    def named_children(self) -> Iterator[tuple[str, "Block"]]:
        """Yields all direct children of this block.

        By default, this will find all members that are a subclass of `Block`
        and yields those. A specific subclass of `Block` may require other
        behaviour. In that case, this method should be overridden. If all
        members that are `Blocks` are children, the method works as expected.

        Returns:
            An iterator over direct children and their names.
        """
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
        """Traverse graph and yield blocks in (reversed) post-order.

        In post-order traversal, the children of a node are traversed before
        the node itself is visited/yielded.

        Args:
            reverse: yield blocks in reversed order if True, otherwise
                yield blocks in post order.
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
        """True if `other` dominates `self`, False otherwise.

        Args:
            other: `Block` to test for dominance.
        """
        return _dominates(other, self)

    def dominates(self, other: "Block") -> bool:
        """True if `self` dominates `other`, False otherwise.

        Args:
            other: `Block` to test for dominance.
        """
        return _dominates(self, other)

    @abc.abstractmethod
    def visit(self, visitor: "BlockVisitor[_VT]") -> "_VT":
        """Visit block using visitor."""


@dataclasses.dataclass(eq=False)
class SimpleBlock(Block):
    """A block that encodes simple control flow.

    Each statement in `self.statements` is executed in order. After this block
    completes control flow deterministically proceeds to the block pointed to
    by `self.next_block`.
    """

    next_block: Block | None
    statements: Sequence[libcst.SimpleStatementLine]

    @override
    def set_tail(self, tail: Block) -> None:
        if tail is self:
            return

        if next_block := self.next_block:
            next_block.set_tail(tail)
        else:
            self.next_block = tail

    @override
    def visit(self, visitor: "BlockVisitor[_VT]") -> "_VT":
        return visitor.visit_SimpleBlock(self)


@dataclasses.dataclass(eq=False)
class BranchingBlock(Block, abc.ABC):
    """Abstract base class for blocks that represent branching logic.

    Subclasses of `BranchingBlock` represent control flow that may proceed to
    different blocks depending on runtime evaluation/information.
    """


@dataclasses.dataclass(eq=False)
class IfBlock(BranchingBlock):
    """Block that represents an `if` statement.

    The block contains the `test` expression and a `true` and `false` edge.
    Depending on the evaluation of `test`, control flow proceeds with the
    `true` or `false` block.
    """

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

    @override
    def visit(self, visitor: "BlockVisitor[_VT]") -> "_VT":
        return visitor.visit_IfBlock(self)


@dataclasses.dataclass(eq=False)
class FunctionBlock(Block):
    """Block that represents a function.

    From a control flow perspective, this block is similar to `SimpleBlock`.
    However, each function can only have a single `FunctionBlock` and it stores
    function level meta information as a wrapper.
    """

    body: Block

    @override
    def set_tail(self, tail: Block) -> None:
        self.body.set_tail(tail)

    @override
    def visit(self, visitor: "BlockVisitor[_VT]") -> "_VT":
        return visitor.visit_FunctionBlock(self)

    def params(self) -> Iterator[str]:
        """Yields names of all parameters of function."""
        if not self.wrappers:
            return

        funcdef = self.wrappers[0]
        assert isinstance(funcdef, libcst.FunctionDef)

        parameters = funcdef.params
        for param in itertools.chain(
            parameters.params, parameters.kwonly_params, parameters.posonly_params
        ):
            yield param.name.value

        if isinstance(parameters.star_arg, libcst.Param):
            yield parameters.star_arg.name.value

        if (star_kwarg := parameters.star_kwarg) is not None:
            yield star_kwarg.name.value

    @override
    def push_wrapper(self, node: libcst.CSTNode) -> None:
        if len(self.wrappers) == 0 and not isinstance(node, libcst.FunctionDef):
            raise ValueError(f"The root wrapper type of {type(self).__name__} must be FunctionDef.")
        super().push_wrapper(node)


@dataclasses.dataclass(eq=False)
class ExitBlock(Block):
    """Block that represents the exit of a function.

    There can only be a single exit block in a CFG and all paths will end up in
    the exit block. The exit block itself does not represent any further
    control flow.
    """

    @override
    def set_tail(self, tail: Block) -> None:
        pass

    @override
    def visit(self, visitor: "BlockVisitor[_VT]") -> "_VT":
        return visitor.visit_ExitBlock(self)


def _dominates(maybe_dom: Block, block: Block) -> bool:
    """True if `maybe_dom` dominates `block`, False otherwise."""
    dom: Block | None = block
    while dom is not None:
        if maybe_dom is dom:
            return True
        dom = dom.immediate_dominator
    return False


def _is_block_annotation(cls: type) -> bool:
    """Test if `cls` is `Block` type.

    Also returns True if `cls` is a Union type that includes a `Block`-type.
    """
    if origin := get_origin(cls):
        if origin is Union or origin is UnionType:
            for arg in get_args(cls):
                if _is_block_annotation(arg):
                    return True
        return False
    else:
        return issubclass(cls, Block)


_VT = TypeVar("_VT", covariant=True)


class BlockVisitor(Protocol[_VT]):
    """Protocol for a BlockVisitor.

    An object that satisfies the `BlockVisitor` can be passed to `Block.visit`
    which will dispatch to the appropriate visit method.

    Note that this visitor does not traverse the graph by default. Any form of
    traversal must be implemented in the respective visit methods.
    """

    def visit_FunctionBlock(self, __block: FunctionBlock) -> _VT:
        """Visitor for `FunctionBlock`.

        Args:
            __block: The block that is visisted.
        """
        raise NotImplementedError

    def visit_ExitBlock(self, __block: ExitBlock) -> _VT:
        """Visitor for `ExitBlock`.

        Args:
            __block: The block that is visisted.
        """
        raise NotImplementedError

    def visit_IfBlock(self, __block: IfBlock) -> _VT:
        """Visitor for `IfBlock`.

        Args:
            __block: The block that is visisted.
        """
        raise NotImplementedError

    def visit_SimpleBlock(self, __block: SimpleBlock) -> _VT:
        """Visitor for `SimpleBlock`.

        Args:
            __block: The block that is visisted.
        """
        raise NotImplementedError
