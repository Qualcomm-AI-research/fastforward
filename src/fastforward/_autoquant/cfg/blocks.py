# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import abc
import dataclasses
import itertools

from collections.abc import Iterator, Sequence
from types import UnionType
from typing import Callable, Protocol, TypeVar, Union, get_args, get_origin, get_type_hints

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

    _wrappers: list[libcst.CSTNode] = dataclasses.field(
        init=False, default_factory=list, repr=False
    )
    immediate_dominator: "Block | None" = dataclasses.field(init=False, default=None, repr=False)
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

    def children(self) -> Iterator["Block"]:
        """Yields all direct children of this block.

        By default, this will find all members that are a subclass of `Block`
        and yields those. A specific subclass of `Block` may require other
        behaviour. In that case, this method should be overridden. If all
        members that are `Blocks` are children, the method works as expected.

        Returns:
            An iterator over direct children and their names.
        """
        for _, child in self.named_children():
            yield child

    def blocks(
        self,
        *,
        reverse: bool = False,
    ) -> Iterator["Block"]:
        """Traverse graph and yield blocks in (reversed) post-order.

        In post-order traversal, the children of a node are traversed before
        the node itself is visited/yielded.

        Args:
            reverse: yield blocks in reversed order if True, otherwise
                yield blocks in post order.
        """
        yield from _traverse_cfg(self, reversed=reverse)

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


def _traverse_cfg(cfg: Block, reversed: bool = False) -> Iterator[Block]:
    """Traverse a CFG using a priority-based topological sorting heuristic.

    This function produces an ordering of nodes in a CFG, with the following
    properties:

    If `reversed` is False:
    1. For all nodes u, v, if there is a simple path root, ..., u, ..., v, then
       u <= v.
    2. For all nodes u, v, if there is a simple path u, ..., v, ..., sink, then
        u <= v.
    3. in the presence of cycles in the CFG, the algorithm
       prioritizes preserving the root-to-node ordering over the sink-to-node
       ordering.

    If `reversed` is True:
    1. For all nodes u, v, if there is a simple path sink, ..., u, ..., v, then
       u <= v.
    2. For all nodes u, v, if there is a simple path u, ..., v, ..., root, then
        u <= v.
    3. in the presence of cycles in the CFG, the algorithm
       prioritizes preserving the sink-to-node ordering over the sink-to-node
       ordering.

    The method assigns a priority score to each node based on its distance from
    the root and to the sink, then sorts nodes accordingly. Cycles are handled
    by breaking edges that violate the root-first (or sink-first) constraint.
    """
    dist_from_root: dict["Block", int] = {}
    dist_from_sink: dict["Block", int] = {}
    parents: dict[Block, list[Block]] = {}

    def _infer_distance(
        block: "Block",
        *,
        child_fn: Callable[[Block], Iterator[Block]],
        dists: dict["Block", int],
        parents_collector: dict[Block, list[Block]],
        dist: int = 0,
    ) -> None:
        if dists.get(block, float("inf")) <= dist:
            return
        dists[block] = dist
        for child in child_fn(block):
            parents_collector.setdefault(child, []).append(block)
            _infer_distance(
                child,
                dists=dists,
                child_fn=child_fn,
                dist=dist + 1,
                parents_collector=parents_collector,
            )

    def parents_fn(block: Block) -> Iterator[Block]:
        yield from parents.get(block) or []

    _infer_distance(cfg, dists=dist_from_root, child_fn=Block.children, parents_collector=parents)
    sink = next(block for block in dist_from_root if isinstance(block, ExitBlock))
    _infer_distance(sink, dists=dist_from_sink, child_fn=parents_fn, parents_collector={})

    node_priorities = [
        (
            (dist_from_root[block], -dist_from_sink[block], block)
            if not reversed
            else (dist_from_sink[block], -dist_from_root[block], block)
        )
        for block in dist_from_root
    ]

    for _, _, block in sorted(node_priorities, key=lambda prio: prio[:2]):
        yield block


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
class ForBlock(BranchingBlock):
    body: Block
    iter: libcst.BaseExpression
    target: libcst.BaseAssignTargetExpression
    next_block: Block | None = None

    @override
    def set_tail(self, tail: Block) -> None:
        if self.next_block is not None:
            self.next_block.set_tail(tail)
        else:
            self.next_block = tail

    @override
    def visit(self, visitor: "BlockVisitor[_VT]") -> "_VT":
        return visitor.visit_ForBlock(self)


@dataclasses.dataclass(eq=False)
class WhileBlock(BranchingBlock):
    body: Block
    test: libcst.BaseExpression
    next_block: Block | None = None

    @override
    def set_tail(self, tail: Block) -> None:
        if self.next_block is not None:
            self.next_block.set_tail(tail)
        else:
            self.next_block = tail

    @override
    def visit(self, visitor: "BlockVisitor[_VT]") -> "_VT":
        return visitor.visit_WhileBlock(self)


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

    def visit_ForBlock(self, __block: ForBlock) -> _VT:
        """Visitor for `ForBlock`.

        Args:
            __block: The block that is visisted.
        """
        raise NotImplementedError

    def visit_WhileBlock(self, __block: WhileBlock) -> _VT:
        """Visitor for `WhileBlock`.

        Args:
            __block: The block that is visisted.
        """
        raise NotImplementedError


def insert_block_between(
    source: Block,
    target: Block,
    new_block: Block | None = None,
    child_attr: str | None = None,
) -> None:
    """Insert `new_block` between `source` and `target` blocks.

    `target` must be a child of `source`. `new_block` is inserted in between
    `source` and `target`, making `new_block` a child of `source` and `target`
    a child of `new_block`. Dominator information on all blocks is updated
    appropriately.

    If `new_block` is None, a new `SimpleBlock` is created. Otherwise the
    provided `new_block` is used. When a `new_block` is provided, `child_attr`
    must be provided which indicates the attribute to which `target` must be
    assigned to on `new_block`.
    """
    if new_block is not None and child_attr is None:
        raise ValueError("'child_attr' cannot be 'None' if 'new_block' is provided.")

    if new_block is None:
        new_block = SimpleBlock(next_block=target, statements=[])
        new_block.push_wrapper(libcst.IndentedBlock(body=()))
    else:
        if child_attr is None:
            raise ValueError("'child_attr' cannot be 'None' if 'new_block' is provided.")
        setattr(new_block, child_attr, target)

    for name, child in source.named_children():
        if child is target:
            setattr(source, name, new_block)
            break
    else:
        raise ValueError("'target' is not a child of 'source'")

    if target.immediate_dominator is source:
        target.immediate_dominator = new_block

    if source.immediate_post_dominator is target:
        source.immediate_post_dominator = new_block

    new_block.immediate_dominator = source
    new_block.immediate_post_dominator = target
