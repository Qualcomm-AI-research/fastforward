# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import copy
import dataclasses
import enum

from collections.abc import Hashable, Iterable, Iterator, Set
from types import EllipsisType
from typing import Any, Generic, TypeVar, overload

import libcst

from typing_extensions import Self, override

from ..cst import nodes
from ..cst.node_processing import unpack_sequence_expression
from . import _dominance, block_node_elems, blocks


def infer_block_dataflow(cfg: blocks.Block) -> dict[blocks.Block, "BlockTracker"]:
    """Infer dataflow sets for each block in `cfg`.

    Dataflow analysis helps answer questions such as:
        - Which variable was defined where?
        - Can there be multiple version for a given variable at a specific point in a function?
        - Is the variable already quantized? (when used in conjunction with some extra bookkeeping).

    Args:
        cfg: The CFG to infer dataflow sets for.

    Returns:
        A dictionary mapping from blocks to a `BlockTracker` which contains the
        dataflow sets for each block in `cfg`.
    """
    variable_collection = VariableCollection()
    tracking_visitor = _VariableTrackerVisitor(variable_collection)
    cfg.visit(tracking_visitor)

    # Infer the `vars_gen` set for each block and initialize the `vars_out` set
    # to equal `vars_gen`.
    trackers = tracking_visitor.trackers
    for tracker in trackers.values():
        tracker.vars_out = copy.copy(tracker.vars_gen)

    # Propagate variables through CFG.
    _propagate(cfg, trackers)
    return trackers


def _propagate(cfg: blocks.Block, trackers: dict[blocks.Block, "BlockTracker"]) -> None:
    """Propagate dataflow through `cfg`.

    Iteratively update `vars_in` and `vars_out` for each block based on the
    parents. `vars_in` is updated to be the union of all parents and `vars_out`
    is updated to equal `vars_in - vars_gen[@] - vars_kill[@] + vars_gen` where
    `[@]` indicates that version numbers are ignored for set subtraction.

    Args:
        cfg: The CFG to propagate the dataflow for.
        trackers: A dict mapping blocks to a `BlockTracker` for each block in
            `cfg`.
    """
    open_set = _OrderedSet(cfg.blocks())
    parents = _dominance.infer_parents(cfg)

    # If the `vars_out` set of a block is updated, all direct children are
    # added to the `open_set`. The algorithm will terminate once the `open_set`
    # is empty. Since there is a fixed set of variables and `vars_out` can only
    # grow, this is guaranteed to terminate.
    while block := open_set.pop(default=None):
        tracker = trackers[block]
        parents_block = parents[block]
        parents_tracker = {blk: trackers[blk] for blk in parents_block}
        parents_out = [p.vars_out for p in parents_tracker.values()]

        tracker.vars_in = VariableSet().union_(*parents_out)

        vars_out = (
            tracker.vars_in.subtract(tracker.vars_kill, all_versions=True)
            .subtract_(tracker.vars_gen, all_versions=True)
            .union_(tracker.vars_gen)
        )

        if vars_out != tracker.vars_out:
            tracker.vars_out = vars_out
            for _, child in block.named_children():
                open_set.add(child)


class _VariableTrackerVisitor:
    def __init__(self, variable_collection: "VariableCollection") -> None:
        self.variable_collection = variable_collection
        self.trackers: dict[blocks.Block, BlockTracker] = {}

    def _visit_children(self, block: blocks.Block) -> None:
        for _, child in block.named_children():
            if child not in self.trackers:
                child.visit(self)

    def _process_assignments(self, block: blocks.Block, tracker: "BlockTracker") -> None:
        for assignment, statement in block_node_elems.assignments_in_block(block):
            quantization_status = get_quantization_status(assignment.value)
            for target in assignment.targets:
                if isinstance(target, libcst.Name):
                    name = target.value
                    var = self.variable_collection.get(
                        name=name,
                        quantization_status=quantization_status,
                        declaration_block=block,
                        declaration_node=statement,
                    )
                    for cur_var in tracker.vars_gen.remove(name):
                        tracker.vars_local.add(cur_var)
                    tracker.vars_gen.add(var)

        self._visit_children(block)

    def _create_tracker(self, block: blocks.Block) -> "BlockTracker":
        assert block not in self.trackers
        tracker = self.trackers[block] = BlockTracker()
        return tracker

    def visit_SimpleBlock(self, block: blocks.SimpleBlock) -> None:
        tracker = self._create_tracker(block)
        self._process_assignments(block, tracker)

    def visit_IfBlock(self, block: blocks.IfBlock) -> None:
        tracker = self._create_tracker(block)
        self._process_assignments(block, tracker)

    def visit_WhileBlock(self, block: blocks.WhileBlock) -> None:
        tracker = self._create_tracker(block)
        self._process_assignments(block, tracker)

    def visit_ForBlock(self, block: blocks.ForBlock) -> None:
        tracker = self._create_tracker(block)

        for target in unpack_sequence_expression(block.target):
            if not isinstance(target, libcst.Name):
                continue
            var = self.variable_collection.get(
                name=target.value,
                quantization_status=QuantizationStatus.Unknown,
                declaration_block=block,
                declaration_node=block.target,
            )
            tracker.vars_gen.add(var)

        self._visit_children(block)

    def visit_FunctionBlock(self, block: blocks.FunctionBlock) -> None:
        tracker = self._create_tracker(block)
        for param in block.params():
            # We always assume that function inputs are not quantized. This
            # ensures that an input quantizer is created if required.
            var = self.variable_collection.get(
                name=param,
                declaration_block=block,
                quantization_status=QuantizationStatus.NotQuantized,
            )
            tracker.vars_gen.add(var)
        self._visit_children(block)

    def visit_ExitBlock(self, block: blocks.ExitBlock) -> None:
        assert block not in self.trackers
        self.trackers[block] = BlockTracker()
        self._visit_children(block)

    def visit_WithBlock(self, block: blocks.WithBlock) -> None:
        tracker = self._create_tracker(block)
        for item in block.items:
            if (asname := item.asname) is None:
                continue
            for elem in unpack_sequence_expression(asname.name):
                if isinstance(elem, libcst.Name):
                    var = self.variable_collection.get(
                        name=elem.value,
                        declaration_block=block,
                        quantization_status=QuantizationStatus.Unknown,
                    )
                    tracker.vars_gen.add(var)
        self._visit_children(block)

    def visit_MarkerBlock(self, block: blocks.MarkerBlock) -> None:
        _ = self._create_tracker(block)
        self._visit_children(block)


class QuantizationStatus(enum.Enum):
    """Enum encoding quantization status of a variable."""

    NotQuantized = enum.auto()
    Quantized = enum.auto()
    Unknown = enum.auto()


@dataclasses.dataclass(frozen=True)
class Variable:
    """Represents a variable and its version."""

    name: str
    version: int | None
    quantization_status: QuantizationStatus = dataclasses.field(
        default=QuantizationStatus.Unknown, repr=True, compare=False, hash=False
    )
    declaration_block: blocks.Block = dataclasses.field(
        repr=False, compare=False, hash=False, kw_only=True
    )
    declaration_node: libcst.SimpleStatementLine | libcst.BaseExpression | None = dataclasses.field(
        repr=False, compare=False, hash=False, default=None
    )

    def mark_quantized(self, quantized: bool = True) -> None:
        """Set quantized status."""
        status = QuantizationStatus.Quantized if quantized else QuantizationStatus.NotQuantized
        object.__setattr__(self, "quantization_status", status)


class VariableCollection:
    """A collection for variable creation.

    This collection is used to create new versions of variables. It ensures
    that variables have an increasing version number.
    """

    def __init__(self) -> None:
        self._variables: dict[str, list[Variable]] = {}

    def get(
        self,
        name: str,
        quantization_status: QuantizationStatus,
        declaration_block: blocks.Block,
        declaration_node: libcst.SimpleStatementLine | libcst.BaseExpression | None = None,
    ) -> Variable:
        """Get a new version of a variable with name `name`.

        Args:
            name: The name of the variable.
            quantization_status: Flag that indicates if variable is quantized,
                not quantized or if the status is unknown.
            declaration_block: The block in which the variable is declared.
            declaration_node: The statement in which the variable is declared.

        Returns:
            A new `Variable` instance for `name`.
        """
        variables = self._variables.setdefault(name, [])
        variables.append(
            Variable(
                name=name,
                version=len(variables),
                quantization_status=quantization_status,
                declaration_block=declaration_block,
                declaration_node=declaration_node,
            )
        )
        return variables[-1]


class VariableSet(Set[Variable]):
    """A set specific to Variables.

    Contains utility methods to deal with name/version combinations.
    """

    _variables: dict[str, dict[int, Variable]]

    def __init__(self) -> None:
        self._variables = {}

    def add(self, var: Variable) -> None:
        """Add a variable to the set.

        Args:
            var: `Variable` to add to the set.
        """
        if var.version is None:
            raise ValueError("var.version cannot be None")
        if not (version_set := self._variables.get(var.name)):
            version_set = self._variables[var.name] = {}
        if var.version not in version_set:
            version_set[var.version] = var

    def __getitem__(self, name: str) -> Iterator[Variable]:
        yield from self._variables.get(name, {}).values()

    def __copy__(self) -> Self:
        instance = type(self)()
        for var in self:
            instance.add(var)
        return instance

    @override
    def __len__(self) -> int:
        return sum(len(version_set) for version_set in self._variables.values())

    @override
    def __iter__(self) -> Iterator[Variable]:
        for version_set in self._variables.values():
            yield from version_set.values()

    @overload
    def contains(self, var: Variable, /, version: None = None) -> bool: ...
    @overload
    def contains(self, name: str, /, version: int | None = None) -> bool: ...

    def contains(self, var_or_name: Variable | str, /, version: int | None = None) -> bool:
        """Check if set contains variable."""
        if isinstance(var_or_name, Variable):
            name = var_or_name.name
            version = var_or_name.version
        else:
            name = var_or_name

        if name not in self._variables:
            return False
        return version is None or version in self._variables[name]

    @override
    def __contains__(self, object: Any) -> bool:
        if not isinstance(object, Variable):
            return False
        return self.contains(object)

    def versions_of(self, name: str) -> Iterator[int]:
        """Return iterator over all version of variable `name` in set."""
        if name not in self._variables:
            return
        yield from self._variables[name]

    @overload
    def remove(self, var: Variable, /, version: None = None) -> list[Variable]: ...
    @overload
    def remove(self, name: str, /, version: int | None = None) -> list[Variable]: ...

    def remove(self, var_or_name: Variable | str, /, version: int | None = None) -> list[Variable]:
        """Remove variable from set.

        Args:
            var_or_name: A `Variable` to remove or `str` that is the name of the variable to remove.
            version: If `var_or_name` is a `str`, the version of the variable
                to remove. Remove all variables that match a name if `version`
                is None.

        Returns:
            A list of variables removed from this set.
        """
        if isinstance(var_or_name, Variable):
            name = var_or_name.name
            version = var_or_name.version
        else:
            name = var_or_name

        # Collect removed variables in list instead of yielding to ensure that
        # the entire remove operation is executed -- even when the result is not
        # consumed.
        removed_vars: list[Variable] = []
        try:
            if version is None:
                for var in self._variables[name].values():
                    removed_vars.append(var)
                del self._variables[name]
            else:
                removed_vars.append(self._variables[name][version])
                del self._variables[name][version]
        except KeyError:
            # Variable is not part of set
            pass

        return removed_vars

    def subtract(self, other: "VariableSet", all_versions: bool = False) -> Self:
        """Create a new set that contains all elements in `self` that or not in `other`.

        Args:
            other: The set to base the new set on
            all_versions: If `True` remove all version of a variable if it is
                present in `other`.
        """
        new_set = copy.copy(self)
        return new_set.subtract_(other, all_versions=all_versions)

    def subtract_(self, other: "VariableSet", all_versions: bool = False) -> Self:
        """In-place variant of `subtract`."""
        for var in other:
            self.remove(var.name, version=None if all_versions else var.version)
        return self

    def union(self, *others: "VariableSet") -> Self:
        """Create a new set that is the union of `self` and `others`."""
        new_set = copy.copy(self)
        return new_set.union_(*others)

    def union_(self, *others: "VariableSet") -> Self:
        """In-place variant of `union`."""
        for other in others:
            for var in other:
                self.add(var)
        return self

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VariableSet):
            return NotImplemented
        return self._variables == other._variables

    @override
    def __repr__(self) -> str:
        names = (
            f"{v.name}:{v.version}"
            for version_set in self._variables.values()
            for v in version_set.values()
        )
        return f"{{{', '.join(names)}}}"

    def find_variables_for_node(self, node: libcst.CSTNode) -> Iterator[Variable]:
        """Yield all variables that define `node` as `declaration_node`."""
        for var in self:
            if var.declaration_node is node:
                yield var


@dataclasses.dataclass
class BlockTracker:
    """Collection of Variable sets used for tracking for an associated block.

    The collections contains four sets:
        - `vars_in`: The variables that reach the associated block via its
          parents.
        - `vars_out`: The variables that 'flow' to the children of the
          associated block.
        - `vars_gen`: Variables that are assigned in the associated block.
        - `vars_kill` Variables that are deleted in the associated block.
        - `vars_local` Variables that assigned in the associated block but are
            overridden or deleted before the end of the block.
    """

    vars_in: VariableSet = dataclasses.field(default_factory=VariableSet)
    vars_out: VariableSet = dataclasses.field(default_factory=VariableSet)
    vars_gen: VariableSet = dataclasses.field(default_factory=VariableSet)
    vars_kill: VariableSet = dataclasses.field(default_factory=VariableSet)
    vars_local: VariableSet = dataclasses.field(default_factory=VariableSet)


_T = TypeVar("_T", bound=Hashable)
_S = TypeVar("_S")


class _OrderedSet(Generic[_T]):
    """Simple ordered set.

    This ordered set only supports `pop`, `add` and `remove` and not other set
    operations.
    """

    def __init__(self, items: Iterable[_T] | None = None) -> None:
        self._storage: dict[_T, None] = {}
        if items is not None:
            for item in items:
                self.add(item)

    def pop(self, default: _S | EllipsisType = ...) -> _T | _S:
        """Pop element from the set in LIFO order.

        The order is determined based on when the elements was first added to the set, i.e.,
        re-adding an elements does not alter the pop order.

        Args:
            default: If the set is empty, return this element instead of
                raising an error.

        Returns:
            The element that was added to the set last, or `default` if the
            set is empty.
        """
        try:
            return self._storage.popitem()[0]
        except KeyError:
            if default is not ...:
                return default
            raise IndexError(f"pop from empty {type(self).__name__}") from None

    def __iter__(self) -> Iterator[_T]:
        yield from self._storage

    def add(self, elem: _T) -> None:
        """Add `elem` to the set.

        If `elem` is already in the set, the order of elements remains unchanged.
        """
        self._storage[elem] = None

    def remove(self, elem: _T) -> None:
        """Remove `elem` from the set.

        It is safe to call this method with an `elem` that is not a member of
        the set.
        """
        try:
            del self._storage[elem]
        except KeyError:
            pass

    @override
    def __repr__(self) -> str:
        elems = ", ".join(repr(e) for e in self)
        return f"{type(self).__name__}({{{elems}}})"


def get_quantization_status(expr: libcst.BaseExpression) -> QuantizationStatus:
    """Quantization status for the result of an expression.

    All expressions that are not a `QuantizedCall` have an implied quantization
    status `NotQuantized`. For a quantized call, the quantization status
    depends on the return type of the function.

    Args:
        expr: The expr to infer the quantization status of.

    Returns:
        `QauntizationStatus` of `expr`.
    """
    if not isinstance(expr, nodes.QuantizedCall):
        return QuantizationStatus.NotQuantized
    if expr.operator.returns_quantized:
        return QuantizationStatus.Quantized
    else:
        return QuantizationStatus.NotQuantized
