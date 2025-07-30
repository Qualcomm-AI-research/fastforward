# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


import dataclasses

from typing import Generic, Iterator, Literal, TypeAlias, TypeVar

import libcst

from typing_extensions import Self, overload

_MetadataT = TypeVar("_MetadataT")


@dataclasses.dataclass
class _Assignment(Generic[_MetadataT]):
    name: str
    producer: libcst.CSTNode = dataclasses.field(repr=False)
    metadata: _MetadataT
    uses: list[libcst.CSTNode] = dataclasses.field(default_factory=list[libcst.CSTNode])


@dataclasses.dataclass(repr=False)
class _Assignments(Generic[_MetadataT]):
    """A collection to store and manage assignments of a scope scope.

    This class provides methods to record assignments, merge assignments from
    other scopes, and retrieve assignments for a given variable or producer.

    Attributes:
        _assignments: A dictionary mapping variable names to dictionaries of
            producers and their corresponding quantization statuses.
    """

    _assignments: dict[str, dict[libcst.CSTNode, _Assignment[_MetadataT]]]

    def record_assignment(self, name: str, producer: libcst.CSTNode, metadata: _MetadataT) -> None:
        """Records an assignment in the current scope.

        This method updates the internal state of the _Assignments instance to
        reflect the assignment of a variable. It checks if the variable is
        already present in the scope, and if so, updates its quantization
        status. If the variable is not present, it adds a new entry to the
        scope.

        Args:
            name: The name of the variable being assigned.
            producer: The node that produced the assignment.
            metadata: Metadata related to the assignment.

        Returns:
            None
        """
        if name not in self._assignments:
            self._assignments[name] = {}

        if producer in self._assignments[name]:
            self.update_metadata(self._assignments[name][producer], metadata)

        else:
            self._set_status(_Assignment(name=name, producer=producer, metadata=metadata))

    def _set_status(self, status: _Assignment[_MetadataT], *, overwrite: bool = True) -> None:
        var, producer = status.name, status.producer
        if overwrite:
            self._assignments[var] = {producer: status}
        else:
            if var not in self._assignments:
                self._assignments[var] = {}

            current = self._assignments[var].get(producer)
            if current is not None and not self.can_overwrite_assignment(current, status):
                msg = f"A status was already recorded for ({var}, {type(producer)})"
                raise ValueError(msg)

            self._assignments[var][producer] = status

    def update_metadata(self, assignment: _Assignment[_MetadataT], metadata: _MetadataT) -> None:
        """Updates the metadata of an existing assignment.

        This method updates the metadata associated with an assignment. It replaces
        the current metadata with the provided metadata value.

        Args:
            assignment: The assignment whose metadata needs to be updated.
            metadata: The new metadata value to be assigned.
        """
        assignment.metadata = metadata

    def can_overwrite_assignment(
        self, current: _Assignment[_MetadataT], new: _Assignment[_MetadataT]
    ) -> bool:
        """Determines if a new assignment can overwrite an existing one.

        This method checks whether it's permissible to replace an existing assignment
        with a new one for the same variable and producer. The default implementation
        always returns True, allowing any assignment to be overwritten. Subclasses may
        override this method to implement more specific overwrite policies.

        Args:
            current: The existing assignment that would be overwritten.
            new: The new assignment that would replace the current one.

        Returns:
            True if the new assignment can overwrite the current one, False otherwise.
        """
        del current, new
        return True

    def can_merge_assignment(
        self, current: _Assignment[_MetadataT], new: _Assignment[_MetadataT]
    ) -> bool:
        """Determines if a new assignment can be merged with an existing one.

        This method checks whether it's permissible to merge a new assignment
        with an existing one for the same variable and producer. The default
        implementation always returns True, allowing any assignments to be merged.
        Subclasses may override this method to implement more specific merge policies.

        Args:
            current: The existing assignment that would be merged with.
            new: The new assignment that would be merged.

        Returns:
            True if the new assignment can be merged with the current one, False otherwise.
        """
        del current, new
        return True

    @overload
    def __getitem__(self, key: tuple[str, libcst.CSTNode]) -> _Assignment[_MetadataT]: ...

    @overload
    def __getitem__(self, key: str) -> Iterator[_Assignment[_MetadataT]]: ...

    def __getitem__(
        self, key: tuple[str, libcst.CSTNode] | str
    ) -> _Assignment[_MetadataT] | Iterator[_Assignment[_MetadataT]]:
        """Retrieves the quantization status for a given variable or producer.

        If a tuple of (variable name, producer) is provided, this method returns the
        corresponding _QuantizationStatus instance. If a string (variable name) is
        provided, this method returns an iterator over all _QuantizationStatus instances
        for the given variable.

        Args:
            key: A tuple of (variable name, producer) or a string (variable name).

        Returns:
            A _QuantizationStatus instance or an iterator over _QuantizationStatus instances.
        """
        match key:
            case str():
                return iter(self._assignments.get(key, {}).values())
            case (str(), libcst.CSTNode()):
                var, producer = key
                try:
                    return self._assignments[var][producer]
                except KeyError as e:
                    msg = f"({var}, {type(producer)}) was not recorded as assignment"
                    raise KeyError(msg) from e
            case _:
                msg = "Got unsupported key"
                raise KeyError(msg)

    def __contains__(self, key: tuple[str, libcst.CSTNode] | str) -> bool:
        match key:
            case str():
                return key in self._assignments
            case (str(), libcst.CSTNode()):
                var, producer = key
                return var in self._assignments and producer in self._assignments[var]
            case _:
                return False

    def __iter__(self) -> Iterator[_Assignment[_MetadataT]]:
        for producers in self._assignments.values():
            yield from producers.values()

    @property
    def variables(self) -> Iterator[str]:
        """Return an iterator over the variable names in the current scope.

        This property provides a way to access the variable names that are being tracked
        in the current scope. It yields each variable name as a string.

        Yields:
            str: The name of a variable in the current scope.
        """
        yield from self._assignments

    def clone(self) -> Self:
        """Create a deep copy of the current _Assignments instance.

        This method is used to create a new, independent copy of the current scope.

        Returns:
            A new _Assignments instance that is a deep copy of the current instance.
        """
        assignments = {var: {**self._assignments[var]} for var in self._assignments}
        return type(self)(_assignments=assignments)

    def merge(self, other: Self) -> Self:
        """Merge the assignments from another _Assignments and this instance into a new instance."""
        clone = self.clone()
        clone.merge_(other)
        return clone

    def merge_(self, other: Self) -> None:
        """Merge the assignments from another _Assignments instance into the current instance.

        This method updates the internal state of the current _Assignments instance to reflect
        the assignments from the other instance. It checks for conflicts between the two sets
        of assignments and updates the quantization status accordingly.

        Args:
            other: The _Assignments instance to merge into the current instance.
        """
        for status in other:
            key = (status.name, status.producer)
            if key not in self:
                self._set_status(status, overwrite=False)
            elif not self.can_merge_assignment(self[key], status):
                msg = (
                    "Tried to merge two assignments with mismatched quantization status for "
                    + "the same variable name and producer"
                )
                raise RuntimeError(msg)

    def overwrite(self, other: Self) -> None:
        """Overwrite the current _Assignments instance with the assignments from another instance.

        This method updates the internal state of the current _Assignments
        instance to reflect the assignments from the other instance. It
        replaces the current assignments with the ones from the other instance.

        Args:
            other: The _Assignments instance to overwrite the current instance with.
        """
        for var, producers in other._assignments.items():
            self._assignments[var] = {**producers}


TerminationStatus: TypeAlias = Literal["return"] | Literal["break"] | Literal["raise"] | None


@dataclasses.dataclass(repr=False)
class Scope(Generic[_MetadataT]):
    """Represents a scope in the code, a region of the code where variables are defined and used.

    A scope can have a parent scope, which is the scope that contains it. It
    can also have a set of assignments, which are the variables that are
    defined in the scope.

    The scope also keeps track of whether it is a looping branch, which means
    it is a scope that is inside a loop. It also keeps track of whether
    repeated evaluation is enabled, which means that the scope is being
    evaluated multiple times.

    The scope can be terminated, which means that it is no longer active. This
    can happen when a return or break statement is encountered. The exact
    handling for break and return terminated scopes is different.

    Attributes:
        parent: The parent scope of this scope.
        assignments: The assignments in this scope.
        is_looping_branch: Whether this scope is a looping branch.
        repeated_evaluation: Whether repeated evaluation is enabled for this scope.
        _termination_status: The termination status of this scope.
        _breaking_scopes: The scopes that are broken out of.
    """

    parent: "Self | None" = None
    assignments: _Assignments[_MetadataT] = dataclasses.field(
        default_factory=lambda: _Assignments({})
    )
    is_looping_branch: bool = False
    repeated_evaluation: bool = False
    _termination_status: TerminationStatus = dataclasses.field(default=None)
    _breaking_scopes: list[Self] = dataclasses.field(default_factory=list[Self])

    @property
    def is_terminated(self) -> bool:
        return self._termination_status is not None

    def clone(self) -> Self:
        """Creates a deep copy of the current scope.

        Returns:
            A new Scope instance that is a deep copy of the current instance.
        """
        return dataclasses.replace(self, assignments=self.assignments.clone())

    def record_assignment(self, name: str, producer: libcst.CSTNode, metadata: _MetadataT) -> None:
        """Records an assignment in the current scope.

        Args:
            name: The name of the variable being assigned.
            producer: The node that produced the assignment.
            metadata: Metadata associated with the assignment.
        """
        if self._termination_status is not None:
            msg = "Trying to record assignment in terminated scope"
            raise RuntimeError(msg)
        self.assignments.record_assignment(name, producer, metadata)

    def terminate(
        self, reason: Literal["return"] | Literal["break"] | Literal["raise"] = "return"
    ) -> None:
        """Terminates the current scope.

        Args:
            reason: The reason for termination.
        """
        self._termination_status = reason
        if reason == "break":
            self._resolve_scope_break()

    def _resolve_scope_break(self) -> None:
        """Resolves a scope break.

        This method is called when a break statement is encountered in a loop.
        It resolves the scope break by merging the broken scope into a 'predecessor'
        scope.
        """
        # 'break' only occurs in loops, which are evaluated twice. We resolve
        # break-terminated scopes only on the second pass, when quantization
        # info is accurate.
        #
        # The scope where a 'break' occurs is attached to the nearest loop
        # scope and merged into its parent after the loop, ensuring variables
        # from the broken scope remain visible post-loop, but not in the remainder
        # of the loop.
        if self.repeated_evaluation:
            scope = self
            while not scope.is_looping_branch:
                if (scope_ := scope.parent) is None:
                    msg = "Encountered break outside of loop"
                    raise RuntimeError(msg)
                scope = scope_
            patched_scope = dataclasses.replace(
                self, assignments=self.assignments, parent=scope.parent
            )
            scope._breaking_scopes.append(patched_scope)
        self.assignments = _Assignments({})

    def __getitem__(self, key: str) -> Iterator[_Assignment[_MetadataT]]:
        """Retrieves the quantization status for a given variable.

        Args:
            key (str): The name of the variable.

        Yields:
            Iterator[_QuantizationStatus]: An iterator over the quantization statuses for the given variable.
        """
        yield from self.assignments[key]
        if (parent := self.parent) is not None:
            yield from parent[key]

    def merge(self, other: Self, *, inplace: bool = False) -> Self:
        """Merges the current scope with another scope.

        `other` can be a `Scope` with the same parent or with this scope as
        parent. I.e., `scope1.merge(scope2) != scope2.merge(scope1)`.

        Args:
            other: The scope to merge with.
            inplace: Whether to merge in place. Defaults to False.

        Returns:
            Scope: The merged scope.
        """
        if self.parent is not other.parent and other.parent is not self:
            msg = "Cannot merges scopes with different parent scopes"
            raise ValueError(msg)

        def _no_merge_required(scope: Self) -> Self:
            return scope if inplace else scope.clone()

        match self._termination_status, other._termination_status:
            case "return" | "raise", "return" | "raise":
                return type(self)(parent=self.parent)
            case _, "return" | "raise":
                return _no_merge_required(self)
            case "return" | "raise", _:
                return _no_merge_required(other)
            case _:
                return self._merge(other, inplace=inplace)

    def _merge(self, other: Self, inplace: bool = False) -> Self:
        """Merges the current scope with another scope.

        Args:
            other: The scope to merge with.
            inplace: Whether to merge in place. Defaults to False.

        Returns:
            Scope: The merged scope.
        """
        assert self.parent is other.parent or other.parent is self

        self_status, other_status = self._termination_status, other._termination_status
        assert self_status != "return"
        assert other_status != "return"

        if inplace:
            self.assignments.merge_(other.assignments)
            merged_scope = self
        else:
            assignments = self.assignments.merge(other.assignments)
            merged_scope = dataclasses.replace(self, assignments=assignments)

        if other.is_looping_branch:
            for breaking_scope in other._breaking_scopes:
                merged_scope.merge(breaking_scope, inplace=True)

        return merged_scope

    def overwrite(self, other: Self) -> None:
        """Overwrites the current scope with another scope.

        Args:
            other: The scope to overwrite with.
        """
        if self.parent is not other.parent and other.parent is not self:
            msg = "Cannot overwrite scope with scope that has different parent scopes"
            raise ValueError(msg)
        self.assignments.overwrite(other.assignments)


@dataclasses.dataclass
class QuantizationMetadata:
    is_quantized: bool


@dataclasses.dataclass(repr=False)
class _QuantizedAssignments(_Assignments[QuantizationMetadata]):
    def update_metadata(
        self, assignment: _Assignment[QuantizationMetadata], metadata: QuantizationMetadata
    ) -> None:
        assignment.metadata.is_quantized |= metadata.is_quantized

    def can_overwrite_assignment(
        self, current: _Assignment[QuantizationMetadata], new: _Assignment[QuantizationMetadata]
    ) -> bool:
        return current.metadata == new.metadata

    def can_merge_assignment(
        self, current: _Assignment[QuantizationMetadata], new: _Assignment[QuantizationMetadata]
    ) -> bool:
        return current.metadata == new.metadata


@dataclasses.dataclass(repr=False)
class QuantizationScope(Scope[QuantizationMetadata]):
    assignments: _Assignments[QuantizationMetadata] = dataclasses.field(
        default_factory=lambda: _QuantizedAssignments({})
    )
