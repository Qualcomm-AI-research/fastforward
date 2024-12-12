# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import dataclasses

from collections import abc
from typing import Callable, Iterable, Iterator, Optional, TypeVar, overload

import torch

from typing_extensions import Self

from . import _parser as mpath_parser
from . import selector

_T = TypeVar("_T")


@dataclasses.dataclass(frozen=True)
class FilterResult:
    full_name: str
    module: torch.nn.Module
    parent: torch.nn.Module
    parent_attribute: str
    fragment_matches: tuple[selector.Fragment, ...] = dataclasses.field(compare=False)

    def update_module(self, replacement: torch.nn.Module, safe: bool = True) -> Self:
        if safe and getattr(self.parent, self.parent_attribute) is not self.module:
            raise ValueError(
                f"Trying to update module, but the parent attribute ({self.parent_attribute}) "
                "model no longer references the module of this filter result. Use `safe=False` "
                f"if you want to overwrite `parent.{self.parent_attribute}` nonetheless."
            )

        setattr(self.parent, self.parent_attribute, replacement)
        return dataclasses.replace(self, module=replacement)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.full_name}, type={self.module.__class__.__name__})"
        )


class MPathCollection(abc.Sequence[FilterResult]):
    """
    Collection for mpath search results as returned by `mpath.search`

    Args:
        root: The root module that was used to produce the search results
        results: Optional list of search results, may be extended using append
    """

    def __init__(
        self, root: torch.nn.Module, results: Optional[Iterable[FilterResult]] = None
    ) -> None:
        self._root = root
        self._results: list[FilterResult] = list(results) if results is not None else []

    @overload
    def __getitem__(self, key: int) -> FilterResult: ...

    @overload
    def __getitem__(self, key: slice) -> Self: ...

    def __getitem__(self, key: int | slice) -> FilterResult | Self:
        selection = self._results[key]
        if isinstance(selection, FilterResult):
            return selection
        else:
            return type(self)(self._root, selection)

    def __len__(self) -> int:
        return len(self._results)

    def append(self, item: FilterResult) -> None:
        """
        Add `FilterResult` to the collection:

        Args:
            item: Filter result to add to the collection.
        """
        self._results.append(item)

    def __repr__(self) -> str:
        if len(self._results) == 0:
            return f"{type(self).__name__}([])"
        elems = [
            f"    <{i}: {r.full_name}: {r.module.__class__.__name__}>,\n"
            for i, r in enumerate(self._results)
        ]
        return f"{type(self).__name__}([\n{''.join(elems)}])"

    def apply(
        self,
        func: Callable[[str, torch.nn.Module], torch.nn.Module | None],
        safe: bool = True,
        update: bool = True,
    ) -> None:
        """
        Apply `func` to all modules in the collection. If the result of func is
        not None and update is True, the module is replaced in the original
        model by the result of func. This only happens when the module in the
        collection was still part of the original model and not already
        changed. Use safe=False, to ignore this and always update the module
        independent of the current value.

        Args:
            func: Function to apply to all modules
            safe: Boolean indicating whether only safe updated are allowed
            update: Boolean indicating whether to replace original module with
                result of func if not the result is not None
        """
        _results: list[FilterResult] = []
        for result in self._results:
            if update and (new_module := func(result.full_name, result.module)) is not None:
                result = result.update_module(new_module, safe=safe)
            _results.append(result)
        self._results = _results

    def map(self, func: Callable[[str, torch.nn.Module], _T]) -> list[_T]:
        """
        Apply func to all modules in the collection, the result of func is
        returned as a list

        Args:
            func: Function to apply to each member of the collection

        Returns:
            list that contains all the results of the applications of func
        """

        return [func(*named_module) for named_module in self.named_modules()]

    def named_modules(self) -> Iterator[tuple[str, torch.nn.Module]]:
        """
        Returns An interator with name and modules of all modules in the collection.
        The name is relative with respect to the root module, i.e., the module that
        was queried to produce this collection.
        """
        yield from ((result.full_name, result.module) for result in self._results)

    def modules(self) -> Iterator[torch.nn.Module]:
        """
        Returns an iterator over all modules in this collection
        """
        for _, module in self.named_modules():
            yield module

    def parents(self) -> Iterator[tuple[torch.nn.Module, torch.nn.Module]]:
        """
        Returns and iterator that yield (parent, child) tuples in the collection.

        NB: Parent does not have to be part of the collection, but child always is.
        """
        yield from ((result.parent, result.module) for result in self._results)

    def _check_same_root(self, other: "MPathCollection", operation: str) -> None:
        if self._root is not other._root:
            raise ValueError(
                f"Can only create {operation} of {type(self).__name__}s that were "
                "created from the same model"
            )

    def __eq__(self, other: object, /) -> bool:
        if not isinstance(other, MPathCollection):
            return NotImplemented
        return self._root == other._root and self._results == other._results

    def union(self, other: "MPathCollection") -> Self:
        """
        Create a new `MPathCollection` that contains all elements from this
        collection and other. A ValueError is raised if the root elements of
        self and other are not the same.
        """
        self._check_same_root(other, "union")
        return type(self)(root=self._root, results=set(self._results) | set(other._results))

    def __or__(self, other: "MPathCollection") -> Self:
        if not isinstance(other, MPathCollection):
            return NotImplemented  # type: ignore[unreachable]
        return self.union(other)

    def intersection(self, other: "MPathCollection") -> Self:
        """
        Create a new `MPathCollection` that contains all elements that are
        present in this collection and other. A ValueError is raised if the
        root elements of self and other are not the same.
        """
        self._check_same_root(other, "intersection")
        return type(self)(root=self._root, results=set(self._results) & set(other._results))

    def __and__(self, other: "MPathCollection") -> Self:
        if not isinstance(other, MPathCollection):
            return NotImplemented  # type: ignore[unreachable]
        return self.intersection(other)

    def difference(self, other: "MPathCollection") -> Self:
        """
        Create a new `MPathCollection` that contains all elements in this
        collection that are not in other. A ValueError is raised if the root
        elements of self and other are not the same.
        """
        self._check_same_root(other, "union")
        return type(self)(root=self._root, results=set(self._results) - set(other._results))

    def __sub__(self, other: "MPathCollection") -> Self:
        if not isinstance(other, MPathCollection):
            return NotImplemented  # type: ignore[unreachable]
        return self.difference(other)

    def symmetric_difference(self, other: "MPathCollection") -> Self:
        """
        Create a new `MPathCollection` that contains all elements that a in this
        collection or in other, but not in both. A ValueError is raised if the root
        elements of self and other are not the same.
        """
        self._check_same_root(other, "union")
        return type(self)(root=self._root, results=set(self._results) ^ set(other._results))

    def __xor__(self, other: "MPathCollection") -> Self:
        if not isinstance(other, MPathCollection):
            return NotImplemented  # type: ignore[unreachable]
        return self.symmetric_difference(other)

    def __delitem__(self, idx: int) -> None:
        del self._results[idx]


@dataclasses.dataclass(frozen=True)
class _ActiveSearchItem:
    selector: selector.BaseSelector
    result: FilterResult


def _child_result(
    parent: FilterResult,
    module: torch.nn.Module,
    attribute_name: str,
    fragment: selector.Fragment,
) -> FilterResult:
    return FilterResult(
        full_name=(f"{parent.full_name}." if parent.full_name else "") + attribute_name,
        module=module,
        parent=parent.module,
        parent_attribute=attribute_name,
        fragment_matches=parent.fragment_matches + (fragment,),
    )


def search(
    query: selector.BaseSelector | str,
    root: torch.nn.Module,
    *,
    aliases: Optional[dict[str, selector.BaseSelector]] = None,
) -> MPathCollection:
    """
    Search/filter all submodules of `root` that satsify query.

    See `mpath.query` for more details on the string syntax for query or manually create
    a query `Selector` by combinding multiple fragments (e.g., from mpath.fragments).

    Args:
        query: Query to specify included/excluded submodules
        root: Root module, all submodules of the root module are considered for inclusion
            in the result set. root itself is never part of the result set.

    Returns:
        Collection of (sub)modules that satisfy query.
    """
    aliases = aliases or {}
    if isinstance(query, str):
        query = mpath_parser.parse(
            query, context=mpath_parser.get_caller_context(), aliases=aliases
        )
    query = query.simplify()

    results = MPathCollection(root)
    selected_modules: set[torch.nn.Module] = set()

    active_modules: list[_ActiveSearchItem] = [
        _ActiveSearchItem(query, FilterResult("", root, root, "", ()))
    ]

    explored = set([(query.fragment_list(), root)])

    while active_modules:
        active_item = active_modules.pop()
        selector = active_item.selector

        for child_name, child in active_item.result.module.named_children():
            matches, continuations = selector.matches_and_continuations(child_name, child)

            for match in matches:
                if child not in selected_modules:
                    selected_modules.add(child)
                    results.append(
                        _child_result(active_item.result, child, child_name, match.fragment)
                    )

            for continuation in continuations:
                if continuation.consume:
                    filter_result = _child_result(
                        active_item.result, child, child_name, continuation.fragment
                    )
                else:
                    active_result = active_item.result
                    filter_result = dataclasses.replace(
                        active_result,
                        fragment_matches=active_result.fragment_matches + (continuation.fragment,),
                    )

                marker = (continuation.selector.fragment_list(), filter_result.module)
                if marker not in explored:
                    explored.add(marker)
                    active_modules.append(_ActiveSearchItem(continuation.selector, filter_result))

    return results
