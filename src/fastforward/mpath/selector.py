# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import abc
import dataclasses
import functools
import itertools

from operator import truediv
from typing import TYPE_CHECKING, Iterator, TypeAlias, TypeVar, overload

import torch

from typing_extensions import Self

if TYPE_CHECKING:
    from .fragments import InvertedFragment


class MPathQueryError(Exception):
    """Exception raised for errors in the MPath query."""


@dataclasses.dataclass(frozen=True)
class _SelectorMatch:
    """Represents a match for a selector fragment.

    Attributes:
        fragment: The matched fragment.
    """

    fragment: "Fragment"


@dataclasses.dataclass(frozen=True)
class _SelectorContinuation:
    """Represents a continuation for a selector fragment.

    Attributes:
        selector: The next selector to continue matching.
        fragment: The current fragment.
        consume: Flag indicating whether to consume the fragment.
    """

    selector: "BaseSelector"
    fragment: "Fragment"
    consume: bool = True


FragmentList: TypeAlias = "tuple[Fragment | FragmentList, ...]"


@dataclasses.dataclass(frozen=True, repr=False)
class BaseSelector(abc.ABC):
    """Selector is the representation of MPath query.

    A selector consists of a sequence of SelectorFragments that all need to
    match for a module to be included in the results set.

    Selectors can be appended through using various methods, but we also
    support concattenating two selectors using 'selector_a / selector_b'. The
    order of fragments is kept unchanged. I.e., in this example the new
    selector contains first all the the fragments from selector_a and then all
    the fragments from selector_b. Hence, this syntax can be used to prepend a
    (Selector)Fragment to another.

    When the Selector has length one (i.e., it contains only a single
    fragment), the '&' operator can be used to produce Selector with a single
    joint fragment. In all other cases, using these operators results in an
    exception.

    Selectors of arbitrary length can be joined using the '|' operator to
    produce a selector that will match when any any of the selectors match.
    """

    next: "BaseSelector | None"

    def __truediv__(self, rhs: "BaseSelector | str") -> Self:
        if isinstance(rhs, str):
            from fastforward import mpath

            rhs = mpath.query(rhs, context=mpath.caller_context())
        return self.extends(rhs)

    def __rtruediv__(self, lhs: str) -> "BaseSelector":
        from fastforward import mpath

        root = mpath.query(lhs, context=mpath.caller_context())
        return root.extends(self)

    def __and__(self, rhs: object) -> Self:
        return NotImplemented

    def extends(self, selector: "BaseSelector") -> Self:
        """Extend the current selector with another selector.

        This method is used to concatenate two selectors. The order of fragments
        is kept unchanged, i.e., the new selector contains first all the fragments
        from the current selector and then all the fragments from the given selector.

        Args:
            selector: The selector to extend with.

        Returns:
            A new selector that is the concatenation of the current selector and the given selector.
        """
        return _duplicate_extend(self, selector)

    def __or__(self, rhs: "BaseSelector") -> "MultiSelector":
        return MultiSelector(None, (self, rhs))

    def __invert__(self) -> "BaseSelector":
        msg = f"Cannot invert {type(self).__name__}"
        raise MPathQueryError(msg)

    def __getitem__(self, key: int | slice) -> "BaseSelector":
        if isinstance(key, slice):
            raise IndexError("slice indexes are not supported")
        if key == 0:
            return dataclasses.replace(self, next=None)
        elif self.next:
            return self.next.__getitem__(key - 1)
        else:
            raise IndexError("Selector index out of range")

    def __iter__(self) -> Iterator["BaseSelector"]:
        yield self
        if self.next:
            yield from self.next

    def __str__(self) -> str:
        head = "<selector>"
        if self.next:
            return f"{head}/{str(self.next)}"
        return head

    def __repr__(self) -> str:
        return f"<selector: {str(self)}>"

    def __len__(self) -> int:
        if self.next:
            return len(self.next) + 1
        return 1

    def _selectors(self) -> list["BaseSelector"]:
        return [dataclasses.replace(self, next=None)]

    def _selector_lists(
        self, *, head: list[list["BaseSelector"]] | None = None
    ) -> list[list["BaseSelector"]]:
        head = (head or []) + [self._selectors()]
        if self.next:
            return self.next._selector_lists(head=head)
        return head

    def _expand(self) -> Iterator["BaseSelector"]:
        for selectors in itertools.product(*self._selector_lists()):
            if len(selectors) == 0:
                continue
            yield functools.reduce(truediv, selectors[1:], selectors[0])

    @abc.abstractmethod
    def matches_and_continuations(
        self, name: str, module: torch.nn.Module
    ) -> tuple[list[_SelectorMatch], list[_SelectorContinuation]]:
        """Returns a list of selector matches and a list of selector continuations.

        Matches represent a complete matching, i.e., there is no selector tail and the
        current fragment matched name/module.

        Continuations represent partial matches, i.e., the current fragment matched,
        and the tail of the selector should be considered for further matching.
        """

    @abc.abstractmethod
    def fragments(self) -> tuple["Fragment", ...]:
        """Return a list of fragments that make up this selector.

        Returns:
            A list of fragments that make up this selector.
        """

    @abc.abstractmethod
    def fragment_list(self) -> FragmentList:
        """Return a list of fragments that make up this selector and all subsequent selectors."""

    @abc.abstractmethod
    def remove_multi_wildcard_root(self) -> "BaseSelector | None":
        """Remove the multi-wildcard root from the selector.

        Returns:
            The selector with the multi-wildcard root removed, or None if the selector is made up of only a multi-wildcard fragment
        """

    @abc.abstractmethod
    def has_multi_wildcard_root(self) -> bool:
        """Return True if the root fragment is a multi-wildcard fragment and False otherwise."""

    def simplify(self) -> "BaseSelector":
        """Simplify the selector by removing any redundant or unnecessary fragments.

        This method is used to optimize the selector for better performance.

        Returns:
            A simplified version of the selector.
        """
        selector = self
        if self.has_multi_wildcard_root():
            if simplified_selector := self.remove_multi_wildcard_root():
                selector = "**" / simplified_selector
            else:
                from .fragments import WildcardFragment

                # Selector is made up of only wildcard fragments.
                selector = Selector(next=None, fragment=WildcardFragment(match_multiple=True))

        return selector


@dataclasses.dataclass(frozen=True, repr=False)
class Selector(BaseSelector):
    """Represents a single fragment selector in an MPath query.

    Attributes:
        fragment: The fragment to match.
    """

    fragment: "Fragment"

    def __invert__(self) -> Self:
        """Invert the selector.

        Returns:
            Self: The inverted selector.

        Raises:
            MPathQueryError: If the selector contains more than one fragment.
        """
        if self.next:
            msg = f"Cannot invert {type(self).__name__} that contains more than one fragment"
            raise MPathQueryError(msg)
        return dataclasses.replace(self, fragment=~self.fragment)

    def __and__(self, rhs: object) -> Self:
        """Combine this selector with another using the '&' operator.

        Args:
            rhs: The other selector.

        Returns:
            Self: The combined selector.

        Raises:
            MPathQueryError: If either selector contains more than one fragment.
        """
        from .fragments import JointFragment  # avoid recursive imports

        if not isinstance(rhs, Selector):
            return NotImplemented
        if self.next or rhs.next:
            raise MPathQueryError("Cannot join selectors of size greater than 1")

        return type(self)(None, JointFragment(self.fragment, rhs.fragment))

    def __str__(self) -> str:
        """Return a string representation of the selector.

        Returns:
            A string representation of the selector.
        """
        head = str(self.fragment)
        if self.next:
            return f"{head}/{str(self.next)}"
        return head

    def has_multi_wildcard_root(self) -> bool:
        """Check if the selector has a multi-wildcard root.

        Returns:
            bool: True if the selector has a multi-wildcard root, False otherwise.
        """
        from .fragments import WildcardFragment

        return isinstance(self.fragment, WildcardFragment) and self.fragment.match_multiple

    def remove_multi_wildcard_root(self) -> BaseSelector | None:
        """Remove the multi-wildcard root from the selector.

        Returns:
            BaseSelector | None: The selector without the multi-wildcard root, or None if not applicable.
        """
        if not self.has_multi_wildcard_root():
            return self
        if next := self.next:
            next = self.next.remove_multi_wildcard_root()
        return next

    def matches_and_continuations(
        self, name: str, module: torch.nn.Module
    ) -> tuple[list[_SelectorMatch], list[_SelectorContinuation]]:
        """Return a list of selector matches and continuations.

        Args:
            name: The name of the module.
            module: The module to match.

        Returns:
            tuple[list[SelectorMatch], list[SelectorContinuation]]: A tuple containing
            a list of matches and a list of continuations.
        """
        if not self.fragment.match(name, module):
            return [], []

        continuations = []
        if not self.fragment.must_advance():
            # The current fragment can match multiple times, so we should
            # add it as a continuation.
            continuations.append(_SelectorContinuation(self, self.fragment))

            # Zero match case for wildcard fragment
            if self.next:
                continuations.append(
                    _SelectorContinuation(self.next.simplify(), self.fragment, consume=False)
                )

        # We have not consumed the entire selector, add a continuation
        # for the tail
        if self.next:
            continuations.append(_SelectorContinuation(self.next.simplify(), self.fragment))

        # If there is no next, we have matched, but we can pottently match
        # more if the last fragment can be applied again.
        if not self.next:
            return [_SelectorMatch(self.fragment)], continuations

        # If self.next is equivalent to '**', we do have a match as further
        # fragments don't have to match, otherwise there is no match at this
        # point
        if _is_multi_wildcard(self.next):
            return [_SelectorMatch(self.fragment)], continuations
        else:
            return [], continuations

    def fragments(self) -> tuple["Fragment", ...]:
        """Return a list of fragments in the selector.

        Returns:
            list[Fragment]: A list of fragments.
        """
        if self.next:
            return (self.fragment,) + self.next.fragments()
        return (self.fragment,)

    def fragment_list(self) -> FragmentList:
        """Return a nested list of fragments in the selector.

        Returns:
            FragmentList: A nested list of fragments.
        """
        if self.next:
            return (self.fragment,) + self.next.fragment_list()
        return (self.fragment,)


@dataclasses.dataclass(frozen=True, repr=False)
class MultiSelector(BaseSelector):
    """Represents a multi-fragment selector in an MPath query.

    Attributes:
        selectors: The selectors to match.
    """

    selectors: tuple[BaseSelector, ...]

    def __str__(self) -> str:
        """Return a string representation of the multi-selector.

        Returns:
            str: A string representation of the multi-selector.
        """
        selectors = ", ".join(str(selector) for selector in self.selectors)
        head = f"{{{selectors}}}"
        if self.next:
            return f"{head}/{str(self.next)}"
        return head

    def _selectors(self) -> list[BaseSelector]:
        """Return a list of expanded selectors.

        Returns:
            list[BaseSelector]: A list of expanded selectors.
        """
        selectors: list[BaseSelector] = []
        for s in self.selectors:
            selectors += s._expand()
        return selectors

    def matches_and_continuations(
        self, name: str, module: torch.nn.Module
    ) -> tuple[list[_SelectorMatch], list[_SelectorContinuation]]:
        """Return a list of selector matches and continuations.

        Args:
            name: The name of the module.
            module: The module to match.

        Returns:
            tuple[list[SelectorMatch], list[SelectorContinuation]]: A tuple containing
            a list of matches and a list of continuations.
        """
        all_matches: list[_SelectorMatch] = []
        all_continuations: list[_SelectorContinuation] = []
        for selector in self.selectors:
            extended = _duplicate_extend(selector, self.next)
            matches, continuations = extended.matches_and_continuations(name, module)
            all_matches += matches
            all_continuations += continuations
        return all_matches, all_continuations

    def fragments(self) -> tuple["Fragment", ...]:
        """Return a list of fragments in the multi-selector.

        Returns:
            list[Fragment]: A list of fragments.
        """
        all_fragments: tuple["Fragment", ...] = ()
        for selector in self.selectors:
            all_fragments += selector.fragments()
        return all_fragments

    def fragment_list(self) -> FragmentList:
        """Return a nested list of fragments in the multi-selector.

        Returns:
            FragmentList: A nested list of fragments.
        """
        fragment_list: FragmentList = ()
        for selector in self.selectors:
            fragment_list += selector.fragment_list()
        fragment_list = (fragment_list,)
        if self.next:
            fragment_list += self.next.fragment_list()
        return fragment_list

    def remove_multi_wildcard_root(self) -> "BaseSelector | None":
        """Remove the multi-wildcard root from the multi-selector.

        Returns:
            BaseSelector | None: The multi-selector without the multi-wildcard root, or None if not applicable.
        """
        selectors = []
        for selector in self.selectors:
            if updated_selector := selector.remove_multi_wildcard_root():
                selectors.append(updated_selector)
        return type(self)(self.next, tuple(selectors))

    def has_multi_wildcard_root(self) -> bool:
        """Check if all selectors have a multi-wildcard root.

        Returns:
            bool: True if all selectors have a multi-wildcard root, False otherwise.
        """
        return all(selector.has_multi_wildcard_root() for selector in self.selectors)


def _is_multi_wildcard(selector: BaseSelector) -> bool:
    """Check if the selector is equivalent to `**`, i.e., all remaining fragments match zero or more arbitrary modules.

    Args:
        selector: The selector to check.

    Returns:
        bool: True if the selector is equivalent to `**`, False otherwise.
    """
    from .fragments import WildcardFragment

    fragments = selector.fragments()
    return all(
        isinstance(fragment, WildcardFragment) and fragment.match_multiple for fragment in fragments
    )


class Fragment(abc.ABC):
    """A selector fragment is a matcher for a single element in a module path.

    Here a module path is the sequence of module names and modules through
    which one would access a specific module from a root.

    For example, consider the following module

        RootModule(
            (first): SubModule(
                (second): Linear()
                (third): Conv2d()
            )
        )

    A module path to the linear module 'second' is: `("first", SubModule),
    ("second", Linear)` where `SubModule` and `Linear` refer to the specific
    instances in this module. A selector fragment matches one of the elements
    (or fragments) on a module path.

    A Fragment matches a single element. If it matches, the children of
    the module that matched will be considered for matching the next fragment, or
    if there is no next fragment, the module is included in the result set.

    An implementation of Fragment must implement the `match` function and
    optionally the `must_advance` function.

    - The match function accepts a fragment_name and a module. The fragment
      name is the attribute name of the current module in the module path.
      The module is the actual model. Using this information, match decides if
      the module matches the Fragment.
    - the must_advance function allows for 'wildcard-like' expansion. If must_advance
      returns False, the current module is also included in the expansion set. If it
      returns True, only the children of the current module are included.

    Example of fragment implementations can be found in `mpath.fragments`.

    Fragments can be used to create a query programmatically. This may be
    useful when you want to include more context in the Fragment that
    would not be possible through the standard query string. Two (or more)
    Fragments can form a Selector using the '/' operator (which mirrors
    the use of '/' in query strings). Moreover, higher level joint and disjoint
    fragments can be created using the '&' and '|' operators.
    """

    @abc.abstractmethod
    def match(self, fragment_name: str, module: torch.nn.Module) -> bool:
        """Matches a single fragment of a path on name or module.

        Args:
            fragment_name: The name of the path, corresponds to the attribute
                name on the 'parent' object.
            module: The module that corresponds to the fragment name.

        Returns: Boolean indicating whether the fragment matches the current position
        """
        return False

    def must_advance(self) -> bool:
        """Return whether the current fragment can be repeated.

        Returns a boolean that indicates if the current module can be considered
        for the next fragment. This allows implement wildcard like FragmentSelectors.

        Returns:
            True if the current module is __not__ included in the expansion set
            and False if it can be included.
        """
        return True

    def __invert__(self) -> "InvertedFragment":
        from .fragments import InvertedFragment

        return InvertedFragment(self)


_Selector = TypeVar("_Selector", bound=BaseSelector)
_Ext = TypeVar("_Ext", bound=BaseSelector)


@overload
def _duplicate_extend(selector: _Selector, ext: BaseSelector | None) -> _Selector: ...
@overload
def _duplicate_extend(selector: None, ext: _Ext) -> _Ext: ...
@overload
def _duplicate_extend(selector: None, ext: None) -> None: ...


def _duplicate_extend(selector: _Selector | None, ext: _Ext | None) -> _Selector | _Ext | None:
    if selector is None:
        return ext
    return dataclasses.replace(selector, next=_duplicate_extend(selector.next, ext))
