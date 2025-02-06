# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import re

from typing import Any, Callable

import torch

from typing_extensions import override

from .selector import Fragment, MPathQueryError, Selector


class WildcardFragment(Fragment):
    """Match any fragment.

    Matchses any fragment exactly once if match_multiple=True and match any
    fragment zero or more times otherwise.
    """

    def __init__(self, match_multiple: bool = True) -> None:
        self.match_multiple = match_multiple

    @override
    def must_advance(self) -> bool:
        return not self.match_multiple

    @override
    def match(self, fragment_name: str, module: torch.nn.Module) -> bool:
        return True

    @override
    def __str__(self) -> str:
        return "*" if not self.match_multiple else "**"


class PathFragment(Fragment):
    """Match a specific path.

    Match the fragment name (i.e., module name in parent module) exactly.
    """

    def __init__(self, fragment_str: str) -> None:
        super().__init__()
        self._fragment_str = fragment_str

    @override
    def match(self, fragment_name: str, module: torch.nn.Module) -> bool:
        return fragment_name == self._fragment_str

    @override
    def __str__(self) -> str:
        return self._fragment_str

    @override
    def __hash__(self) -> int:
        return hash(("__mpath_fragment_hash", type(self), self._fragment_str))


class RegexPathFragment(Fragment):
    """Match a specific regex.

    Match a regex against the fragment name (i.e., module name in parent
    module). The regex must match the full name, otherwise it evaluates.
    """

    def __init__(self, fragment_pattern: str) -> None:
        super().__init__()
        try:
            self._matcher = re.compile(fragment_pattern)
        except re.error as e:
            raise MPathQueryError(f"{fragment_pattern} is not a valid regex pattern") from e

    @classmethod
    def from_raw_string(cls, raw_str: str) -> Selector:
        """Create a `RegexPathFragment` from `raw_str`."""
        re_pattern = raw_str.replace(r"\[", "[").replace(r"\]", "]")
        fragment = cls(re_pattern)
        return Selector(None, fragment)

    @override
    def match(self, fragment_name: str, module: torch.nn.Module) -> bool:
        is_match = self._matcher.fullmatch(fragment_name) is not None
        return is_match

    @override
    def __str__(self) -> str:
        return f"[re:{self._matcher.pattern}]"

    @override
    def __hash__(self) -> int:
        return hash(("__mpath_fragment_hash", type(self), self._matcher))


class ClassFragment(Fragment):
    """Match a fragment if the module is an instance of fragment_class."""

    def __init__(self, fragment_class: type) -> None:
        super().__init__()
        self._fragment_class = fragment_class

    @classmethod
    def from_raw_string_with_context(cls, raw_str: str, context: dict[str, Any]) -> Selector:
        """Find an object in context, by name, that matches `raw_str`.

        `raw_str` may be qualified path (i.e., period separated string) that is
        not directly available in `context`, but is indirectly through
        attribute lookups.

        For example, if `raw_str = "my_module.MyClass.my_attribute", we first
        look up `my_module` in `context`. Then we perform an attribute lookup
        for `MyClass` on `my_module` and lastly perform an attribute lookup of
        `my_attribute` on `MyClass`.
        """
        root, *tail = raw_str.split(".")

        try:
            obj = context[root]
        except KeyError:
            raise NameError(f"name '{root}' is not defined")

        for key in tail:
            obj = getattr(obj, key)

        fragment = cls(fragment_class=obj)
        return Selector(None, fragment)

    @override
    def match(self, fragment_name: str, module: torch.nn.Module) -> bool:
        is_match = isinstance(module, self._fragment_class)
        return is_match

    @override
    def __str__(self) -> str:
        return f"[cls:{self._fragment_class.__name__}]"

    @override
    def __hash__(self) -> int:
        return hash(("__mpath_fragment_hash", type(self), self._fragment_class))


class PredicateFragment(Fragment):
    """Match a fragment if predicate evaluates to True."""

    def __init__(self, predicate: Callable[[str, torch.nn.Module], bool]) -> None:
        super().__init__()
        self._predicate = predicate

    @override
    def match(self, fragment_name: str, module: torch.nn.Module) -> bool:
        return self._predicate(fragment_name, module)

    @override
    def __str__(self) -> str:
        return f"<predicate: {self._predicate.__name__}>"

    @override
    def __hash__(self) -> int:
        return hash(("__mpath_fragment_hash", type(self), self._predicate))


class JointFragment(Fragment):
    """Match a fragment if all fragments evaluate to True."""

    def __init__(self, *fragments: Fragment) -> None:
        super().__init__()
        self._selector_fragments = fragments

    @override
    def must_advance(self) -> bool:
        return all(fragment.must_advance for fragment in self._selector_fragments)

    @override
    def match(self, fragment_name: str, module: torch.nn.Module) -> bool:
        for fragment in self._selector_fragments:
            if not fragment.match(fragment_name, module):
                return False
        return True

    @property
    def fragments(self) -> tuple[Fragment, ...]:
        """Fragments that make up this joint fragment."""
        return self._selector_fragments

    @override
    def __str__(self) -> str:
        conjunction = " & ".join(str(fragment) for fragment in self._selector_fragments)
        return f"<{conjunction}>"

    @override
    def __hash__(self) -> int:
        return hash(("__mpath_fragment_hash", type(self), self._selector_fragments))


class DisjointFragment(Fragment):
    """Match a fragment if one of fragments evaluates to True."""

    def __init__(self, *fragments: Fragment) -> None:
        super().__init__()
        self._selector_fragments = fragments

    @override
    def must_advance(self) -> bool:
        return all(fragment.must_advance for fragment in self._selector_fragments)

    @override
    def match(self, fragment_name: str, module: torch.nn.Module) -> bool:
        for fragment in self._selector_fragments:
            if fragment.match(fragment_name, module):
                return True
        return False

    @override
    def __str__(self) -> str:
        disjunction = " | ".join(str(fragment) for fragment in self._selector_fragments)
        return f"<{disjunction}>"

    @override
    def __hash__(self) -> int:
        return hash(("__mpath_fragment_hash", type(self), self._selector_fragments))


class InvertedFragment(Fragment):
    """Match a fragment if fragment evaluates to False."""

    def __init__(self, fragment: Fragment) -> None:
        super().__init__()
        self._selector_fragment = fragment

    @override
    def must_advance(self) -> bool:
        return self._selector_fragment.must_advance()

    @override
    def match(self, fragment_name: str, module: torch.nn.Module) -> bool:
        return not self._selector_fragment.match(fragment_name, module)

    @override
    def __str__(self) -> str:
        return f"~{str(self._selector_fragment)}"

    @override
    def __hash__(self) -> int:
        return hash(("__mpath_fragment_hash", type(self), self._selector_fragment))
