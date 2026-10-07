# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Type metadata shared by native and remote type-checker adapters."""

import dataclasses
import re

from typing import Any, Protocol, TypeVar

_TypeT = TypeVar("_TypeT", contravariant=True)


class TypeOperations(Protocol[_TypeT]):
    """Interpret a backend's type representation without exposing it to passes."""

    def is_proven_scalar(self, typ: _TypeT) -> bool:
        """Return whether every possible value has a supported numeric type."""
        ...

    def is_subtype(self, typ: _TypeT, qualified_name: str) -> bool:
        """Check membership in the requested type."""
        ...

    def is_sequence_of_subtype(self, typ: _TypeT, qualified_name: str) -> bool:
        """Check membership in a sequence of the requested type."""
        ...


@dataclasses.dataclass
class TypeInfo:
    """Shared metadata with backend-specific type operations.

    The opaque type is retained for diagnostics. Transformation passes use
    the predicates, so a native checker can preserve its subtype semantics
    without serializing its internal objects into the TSP wire format.
    """

    typ: Any
    _operations: TypeOperations[Any] = dataclasses.field(repr=False)

    def is_proven_scalar(self) -> bool:
        """Return whether the backend has positive evidence of a scalar."""
        return self._operations.is_proven_scalar(self.typ)

    def is_subtype(self, qualified_name: str) -> bool:
        """Check whether the type is a subtype of the requested type."""
        return self._operations.is_subtype(self.typ, qualified_name)

    def is_sequence_of_subtype(self, qualified_name: str) -> bool:
        """Check whether the type is a sequence of the requested type."""
        return self._operations.is_sequence_of_subtype(self.typ, qualified_name)


_KIND_BUILT_IN = 0
_KIND_DECLARED = 1
_KIND_FUNCTION = 2
_KIND_CLASS = 3
_KIND_UNION = 4
_KIND_SYNTHESIZED = 8
_NUMERIC_NAMES = frozenset({"int", "float", "bool"})
_SEQUENCE_NAMES = frozenset({"list", "tuple", "Sequence"})
_STUB_DEF_RE = re.compile(r"(?:class|def)\s+(\w+)")


def _declared_name(typ: dict[str, Any]) -> str | None:
    if typ.get("kind") in (_KIND_CLASS, _KIND_DECLARED, _KIND_FUNCTION):
        name = (typ.get("declaration") or {}).get("name")
        return str(name) if name is not None else None
    if typ.get("kind") == _KIND_SYNTHESIZED:
        stub = typ.get("stubContent", "")
        offset = (typ.get("metadata") or {}).get("primaryDefinitionOffset", 0)
        match = _STUB_DEF_RE.match(stub, offset) or _STUB_DEF_RE.search(stub, offset)
        return match.group(1) if match else None
    return None


def _is_unknown(typ: dict[str, Any]) -> bool:
    return typ.get("kind") == _KIND_BUILT_IN and typ.get("name") in ("any", "unknown")


class TspTypeOperations:
    """Interpret TSP 0.4.1 type values.

    TSP class declarations expose short names, without an MRO or qualified
    name. These predicates therefore cannot establish arbitrary inheritance
    or distinguish identically named classes. Native mypy operations retain
    the full checker semantics instead of inheriting this wire-format limit.
    """

    def is_proven_scalar(self, typ: dict[str, Any]) -> bool:
        """Require positive numeric evidence for every union member."""
        if typ.get("kind") == _KIND_UNION:
            members = typ.get("subTypes", [])
            return bool(members) and all(self.is_proven_scalar(item) for item in members)
        return _declared_name(typ) in _NUMERIC_NAMES

    def is_subtype(self, typ: dict[str, Any], qualified_name: str) -> bool:
        """Compare declared names, preserving unknowns as possible matches."""
        if _is_unknown(typ):
            return True
        if typ.get("kind") == _KIND_UNION:
            members = typ.get("subTypes", [])
            return bool(members) and all(self.is_subtype(item, qualified_name) for item in members)
        return _declared_name(typ) == qualified_name.rsplit(".", 1)[-1]

    def is_sequence_of_subtype(self, typ: dict[str, Any], qualified_name: str) -> bool:
        """Check known sequence type arguments, including unknown elements."""
        if _is_unknown(typ):
            return True
        if typ.get("kind") == _KIND_UNION:
            members = typ.get("subTypes", [])
            return bool(members) and all(
                self.is_sequence_of_subtype(item, qualified_name) for item in members
            )
        if _declared_name(typ) not in _SEQUENCE_NAMES:
            return False
        args = typ.get("typeArgs", [])
        return bool(args) and all(self.is_subtype(arg, qualified_name) for arg in args)
