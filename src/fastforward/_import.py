# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import importlib

from types import ModuleType
from typing import Any


def fully_qualified_name(obj: Any) -> str:
    if isinstance(obj, ModuleType):
        return obj.__name__
    module: str | None = obj.__module__
    name: str = obj.__qualname__
    if module is None or module == "__builtin__":
        return name
    return f"{module}.{name}"


class QualifiedNameReference:
    def __init__(self, qualified_name: str) -> None:
        self._parts = tuple(qualified_name.split("."))

    @property
    def parent(self) -> "QualifiedNameReference":
        if len(self._parts) == 0:
            raise ValueError(f"Empty {type(self).__name__} has no parent")
        new_reference = QualifiedNameReference.__new__(QualifiedNameReference)
        new_reference._parts = self._parts[:-1]
        return new_reference

    @property
    def stem(self) -> str:
        if len(self._parts) == 0:
            raise ValueError(f"Empty {type(self).__name__} has no stem")
        return self._parts[-1]

    @property
    def qualified_name(self) -> str:
        return ".".join(self._parts)

    # Use import_ since import is a (non-soft) keyword.
    def import_(self) -> Any:
        # Import the module in which self.qualified_name is defined.
        # Then traverse the members/attributes of that module to access
        # the object at `self.qualified_name`
        module, _, members = self._import_module()
        finger = module
        for member in members:
            try:
                finger = getattr(finger, member)
            except AttributeError:
                # On a attribute lookup failure, we can conclude that the
                # provided reference to not refer to an existing module/object.
                raise ImportError(f"Cannot import '{self.qualified_name}'")
        return finger

    def _import_module(
        self, *, _remainder: tuple[str, ...] = ()
    ) -> tuple[ModuleType, str, tuple[str, ...]]:
        # If we reach a root reference, the import must have failed
        if len(self._parts) == 0:
            raise ImportError(f"Cannot import '{'.'.join(_remainder)}'")

        # Try to import the full reference as a module. This may fail
        # if the reference does not refer to a python module. In thas case
        # 'non-module' attribute in `_remainder`
        # try to import the 'parent' of qualified_name and collect the
        try:
            return importlib.import_module(self.qualified_name), self.qualified_name, _remainder
        except ModuleNotFoundError:
            return self.parent._import_module(_remainder=(self.stem,) + _remainder)

    def import_module(self) -> tuple[ModuleType, str]:
        module, qualified_module_name, _ = self._import_module()
        return module, qualified_module_name
