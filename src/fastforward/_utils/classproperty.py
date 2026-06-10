# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""This module defines the `classproperty` decorator.

A ``@classproperty`` decorated method will behave similarly to chaining
``@classmethod`` and ``@property`` decorators.

Note that chaining of ``@classmethod`` and ``@property`` is deprecated since
python 3.11 and disallowed since python 3.13.

Adapted from the pytorch version of ``classmethod`` decorator.
"""

from collections.abc import Callable
from typing import Any, Generic, TypeVar

_R = TypeVar("_R")


class _ClassPropertyDescriptor(Generic[_R]):
    def __init__(self, fget: "classmethod[Any, ..., _R] | staticmethod[..., _R]") -> None:
        self.fget = fget

    def __get__(self, instance: object, owner: type[Any] | None = None) -> _R:
        if owner is None:
            owner = type(instance)
        return self.fget.__get__(instance, owner)()


def classproperty(
    func: "Callable[..., _R] | classmethod[Any, ..., _R] | staticmethod[..., _R]",
) -> _ClassPropertyDescriptor[_R]:
    """Decorate a method so it behaves like a read-only property bound to the class."""
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)
    return _ClassPropertyDescriptor(func)
