# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import enum
import types

from typing import TypeAlias, TypeVar

import torch

T = TypeVar("T")

ScalarOrTuple: TypeAlias = T | tuple[T, ...]
ScalarOr2Tuple: TypeAlias = T | tuple[T, T]


SizeT = torch.Size | tuple[int, ...]

IntOrTuple: TypeAlias = ScalarOrTuple[int]
IntOr2Tuple: TypeAlias = ScalarOr2Tuple[int]

Tuple3: TypeAlias = tuple[T, T, T]


class MethodType(enum.Enum):
    """Enumeration representing different types of methods in a class.

    Attributes:
        METHOD: Regular instance method that takes self as first parameter.
        CLASS_METHOD: Class method decorated with @classmethod that takes cls as first parameter.
        STATIC_METHOD: Static method decorated with @staticmethod that doesn't take self or cls.
        NO_METHOD: Indicates that the class does not have a method with the
            specified name. Can also be used to indicate that a function reference
            is not a method.
    """

    METHOD = enum.auto()
    CLASS_METHOD = enum.auto()
    STATIC_METHOD = enum.auto()
    NO_METHOD = enum.auto()


def method_type(cls_or_module: type | types.ModuleType, method_name: str, /) -> MethodType:
    """Determine the type of a method in a class.

    Args:
        cls_or_module: The class or module to inspect.
        method_name: The name of the method to check.

    Returns:
        A MethodType enum value indicating the type of the method:
    """
    if not isinstance(cls_or_module, (type, types.ModuleType)):
        msg = "'cls_or_module' must be a module or class"  # type: ignore[unreachable]
        raise ValueError(msg)
    match cls_or_module, cls_or_module.__dict__.get(method_name, None):
        case type(), classmethod():
            return MethodType.CLASS_METHOD
        case type(), staticmethod():
            return MethodType.STATIC_METHOD
        case type(), types.FunctionType():
            return MethodType.METHOD
        case _:
            return MethodType.NO_METHOD
