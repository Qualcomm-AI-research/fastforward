# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

"""
Builders are utility objects for building (compound) CST nodes. They keep
metadata, collections of members, and implement CST Node construction methods.
This way, they abstract over the actual CST Node construction using libCST.
"""

import abc

from typing import Generic, Sequence

import libcst
import libcst.display


class NodeBuilder(abc.ABC, Generic[libcst.CSTNodeT]):
    """
    Abstract Node builder class. All builders inherit from this
    """

    @abc.abstractmethod
    def build(self) -> libcst.CSTNodeT: ...


class ClassBuilder(NodeBuilder[libcst.ClassDef]):
    """
    Builder for classes. Collections methods and other
    methods.
    """

    def __init__(self, name: str, bases: Sequence[str]) -> None:
        self._name = name
        self._bases = bases
        self._methods: list[FunctionBuilder] = []

    def add_method(self, funcbuilder: "FunctionBuilder") -> None:
        self._methods.append(funcbuilder)

    def build(self) -> libcst.ClassDef:
        bases = ", ".join(self._bases)
        base_def: libcst.ClassDef = libcst.parse_statement(f"class {self._name}({bases}): pass")  # type: ignore[assignment]

        return base_def.with_changes(
            body=libcst.IndentedBlock([func.build() for func in self._methods])
        )


class FunctionBuilder(NodeBuilder[libcst.FunctionDef]):
    """
    Builder for FunctionDef
    """

    def __init__(self, funcdef: libcst.FunctionDef) -> None:
        self._funcdef = funcdef

    def build(self) -> libcst.FunctionDef:
        # Currently, the FunctionBuilder simply holds a FunctionDef node
        # which is returned.
        return self._funcdef
