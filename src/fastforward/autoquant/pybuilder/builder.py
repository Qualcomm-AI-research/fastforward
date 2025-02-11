# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

"""Builders are utility objects for building (compound) CST nodes.

Builders keep metadata, collections of members, and implement CST Node
construction methods. This way, they abstract over the actual CST Node
construction using libCST.
"""

import abc

from collections.abc import Sequence
from typing import Generic, Protocol, runtime_checkable

import libcst
import libcst.display

from typing_extensions import override


class NodeBuilder(abc.ABC, Generic[libcst.CSTNodeT]):
    """Abstract Node builder class. All builders inherit from this."""

    @abc.abstractmethod
    def build(self) -> libcst.CSTNodeT:
        """Create CSTNode from collected metadata."""


class ClassBuilder(NodeBuilder[libcst.ClassDef]):
    """Builder for classes.

    Collects methods and other methods.
    """

    def __init__(self, name: str, bases: Sequence[str]) -> None:
        self._name = name
        self._bases = bases
        self._methods: list[_FunctionBuilderP] = []

    def add_method(self, funcbuilder: "_FunctionBuilderP") -> None:
        """Add a method to the class represented by this `Builder`."""
        self._methods.append(funcbuilder)

    @override
    def build(self) -> libcst.ClassDef:
        bases = ", ".join(self._bases)
        base_def: libcst.ClassDef = libcst.parse_statement(f"class {self._name}({bases}): pass")  # type: ignore[assignment]

        return base_def.with_changes(
            body=libcst.IndentedBlock([func.build() for func in self._methods])
        )


@runtime_checkable
class _FunctionBuilderP(Protocol):
    def build(self) -> libcst.FunctionDef: ...


class FunctionBuilder(NodeBuilder[libcst.FunctionDef]):
    """Builder for FunctionDef."""

    def __init__(self, funcdef: libcst.FunctionDef) -> None:
        self._funcdef = funcdef

    @override
    def build(self) -> libcst.FunctionDef:
        # Currently, the FunctionBuilder simply holds a FunctionDef node
        # which is returned.
        return self._funcdef


class InitQuantizationMethod(_FunctionBuilderP):
    """Builder for `__init_quantization__` method."""

    def __init__(self, quantizer_collection: Sequence[str]):
        self.quantizer_collection = quantizer_collection

    @override
    def build(self) -> libcst.FunctionDef:
        # Parse minimalist method into a subtree, for simplicity of creation
        magic_method = libcst.parse_statement(
            "def __init_quantization__(self) -> None:\n    super().__init_quantization__()"
        )
        assert isinstance(magic_method, libcst.FunctionDef)
        init_quantizer_vars = []

        for name in self.quantizer_collection:
            init_quantizer_vars.append(
                libcst.parse_statement(f"self.{name} = fastforward.nn.QuantizerStub()")
            )
        magic_method = magic_method.with_changes(
            body=magic_method.body.with_changes(
                body=(*magic_method.body.body, *init_quantizer_vars)
            )
        )
        return magic_method
