# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Builders are utility objects for building (compound) CST nodes.

Builders keep metadata, collections of members, and implement CST Node
construction methods. This way, they abstract over the actual CST Node
construction using libCST.
"""

import abc

from collections.abc import Sequence
from typing import Generic, Protocol, TypeVar, runtime_checkable

import libcst
import libcst.display

from typing_extensions import override

_CSTNodeT_co = TypeVar("_CSTNodeT_co", bound=libcst.CSTNode, covariant=True)


class NodeBuilder(abc.ABC, Generic[_CSTNodeT_co]):
    """Abstract Node builder class. All builders inherit from this."""

    @abc.abstractmethod
    def build(self) -> _CSTNodeT_co:
        """Create CSTNode from collected metadata."""


class ModuleBuilder(NodeBuilder[libcst.Module]):
    """Builder for modules.

    Collects ClassBuilders and builds a module based on them.
    """

    def __init__(self) -> None:
        self._statements: list[
            NodeBuilder[libcst.SimpleStatementLine | libcst.BaseCompoundStatement]
        ] = []

    def add_class(self, klass: "ClassBuilder") -> None:
        """Add classbuilder to module."""
        self._statements.append(klass)

    def build(self) -> libcst.Module:
        """Build the module."""
        return libcst.Module(body=self.build_module())

    def build_module(self) -> Sequence[libcst.SimpleStatementLine | libcst.BaseCompoundStatement]:
        """Create Module from collected metadata."""
        return [c.build() for c in self._statements]


class ClassBuilder(NodeBuilder[libcst.ClassDef]):
    """Builder for classes.

    Collects methods and other methods.
    """

    def __init__(self, name: str, bases: Sequence[str]) -> None:
        self._name = name
        self._bases = tuple(bases)
        self._methods: list[_FunctionBuilderP] = []

    def add_method(self, funcbuilder: "_FunctionBuilderP") -> None:
        """Add a method to the class represented by this `Builder`."""
        self._methods.append(funcbuilder)

    @override
    def build(self) -> libcst.ClassDef:
        return self.build_class(bases=self._bases, methods=self._methods)

    def build_class(
        self, bases: Sequence[str], methods: Sequence["_FunctionBuilderP"]
    ) -> libcst.ClassDef:
        """Create ClassDef from collected metadata."""
        bases = ", ".join(bases)
        base_def: libcst.ClassDef = libcst.parse_statement(f"class {self._name}({bases}): pass")  # type: ignore[assignment]

        return base_def.with_changes(body=libcst.IndentedBlock([func.build() for func in methods]))


class QuantizedModuleBuilder(ClassBuilder):
    """Builder for QuantizedModules."""

    def __init__(self, name: str, bases: Sequence[str]) -> None:
        super().__init__(name=name, bases=bases)
        self._quantizers: list[str] = []

    def add_quantizer(self, name: str, add_count_suffix: bool = True) -> str:
        """Add a quantizer to this class."""
        if add_count_suffix:
            name = f"{name}_{len(self._quantizers) + 1}"
        if name in self._quantizers:
            msg = f"Quantizer with name '{name}' was already added"
            raise ValueError(msg)
        self._quantizers.append(name)
        return name

    @override
    def build(self) -> libcst.ClassDef:
        bases = ("fastforward.nn.QuantizedModule",) + self._bases
        methods = [InitQuantizationMethod(self._quantizers)] + self._methods
        return self.build_class(bases=bases, methods=methods)


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
