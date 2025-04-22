# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Builders are utility objects for building (compound) CST nodes.

Builders keep metadata, collections of members, and implement CST Node
construction methods. This way, they abstract over the actual CST Node
construction using libCST.
"""

import abc

from collections.abc import Mapping, Sequence
from typing import Generic, Iterator, Protocol, TypeAlias, TypeVar, runtime_checkable

import libcst
import libcst.helpers

from typing_extensions import override

from fastforward._autoquant.cst import nodes
from fastforward._autoquant.cst.filter import filter_nodes_by_type

_CSTNodeT_co = TypeVar("_CSTNodeT_co", bound=libcst.CSTNode, covariant=True)

QuantizerInfo: TypeAlias = nodes.QuantizerReference.QuantizerInfo


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

    def quantizer_info(self) -> Iterator[QuantizerInfo]:
        """Iterator over all unique `QuantizerInfo` objects referenced by quantizer references."""
        seen_info: set[QuantizerInfo] = set()
        for method in self._methods:
            if not isinstance(method, QuantizedFunctionBuilder):
                continue
            for quantizer_ref in method.quantizer_references:
                quantizer_info = quantizer_ref.quantizer_info
                if quantizer_info in seen_info:
                    continue
                seen_info.add(quantizer_info)
                yield quantizer_info

    @override
    def build(self) -> libcst.ClassDef:
        bases = ("fastforward.nn.QuantizedModule",) + self._bases
        quantizers = list(self.quantizer_info())
        methods = [InitQuantizationMethod(quantizers)] + self._methods
        module_tree = self.build_class(bases=bases, methods=methods)
        return _disambiguate_quantizers(module_tree, quantizers)


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

    def __init__(self, quantizer_info: Sequence[QuantizerInfo]):
        self.quantizers = quantizer_info

    @override
    def build(self) -> libcst.FunctionDef:
        body_statements = [libcst.parse_statement("super().__init_quantization__()")]

        for quantizer_info in self.quantizers:
            quantizer_name = nodes.QuantizerReference.from_quantizer_info(quantizer_info)
            body_statements.append(
                libcst.helpers.parse_template_statement(
                    "self.{name} = fastforward.nn.QuantizerStub()", name=quantizer_name
                )
            )

        init_quant_method_node = libcst.helpers.parse_template_statement(
            "def __init_quantization__(self) -> None:{body}",
            body=libcst.IndentedBlock(body_statements),
        )
        assert isinstance(init_quant_method_node, libcst.FunctionDef)
        return init_quant_method_node


class QuantizedFunctionBuilder(FunctionBuilder):
    """Builder for quantized methods.

    Quantized methods are methods that use (or introduce) quanitzers that are
    defined on the instance.
    """

    def __init__(self, funcdef: libcst.FunctionDef) -> None:
        super().__init__(funcdef)
        self._quantizer_references: list[nodes.QuantizerReference] = list(
            filter_nodes_by_type(funcdef, nodes.QuantizerReference)
        )

    @property
    def quantizer_references(self) -> list[nodes.QuantizerReference]:
        """All quantizer references used in this method."""
        return self._quantizer_references[:]


def _disambiguate_quantizers(
    tree: libcst.CSTNodeT,
    quantizer_info: Sequence[QuantizerInfo],
) -> libcst.CSTNodeT:
    quantizer_info_by_name: dict[str, list[QuantizerInfo]] = {}
    for qinfo in quantizer_info:
        quantizer_info_by_name.setdefault(qinfo.base_name, []).append(qinfo)

    quantizer_name_map: dict[QuantizerInfo, str] = {}
    prefix = "quantizer_"
    for name, quantizers in quantizer_info_by_name.items():
        if len(quantizers) == 1:
            quantizer_name_map[quantizers[0]] = f"{prefix}{name}"
        else:
            for i, quantizer in enumerate(quantizers):
                quantizer_name_map[quantizer] = f"{prefix}{name}_{i + 1}"

    updated_tree = tree.visit(_DisambiguateQuantizerNameTransformer(quantizer_name_map))
    assert isinstance(updated_tree, type(tree))
    return updated_tree


class _DisambiguateQuantizerNameTransformer(libcst.CSTTransformer):
    def __init__(self, name_map: Mapping[QuantizerInfo, str]) -> None:
        self._name_map = name_map

    def leave_QuantizerReference(
        self, _original_node: nodes.QuantizerReference, updated_node: nodes.QuantizerReference
    ) -> libcst.Name:
        return libcst.Name(
            value=self._name_map[updated_node.quantizer_info],
            lpar=updated_node.lpar,
            rpar=updated_node.rpar,
        )
