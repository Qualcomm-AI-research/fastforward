# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Builders are utility objects for building (compound) CST nodes.

Builders keep metadata, collections of members, and implement CST Node
construction methods. This way, they abstract over the actual CST Node
construction using libCST.
"""

import abc

from collections.abc import Sequence
from typing import Any, Generic, Iterable, Iterator, Protocol, TypeVar, cast, runtime_checkable

import libcst
import libcst.helpers
import libcst.metadata

from typing_extensions import override

from fastforward._autoquant.cst import node_creation, nodes
from fastforward._autoquant.function_context import FunctionContext
from fastforward._autoquant.pybuilder import QuantizerReferenceCollection
from fastforward._autoquant.pysource.scope import ImportSymbol

_CSTNodeT_co = TypeVar("_CSTNodeT_co", bound=libcst.CSTNode, covariant=True)


class AbstractNodeBuilder(abc.ABC, Generic[_CSTNodeT_co]):
    """Abstract Node builder class. All builders inherit from this."""

    @abc.abstractmethod
    def build(self, quantizer_refs: QuantizerReferenceCollection) -> _CSTNodeT_co:
        """Create CSTNode from collected metadata."""


class NodeBuilder(AbstractNodeBuilder[_CSTNodeT_co]):
    def __init__(self, origin: Any | None) -> None:
        super().__init__()
        self._origin = origin

    @property
    def origin(self) -> Any:
        return self._origin


class ModuleBuilder(NodeBuilder[libcst.Module]):
    """Builder for modules.

    Collects ClassBuilders and builds a module based on them.
    """

    def __init__(self, origin: Any | None) -> None:
        super().__init__(origin=origin)
        self._statements: list[
            NodeBuilder[libcst.SimpleStatementLine | libcst.BaseCompoundStatement]
        ] = []

    def add_class(self, klass: "ClassBuilder") -> None:
        """Add classbuilder to module."""
        self._statements.append(klass)

    def build(self, quantizer_refs: QuantizerReferenceCollection) -> libcst.Module:
        """Build the module."""
        return libcst.Module(body=self.build_module(quantizer_refs))

    def import_statements(self) -> Iterator[libcst.SimpleStatementLine]:
        """Combines import statements for all statements."""
        required_imports: set[ImportSymbol] = set()
        for statement in self._statements:
            if imports := getattr(statement, "required_imports", None):
                required_imports |= set(imports)

        yield from (import_symbol.as_node() for import_symbol in required_imports)

    def build_module(
        self, quantizer_refs: QuantizerReferenceCollection
    ) -> Sequence[libcst.SimpleStatementLine | libcst.BaseCompoundStatement]:
        """Create Module from collected metadata."""
        return list(self.import_statements()) + [c.build(quantizer_refs) for c in self._statements]

    def classes(self) -> Iterator["ClassBuilder"]:
        yield from (stmt for stmt in self._statements if isinstance(stmt, ClassBuilder))

    def functions(self) -> Iterator["FunctionBuilder"]:
        yield from (stmt for stmt in self._statements if isinstance(stmt, FunctionBuilder))
        for clsbuilder in self.classes():
            yield from clsbuilder.methods()

    def quantized_functions(self) -> Iterator["QuantizedFunctionBuilder"]:
        for stmt in self._statements:
            if isinstance(stmt, QuantizedFunctionBuilder):
                yield stmt
        for clsbuilder in self.classes():
            for method in clsbuilder.methods():
                if isinstance(method, QuantizedFunctionBuilder):
                    yield method


class ClassBuilder(NodeBuilder[libcst.ClassDef]):
    """Builder for classes.

    Collects methods and other methods.

    Args:
        name: Name of the class.
        bases: Name of base classes.
        required_imports: Sequence of `ImportSymbol`s required for class.
    """

    def __init__(
        self,
        name: str,
        bases: Sequence[str],
        required_imports: Sequence[ImportSymbol],
        origin: Any | None,
    ) -> None:
        super().__init__(origin=origin)
        self._name = name
        self._bases = tuple(bases)
        self._methods: list[FunctionBuilder] = []
        self._required_imports = tuple(required_imports)

    def add_method(self, funcbuilder: "FunctionBuilder") -> None:
        """Add a method to the class represented by this `Builder`."""
        self._methods.append(funcbuilder)

    def has_method(self, method_name: str) -> bool:
        return any(meth.name == method_name for meth in self._methods)

    @property
    def required_imports(self) -> tuple[ImportSymbol, ...]:
        """Imports that are required for this class."""
        required_imports: set[ImportSymbol] = set(self._required_imports)
        for method in self._methods:
            required_imports |= set(method.required_imports)
        return tuple(required_imports)

    @override
    def build(self, quantizer_refs: QuantizerReferenceCollection) -> libcst.ClassDef:
        return self.build_class(
            bases=self._bases, methods=self._methods, quantizer_refs=quantizer_refs
        )

    def build_class(
        self,
        bases: Sequence[str],
        methods: Sequence["_FunctionBuilderP"],
        quantizer_refs: QuantizerReferenceCollection,
    ) -> libcst.ClassDef:
        """Create ClassDef from collected metadata."""
        bases = ", ".join(bases)
        base_def: libcst.ClassDef = libcst.parse_statement(f"class {self._name}({bases}): pass")  # type: ignore[assignment]

        class_def = base_def.with_changes(
            body=libcst.IndentedBlock([func.build(quantizer_refs) for func in methods])
        )
        return cast(
            libcst.ClassDef,
            class_def.visit(_ResolveAbstractClassReferences(libcst.Name(self._name))),
        )

    def methods(self) -> Iterator["FunctionBuilder"]:
        yield from self._methods


class QuantizedModuleBuilder(ClassBuilder):
    """Builder for QuantizedModules.

    Args:
        name: Name of the class.
        bases: Name of base classes.
        required_imports: Sequence of `ImportSymbol`s required for class.
    """

    _origin: type

    def __init__(
        self,
        name: str,
        bases: Sequence[str],
        required_imports: Sequence[ImportSymbol],
        origin: type,
    ) -> None:
        required_imports_set = set(required_imports)
        required_imports_set.add(ImportSymbol("fastforward"))
        super().__init__(
            name=name, bases=bases, required_imports=tuple(required_imports_set), origin=origin
        )

    @property
    def origin(self) -> type:
        return self._origin

    @override
    def build(self, quantizer_refs: QuantizerReferenceCollection) -> libcst.ClassDef:
        bases = ("fastforward.nn.QuantizedModule",) + self._bases
        init_quant_method = InitQuantizationMethod(counterpart_type=self.origin)
        methods = [init_quant_method] + self._methods
        module_tree = self.build_class(bases=bases, methods=methods, quantizer_refs=quantizer_refs)
        return module_tree


@runtime_checkable
class _FunctionBuilderP(Protocol):
    @property
    def required_imports(self) -> tuple[ImportSymbol, ...]: ...

    def build(self, quantizer_refs: QuantizerReferenceCollection) -> libcst.FunctionDef: ...

    @property
    def name(self) -> str: ...


class FunctionBuilder(NodeBuilder[libcst.FunctionDef]):
    """Builder for FunctionDef.

    Args:
        funcdef: CST for function.
        required_imports: Sequence of `ImportSymbol`s required for function.
    """

    def __init__(
        self,
        funcdef: libcst.FunctionDef,
        required_imports: Iterable[ImportSymbol],
        origin: Any | None,
    ) -> None:
        super().__init__(origin=origin)
        self._funcdef = funcdef
        self._required_imports = tuple(required_imports)

    @property
    def cst(self) -> libcst.FunctionDef:
        return self._funcdef

    @cst.setter
    def cst(self, funcdef: libcst.FunctionDef) -> None:
        self._funcdef = funcdef

    @property
    def required_imports(self) -> tuple[ImportSymbol, ...]:
        """Imports that are required for this function."""
        return self._required_imports

    @override
    def build(self, quantizer_refs: QuantizerReferenceCollection) -> libcst.FunctionDef:
        # Currently, the FunctionBuilder simply holds a FunctionDef node
        # which is returned.
        return self._funcdef

    @property
    def name(self) -> str:
        return self._funcdef.name.value


class InitQuantizationMethod(NodeBuilder[libcst.FunctionDef]):
    """Builder for `__init_quantization__` method."""

    def __init__(self, counterpart_type: type):
        """Imports that are required for this function."""
        super().__init__(origin=None)
        self._required_imports = (ImportSymbol("fastforward"),)
        self._counterpart_type = counterpart_type

    @property
    def required_imports(self) -> tuple[ImportSymbol, ...]:
        """Imports that are required for this function."""
        return self._required_imports

    @property
    def name(self) -> str:
        return "__init_quantization__"

    @override
    def build(self, quantizer_refs: QuantizerReferenceCollection) -> libcst.FunctionDef:
        body_statements = [libcst.parse_statement(f"super().{self.name}()")]

        body_statements += [
            libcst.helpers.parse_template_statement(
                "self.{name}: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()",
                name=ref,
            )
            for ref in quantizer_refs.instance_quantizers_for_module(self._counterpart_type)
        ]

        init_quant_method_node = libcst.helpers.parse_template_statement(
            "def {name}(self) -> None:{body}",
            name=libcst.Name(self.name),
            body=libcst.IndentedBlock(tuple(body_statements)),
        )

        assert isinstance(init_quant_method_node, libcst.FunctionDef)
        init_quant_method_node = cast(
            libcst.FunctionDef,
            init_quant_method_node.visit(_DisambiguateQuantizerNameTransformer(quantizer_refs)),
        )
        return init_quant_method_node


class QuantizedFunctionBuilder(FunctionBuilder):
    """Builder for quantized methods.

    Quantized methods are methods that use (or introduce) quanitzers that are
    defined on the instance.

    Args:
        funcdef: CST for function.
        required_imports: Sequence of `ImportSymbol`s required for function.
    """

    _quantizer_owned_references: list[nodes.QuantizerReference] | None
    _quantizer_parameter_references: list[nodes.QuantizerReference] | None
    _origin: FunctionContext

    def __init__(
        self,
        funcdef: libcst.FunctionDef,
        required_imports: Iterable[ImportSymbol],
        origin: FunctionContext | None,
    ) -> None:
        super().__init__(funcdef=funcdef, required_imports=required_imports, origin=origin)
        self.quantizer_signature: tuple[nodes.QuantizerReference, ...] = ()

    @property
    def origin(self) -> FunctionContext:
        return self._origin

    @override
    def build(self, quantizer_refs: QuantizerReferenceCollection) -> libcst.FunctionDef:
        funcdef = _add_quantizer_args(self._funcdef, self.quantizer_signature)
        funcdef = cast(
            libcst.FunctionDef, funcdef.visit(_DisambiguateQuantizerNameTransformer(quantizer_refs))
        )
        assert isinstance(funcdef, libcst.FunctionDef)
        return funcdef


def _add_quantizer_args(
    funcdef: libcst.FunctionDef, args: Sequence[nodes.QuantizerReference]
) -> libcst.FunctionDef:
    quant_params = [
        node_creation.get_parameter_node(arg, "fastforward.nn.Quantizer") for arg in args
    ]
    kwonly_params = tuple(funcdef.params.kwonly_params) + tuple(quant_params)
    func_params = funcdef.params.with_changes(kwonly_params=kwonly_params)
    return funcdef.with_changes(params=func_params)


class _DisambiguateQuantizerNameTransformer(libcst.CSTTransformer):
    def __init__(self, quantizer_refs: QuantizerReferenceCollection) -> None:
        self._quantizer_refs = quantizer_refs

    def leave_QuantizerReference(
        self, _original_node: nodes.QuantizerReference, updated_node: nodes.QuantizerReference
    ) -> libcst.Name:
        try:
            disambiguated_name = self._quantizer_refs.disambiguate_reference(updated_node)
        except KeyError:
            disambiguated_name = updated_node.value

        return libcst.Name(
            value=disambiguated_name,
            lpar=updated_node.lpar,
            rpar=updated_node.rpar,
        )


class _ResolveAbstractClassReferences(libcst.CSTTransformer):
    def __init__(self, class_name: libcst.Name):
        self._class_name = class_name

    def leave_AbstractClassReference(
        self,
        original_node: nodes.AbstractClassReference,
        updated_node: nodes.AbstractClassReference,
    ) -> libcst.Name:
        del original_node, updated_node
        return self._class_name.deep_clone()
