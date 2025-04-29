# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import dataclasses
import inspect
import operator as py_operator
import textwrap

from collections.abc import Sequence
from types import ModuleType
from typing import Any, Callable, TypeAlias

import libcst
import libcst.helpers
import libcst.metadata

from typing_extensions import override

from fastforward._autoquant.cst.validation import ensure_type
from fastforward._autoquant.pysource.scope import infer_scopes
from fastforward._import import QualifiedNameReference, fully_qualified_name


class SourceContextError(RuntimeError):
    """General `SourceContext` Exception."""


class SourceContextMemberError(AttributeError):
    """Exception for missing member in `SourceContext`."""


_PassesT: TypeAlias = Sequence[libcst.CSTTransformer]


class SourceContext:
    """Context to obtain references to source objects.

    `PySource` objects can be obtained with the `get` method. `SourceContext`
    will load and parse modules only once and takes CSTs from the module CST
    for members of a module. This ensures that passes that act globally on a
    module are applied once.

    Args:
        preprocessing_passes: Sequence of CST passes that are applied on every
            module CST that is loaded.
    """

    def __init__(
        self,
        *,
        preprocessing_passes: _PassesT = (),
    ) -> None:
        self._preprocessing_passes = preprocessing_passes
        self._modules: dict[ModuleType, _ModuleSource] = {}

    def get(self, qualified_name: str) -> "PySource":
        """Obtain `PySource` object for object referenced by `qualified_name`.

        This method will resolve references, e.g., `fastforward.QuantizedTensor`
        being defined at `fastforward.quantized_tensor.QuantizedTensor`.

        Args:
            qualified_name: A reference to a python object for which to obtain
                a `PySource` object.
        """
        resolved_name = _resolve_name(qualified_name)
        module_src = self._get_module_source(resolved_name.module)
        return module_src.member(resolved_name.obj_name)

    def get_cst(
        self,
        qualified_name: str,
        NodeType: type[libcst.CSTNodeT] = libcst.CSTNode,  # type: ignore[assignment]
    ) -> libcst.CSTNodeT:
        """Obtain CST for object referenced by `qualified_name`.

        This method will resolve references, e.g., `fastforward.QuantizedTensor`
        being defined at `fastforward.quantized_tensor.QuantizedTensor`.

        Args:
            qualified_name: A reference to a python object for which to obtain
                a `PySource` object.
            NodeType: Node type that is expected. If the resulting node does not
                match `NodeType`, a `TypeError` is raised.
        """
        resolved_name = _resolve_name(qualified_name)
        module_src = self._get_module_source(resolved_name.module)
        node = module_src.member_cst(resolved_name.obj_name)
        return ensure_type(node, NodeType)

    def get_scope(self, qualified_name: str) -> libcst.metadata.Scope | None:
        """Obtain scope for object referend by `qualified_name`.

        This method will resolve references, e.g., `fastforward.QuantizedTensor`
        being defined at `fastforward.quantized_tensor.QuantizedTensor`.

        Args:
            qualified_name: A reference to a python object for which to obtain
                a `Scope` object.
        """
        resolved_name = _resolve_name(qualified_name)
        module_src = self._get_module_source(resolved_name.module)
        return module_src.member_scope(resolved_name.obj_name)

    def _get_module_source(self, module: ModuleType) -> "_ModuleSource":
        if module not in self._modules:
            self._modules[module] = _ModuleSource(
                module, source_context=self, preprocessing_passes=self._preprocessing_passes
            )
        return self._modules[module]


class PySource:
    """Represents a reference to the Python source code for a specific object.

    The `PySource` class provides methods to obtain the Concrete Syntax Tree
    (CST) for the referenced object and its attributes, allowing for inspection
    and manipulation of the source code.

    Args:
        source_context: The context through which this `PySource` object was obtained.
        qualified_name: The qualified name of the object this `PySource` object represents.
    """

    def __init__(self, source_context: SourceContext, qualified_name: str) -> None:
        self._source_context = source_context
        self._qualified_name = qualified_name

    def cst(self, *, NodeType: type[libcst.CSTNodeT] = libcst.CSTNode) -> libcst.CSTNodeT:  # type: ignore[assignment]
        """Obtain CST for object represented by `self`.

        Args:
            NodeType: Node type that is expected. If the resulting node does not
                match `NodeType`, a `TypeError` is raised.
        """
        return ensure_type(self._source_context.get_cst(self._qualified_name), NodeType)

    def scope(self) -> libcst.metadata.Scope | None:
        """Obtain scope for object represented by `self`."""
        return self._source_context.get_scope(self._qualified_name)

    def member(self, name: str) -> "PySource":
        """Obtain a `PySource` object for an attribute of the python object represented by `self`.

        Args:
            name: The name of the attribute.
        """
        qualified_name = _join_qualified_names(self._qualified_name, name)
        return self._source_context.get(qualified_name)

    def is_class(self) -> bool:
        """True if this object represents a class object, False otherwise."""
        return isinstance(self.cst(), libcst.ClassDef)

    def is_function(self) -> bool:
        """True if this object represents a function object, False otherwise."""
        return isinstance(self.cst(), libcst.FunctionDef)

    def is_module(self) -> bool:
        """True if this object represents a module object, False otherwise."""
        return isinstance(self.cst(), libcst.Module)

    def module(self) -> "PySource":
        """Obtain a `PySource` object for the module of the object references by `self`.

        If `self` already references a module, returns `self`.
        """
        if self.is_module():
            return self
        return self._source_context.get(self._qualified_name.rsplit(".", 1)[0]).module()

    @property
    def qualified_name(self) -> str:
        """Qualified name of the object referenced by `self`."""
        return self._qualified_name

    @override
    def __repr__(self) -> str:
        return f"<{type(self).__name__}: {self._qualified_name}>"


class _ModuleSource:
    def __init__(
        self,
        module: ModuleType,
        *,
        source_context: SourceContext,
        preprocessing_passes: _PassesT = (),
    ) -> None:
        self._source_context = source_context
        self._preprocessing_passes = tuple(preprocessing_passes)
        self._py_module = module
        self._scopes: dict[libcst.CSTNode, libcst.metadata.Scope] = {}

        module_cst = self._read_module_cst(module)
        self._members: dict[str, libcst.CSTNode] = {"": module_cst}

        def _add_symbol(parts: tuple[str, ...], node: libcst.CSTNode) -> None:
            relative_name = ".".join(parts)
            self._members[relative_name] = node

        module_cst = module_cst.visit(_SymbolVisitor(_add_symbol))
        self._scope_analysis(module_cst)

    def _scope_analysis(self, cst: libcst.Module) -> None:
        self._scopes = infer_scopes(cst)

    def member_cst(self, relative_qualified_name: str) -> libcst.CSTNode:
        if relative_qualified_name not in self._members:
            raise SourceContextMemberError(
                f"'{relative_qualified_name}' is not recorded as a member of {self._py_module}. "
                + "Currently only function and class definitions are recorded as members."
            )
        return self._members[relative_qualified_name]

    def member_scope(self, relative_qualified_name: str) -> libcst.metadata.Scope | None:
        cst = self.member_cst(relative_qualified_name)
        return self._scopes.get(cst)

    def member(self, relative_qualified_name: str) -> "PySource":
        if relative_qualified_name not in self._members:
            raise SourceContextMemberError(
                f"'{relative_qualified_name}' is not recorded as a member of {self._py_module}. "
                + "Currently only function and class definitions are recorded as members."
            )

        qualified_name = _join_qualified_names(self.qualified_name(), relative_qualified_name)
        return PySource(self._source_context, qualified_name)

    def _read_module_cst(self, module: ModuleType) -> libcst.Module:
        src = inspect.getsource(module)
        module_cst = libcst.parse_module(textwrap.dedent(src))
        for cst_pass in self._preprocessing_passes:
            module_cst = module_cst.visit(cst_pass)
        return module_cst

    def qualified_name(self) -> str:
        return fully_qualified_name(self._py_module)


class _SymbolVisitor(libcst.CSTVisitor):
    def __init__(self, symbol_handler: Callable[[tuple[str, ...], libcst.CSTNode], Any]):
        super().__init__()
        self._handler = symbol_handler
        self._context: list[str] = []

    def _enter(self, name: str) -> None:
        self._context.append(name)

    def _leave(self) -> None:
        _ = self._context.pop()

    def _record_member(self, name: str, node: libcst.CSTNode) -> None:
        self._handler(tuple(self._context + [name]), node)

    @override
    def visit_ClassDef(self, node: libcst.ClassDef) -> None:
        name = _get_full_name_or_fail(node)
        self._record_member(name, node)
        self._enter(name)

    @override
    def leave_ClassDef(self, original_node: libcst.ClassDef) -> None:
        self._leave()

    @override
    def visit_FunctionDef(self, node: libcst.FunctionDef) -> bool:
        name = _get_full_name_or_fail(node)
        self._record_member(name, node)
        return False


def _join_qualified_names(*names: str) -> str:
    while names and names[0] == "":
        names = names[1:]
    while names and names[-1] == "":
        names = names[:-1]
    if "" in names:
        raise ValueError("Cannot join empty qualified name")
    return ".".join(names)


@dataclasses.dataclass
class _ResolvedQualifiedName:
    qualified_name: str
    module: ModuleType
    module_name: str
    obj_name: str


def _resolve_name(qualified_name: str) -> _ResolvedQualifiedName:
    ref = QualifiedNameReference(qualified_name)
    module, module_name = ref.import_module()
    obj_name = qualified_name.removeprefix(module_name).removeprefix(".")

    if module_name != qualified_name:
        py_object = py_operator.attrgetter(obj_name)(module)
        resolved_qualified_name = fully_qualified_name(py_object)
    else:
        resolved_qualified_name = module_name

    if resolved_qualified_name != qualified_name:
        return _resolve_name(resolved_qualified_name)

    return _ResolvedQualifiedName(resolved_qualified_name, module, module_name, obj_name)


def _get_full_name_or_fail(node: libcst.CSTNode) -> str:
    name = libcst.helpers.get_full_name_for_node(node)
    if name is None:
        raise SourceContextError(
            f"Expected {type(node).__name__} to have a name, but name resolution failed"
        )
    return name
