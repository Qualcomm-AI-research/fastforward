# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
#
import inspect
import textwrap

from types import ModuleType
from typing import Any, Callable, Sequence, Type, TypeAlias, TypeVar

import libcst
import libcst.helpers

from fastforward._import import QualifiedNameReference, fully_qualified_name


class SourceContextError(RuntimeError):
    pass


class SourceContextMemberError(AttributeError):
    pass


PassesT: TypeAlias = Sequence[libcst.CSTTransformer]
_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)


def _ensure_type(obj: object, T: Type[_T]) -> _T:
    if not isinstance(obj, T):
        raise TypeError(f"Expected a {T.__name__} but got a {obj.__class__.__name__}!")
    return obj


def _join_qualified_names(*names: str) -> str:
    while names[0] == "":
        names = names[1:]
    if "" in names:
        raise ValueError("Cannot join empty qualified name")
    return ".".join(names)


def get_full_name_or_fail(node: libcst.CSTNode) -> str:
    name = libcst.helpers.get_full_name_for_node(node)
    if name is None:
        raise SourceContextError(
            f"Expected {type(node).__name__} to have a name, but name resolution failed"
        )
    return name


class SourceContext:
    def __init__(
        self,
        *,
        preprocessing_passes: PassesT = (),
    ) -> None:
        self._preprocessing_passes = preprocessing_passes
        self._modules: dict[ModuleType, _ModuleSource] = {}

    def get(self, qualified_name: str) -> "PySource":
        ref = QualifiedNameReference(qualified_name)
        module, module_name = ref.import_module()
        obj_name = qualified_name.removeprefix(module_name).removeprefix(".")
        if module not in self._modules:
            self._modules[module] = _ModuleSource(
                module, preprocessing_passes=self._preprocessing_passes
            )
        return self._modules[module].member(obj_name)


class _ModuleSource:
    def __init__(
        self,
        module: ModuleType,
        *,
        preprocessing_passes: PassesT = (),
    ) -> None:
        self._preprocessing_passes = tuple(preprocessing_passes)
        self._py_module = module

        module_cst = self._read_module_cst(module)
        self._members: dict[str, libcst.CSTNode] = {"": module_cst}

        def _add_symbol(parts: tuple[str, ...], node: libcst.CSTNode) -> None:
            relative_name = ".".join(parts)
            self._members[relative_name] = node

        module_cst.visit(_SymbolVisitor(_add_symbol))

    def _member_cst(self, relative_qualified_name: str) -> libcst.CSTNode:
        if relative_qualified_name not in self._members:
            raise SourceContextMemberError(
                f"'{relative_qualified_name}' is not recorded as a member of {self._py_module}. "
                "Currently only function and class defintions are recorded as members."
            )
        return self._members[relative_qualified_name]

    def member(self, relative_qualified_name: str) -> "PySource":
        if relative_qualified_name not in self._members:
            raise SourceContextMemberError(
                f"'{relative_qualified_name}' is not recorded as a member of {self._py_module}. "
                "Currently only function and class defintions are recorded as members."
            )
        return PySource(self, relative_qualified_name)

    def _read_module_cst(self, module: ModuleType) -> libcst.Module:
        src = inspect.getsource(module)
        module_cst = libcst.parse_module(textwrap.dedent(src))
        for cst_pass in self._preprocessing_passes:
            module_cst = module_cst.visit(cst_pass)
        return module_cst

    def qualified_name(self) -> str:
        return fully_qualified_name(self._py_module)


class PySource:
    def __init__(self, module: _ModuleSource, relative_name: str) -> None:
        self._module = module
        self._relative_name = relative_name

    def cst(self, *, NodeType: type[libcst.CSTNodeT] = libcst.CSTNode) -> libcst.CSTNodeT:  # type: ignore[assignment]
        return _ensure_type(self._module._member_cst(self._relative_name), NodeType)

    def member(self, name: str) -> "PySource":
        relative_name = _join_qualified_names(self._relative_name, name)
        return self._module.member(relative_qualified_name=relative_name)

    def is_class(self) -> bool:
        return isinstance(self.cst(), libcst.ClassDef)

    def is_function(self) -> bool:
        return isinstance(self.cst(), libcst.FunctionDef)

    def is_module(self) -> bool:
        return isinstance(self.cst(), libcst.Module)

    def module(self) -> "PySource":
        return self._module.member("")

    def qualified_name(self) -> str:
        return _join_qualified_names(self._module.qualified_name(), self._relative_name)

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: {self.qualified_name()}>"


class _SymbolVisitor(libcst.CSTVisitor):
    def __init__(self, symbol_handler: Callable[[tuple[str, ...], libcst.CSTNode], Any]):
        self._handler = symbol_handler
        self._context: list[str] = []

    def _enter(self, name: str) -> None:
        self._context.append(name)

    def _leave(self) -> None:
        self._context.pop()

    def _record_member(self, name: str, node: libcst.CSTNode) -> None:
        self._handler(tuple(self._context + [name]), node)

    def visit_ClassDef(self, node: libcst.ClassDef) -> None:
        name = get_full_name_or_fail(node)
        self._record_member(name, node)
        self._enter(name)

    def leave_ClassDef(self, original_node: libcst.ClassDef) -> None:
        self._leave()

    def visit_FunctionDef(self, node: libcst.FunctionDef) -> bool:
        name = get_full_name_or_fail(node)
        self._record_member(name, node)
        return False
