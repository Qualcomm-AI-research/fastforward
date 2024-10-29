import inspect
import textwrap

from types import EllipsisType, ModuleType
from typing import Any, Callable, Sequence, TypeVar, Union, overload

import libcst

_T = TypeVar("_T")


class PySourceNode:
    def __init__(
        self, name: str, cst: libcst.CSTNode, parent: "PySourceNode | None" = None
    ) -> None:
        self.name = name
        self.cst = cst
        self._members: dict[str, PySourceNode] = {}
        self._parent = parent

    def set_member(self, name: str, node: "PySourceNode") -> None:
        self._members[name] = node

    @overload
    def get_member(self, name: str) -> "PySourceNode": ...
    @overload
    def get_member(self, name: str, default: EllipsisType) -> "PySourceNode": ...
    @overload
    def get_member(self, name: str, default: _T) -> Union[_T, "PySourceNode"]: ...

    def get_member(self, name: str, default: _T | EllipsisType = ...) -> Union[_T, "PySourceNode"]:
        if name in self._members:
            return self._members[name]
        elif not isinstance(default, EllipsisType):
            return default
        raise KeyError(f"{name} is not a member of {self}")

    def __getattr__(self, name: str) -> "PySourceNode":
        if name in self._members:
            return self._members[name]
        return object.__getattribute__(self, name)  # type: ignore[no-any-return]


class PySourceClass(PySourceNode):
    pass


def py_source_node(node_name: str, node: libcst.CSTNode, parent: PySourceNode) -> PySourceNode:
    match node:
        case libcst.ClassDef():
            return PySourceClass(node_name, node, parent)
        case _:
            return PySourceNode(node_name, node, parent)


class PySourceModule(PySourceNode):
    """
    Representation of python module. Used for interacting with existing code
    and generating new code

    Args:
        module:
        validators: sequence of functions that accept a `libcst.Module`. May
            raise an error if the cst is not matching expectations.
        preprocessors: sequence of `libcst.CSTTransformer` to preprocess CST
            before any processing happens. Each is applied, in sequence, on the
            entire module.
    """

    def __init__(
        self,
        module: ModuleType,
        validators: Sequence[Callable[[libcst.Module], None]] = (),
        preprocessors: Sequence[libcst.CSTTransformer] = (),
    ) -> None:
        name = module.__name__

        src = inspect.getsource(module)
        cst = libcst.parse_module(textwrap.dedent(src))

        for validator in validators:
            validator(cst)

        for preprocessor in preprocessors:
            cst = cst.visit(preprocessor)

        super().__init__(name, cst, parent=None)
        self._module = module

        symbol_visitor = _SymbolVisitor(self._add_symbol)
        self.cst.visit(symbol_visitor)

    def _add_symbol(self, name: tuple[str, ...], node: libcst.CSTNode) -> None:
        parent: PySourceNode = self
        *path, node_name = name
        for ident in path:
            parent = parent.get_member(ident)
        parent.set_member(node_name, py_source_node(node_name, node, parent))


class _SymbolVisitor(libcst.CSTVisitor):
    def __init__(self, symbol_handler: Callable[[tuple[str, ...], libcst.CSTNode], Any]):
        self._symbol_callback = symbol_handler
        self._name_stack: list[str] = []

    def visit_FunctionDef(self, node: libcst.FunctionDef) -> None:
        self._handle(node.name.value, node)

    def _leave(self) -> None:
        self._name_stack.pop()

    def _visit(self, name: str) -> None:
        self._name_stack.append(name)

    def _handle(self, name: str, node: libcst.CSTNode) -> None:
        self._symbol_callback(tuple(self._name_stack + [name]), node)

    def visit_ClassDef(self, node: libcst.ClassDef) -> None:
        self._handle(node.name.value, node)
        self._visit(node.name.value)

    def leave_ClassDef(self, original_node: libcst.ClassDef) -> None:
        self._leave()
