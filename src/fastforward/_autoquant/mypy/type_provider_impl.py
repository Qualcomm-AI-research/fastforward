# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import contextlib
import contextvars
import dataclasses
import io
import sysconfig

from pathlib import Path
from typing import Iterator, TypeAlias

import libcst.metadata
import mypy.build
import mypy.checker
import mypy.checkexpr
import mypy.errors
import mypy.messages
import mypy.modulefinder
import mypy.nodes
import mypy.options
import mypy.subtypes
import mypy.types

from typing_extensions import override

from fastforward._autoquant.mypy.traverser import TraverserVisitor
from fastforward._autoquant.tsp.client import StaleSnapshotError, TypeClient
from fastforward._autoquant.tsp.context import current_module, module_context
from fastforward._autoquant.tsp.type_provider_impl import TspTypeProvider, make_call_scoped_cache
from fastforward._autoquant.tsp.types import TypeInfo as TypeInfo

CodeRange = libcst.metadata.CodeRange
CodePosition = libcst.metadata.CodePosition
mypy_module_context = module_context


@dataclasses.dataclass
class _MypyTypeOperations:
    """Preserve native mypy type operations behind the shared metadata interface."""

    _checker: mypy.checker.TypeChecker = dataclasses.field(repr=False)

    def is_proven_scalar(self, typ: mypy.types.Type) -> bool:
        """Read numeric facts without a checker query; Any is not evidence."""

        def numeric(typ: mypy.types.Type) -> bool:
            proper = mypy.types.get_proper_type(typ)
            if isinstance(proper, mypy.types.UnionType):
                return all(numeric(item) for item in proper.items)
            if isinstance(proper, mypy.types.LiteralType):
                proper = proper.fallback
            return isinstance(proper, mypy.types.Instance) and proper.type.fullname in (
                "builtins.int",
                "builtins.float",
                "builtins.bool",
            )

        return numeric(typ)

    def is_subtype(self, typ: mypy.types.Type, qualified_name: str) -> bool:
        """Check if this type is a subtype of the specified qualified type.

        Args:
            typ: The native mypy type to inspect.
            qualified_name: The fully qualified name of the potential supertype.

        Returns:
            True if this type is a subtype of the specified type, False otherwise.
        """
        if isinstance(typ, mypy.types.AnyType):
            return True
        try:
            supertype = self._checker.named_type(qualified_name)
            return mypy.subtypes.is_subtype(typ, supertype)
        except KeyError:
            return False

    def is_sequence_of_subtype(self, typ: mypy.types.Type, qualified_name: str) -> bool:
        """Check if this type is a subtype of `Sequence[<qualified_name>]`.

        Callers that accept collections of a given element type (e.g.
        `torch.cat` takes `Sequence[Tensor]`) need to recognise
        `list[Tensor]` / `tuple[Tensor, ...]` arguments alongside a bare
        `Tensor`. `is_subtype` alone rejects those because a `list` is
        not a `Tensor`.

        In practice mypy often infers `list[Any]` for expressions like
        `[torch.exp(x), torch.sin(x)]` because the underlying C ops return
        `Any`. A Sequence whose element types are all `Any` is accepted too —
        a strict `Sequence[Tensor]` check would otherwise leave the call
        unquantized when mypy has partial information.
        """
        if isinstance(typ, mypy.types.AnyType):
            return True

        try:
            elem_type = self._checker.named_type(qualified_name)
            seq_type = self._checker.named_generic_type("typing.Sequence", [elem_type])
            if mypy.subtypes.is_subtype(typ, seq_type):
                return True
        except KeyError:
            pass

        if isinstance(typ, mypy.types.Instance) and typ.args:
            if all(isinstance(arg, mypy.types.AnyType) for arg in typ.args):
                try:
                    any_elem = mypy.types.AnyType(mypy.types.TypeOfAny.special_form)
                    seq_of_any = self._checker.named_generic_type("typing.Sequence", [any_elem])
                    return mypy.subtypes.is_subtype(typ, seq_of_any)
                except KeyError:
                    return False

        return False


class MypyClient(TypeClient):
    """Adapt native mypy analysis to the shared document and type-query lifecycle."""

    def __init__(self) -> None:
        self._documents: dict[str, dict[CodeRange, TypeInfo]] = {}
        self._snapshot = 0

    @override
    def open_document(self, uri: str, text: str) -> None:
        self._snapshot += 1
        self._documents[uri] = {}
        state = _get_mypy_tree_and_checker(text)
        if state is None:
            return
        tree, checker = state
        types: dict[CodeRange, mypy.types.Type] = {}
        _TypeExtractionVisitor(checker, range_type_map=types).visit(tree)
        operations = _MypyTypeOperations(checker.chk)
        self._documents[uri] = {
            code_range: TypeInfo(typ, operations) for code_range, typ in types.items()
        }

    @override
    def close_document(self, uri: str) -> None:
        self._documents.pop(uri, None)
        self._snapshot += 1

    @override
    def get_snapshot(self) -> int:
        return self._snapshot

    @override
    def get_computed_type(self, uri: str, code_range: CodeRange, snapshot: int) -> TypeInfo | None:
        if snapshot != self._snapshot:
            msg = "The mypy analysis snapshot is no longer current."
            raise StaleSnapshotError(msg)
        return self._documents.get(uri, {}).get(code_range)

    @override
    def close(self) -> None:
        self._documents.clear()
        self._snapshot += 1


class MypyTypeProvider(TspTypeProvider):
    """Collect mypy types using the same metadata provider as remote TSP backends."""

    CLIENT_FACTORY = MypyClient


_mypy_client_scope = make_call_scoped_cache(MypyClient)


_BUILTIN_CONSTRUCTORS: dict[str, str] = {
    "float": "builtins.float",
    "int": "builtins.int",
    "bool": "builtins.bool",
    "complex": "builtins.complex",
}


class _TypeExtractionVisitor(TraverserVisitor):
    """Visitor that extracts type information from mypy AST nodes.

    This visitor traverses a mypy AST and uses mypy's expression checker to
    extract type information for each node. It maps the extracted types to
    their corresponding code ranges, which can later be used to associate types
    with CST nodes.

    Arguments:
        expr_checker: A mypy ExpressionChecker that can infer types for AST nodes
        range_type_map: A dictionary mapping code ranges to their inferred
            types. The provided dictionary is populated during the AST traversal.
    """

    def __init__(
        self,
        expr_checker: mypy.checkexpr.ExpressionChecker,
        range_type_map: dict[CodeRange, mypy.types.Type],
    ):
        self.expr_checker = expr_checker
        self.range_type_map = range_type_map

        self.expr_checker.chk.binder.push_frame()

    def _code_range_for_node(self, node: mypy.nodes.Node) -> CodeRange | None:
        if node.end_line is None or node.end_column is None:
            return None
        return CodeRange(
            CodePosition(node.line, node.column), CodePosition(node.end_line, node.end_column)
        )

    def visit_func_def(self, o: mypy.nodes.FuncDef, /) -> None:
        with self.expr_checker.chk.scope.push_function(o):
            super().visit_func_def(o)

    def _get_prestored_type(self, node: mypy.nodes.Node) -> mypy.types.Type | None:
        """Return a type for node from pre-stored mypy AST data, without the binder.

        The binder can be in a corrupted state when processing large modules
        (e.g. torch.nn.functional), causing expr_checker.accept() to return
        AnyType for all expressions. This method reads types that mypy stored
        during build() and are immune to that corruption.

        Covers:
          (a) NameExpr / MemberExpr pointing to a Var: read Var.type directly.
          (b) MemberExpr on a typed object: look up the attribute in the class
              symbol table via the object expression's pre-stored Var type.
          (c) Literal expressions: return their known built-in types.
          (d) Built-in type constructor calls (float(), int(), bool(), ...):
              return the constructor's return type without evaluating the call.
          (e) Binary operations where both operands resolve recursively to
              non-Tensor pre-stored types: propagate float as scalar result.
          (f) Unary operations where the operand resolves to a non-Tensor
              pre-stored type: propagate float as scalar result.
        """
        chk = self.expr_checker.chk

        # (a) RefExpr (NameExpr or MemberExpr) pointing directly to a typed Var
        if isinstance(node, mypy.nodes.RefExpr):
            if isinstance(node.node, mypy.nodes.Var) and node.node.type is not None:
                return node.node.type

        # (b) MemberExpr where the object expression is a typed Var
        # e.g. self.num_heads where self: Foo and Foo.num_heads: int
        if isinstance(node, mypy.nodes.MemberExpr):
            expr = node.expr
            if isinstance(expr, mypy.nodes.RefExpr):
                obj_var = expr.node
                if isinstance(obj_var, mypy.nodes.Var) and isinstance(
                    obj_var.type, mypy.types.Instance
                ):
                    sym = obj_var.type.type.names.get(node.name)
                    if sym is not None and isinstance(sym.node, mypy.nodes.Var):
                        member_type = sym.node.type
                        if member_type is not None:
                            return member_type

        # (c) Literal expressions — types are always known regardless of binder
        try:
            if isinstance(node, mypy.nodes.IntExpr):
                return chk.named_type("builtins.int")
            if isinstance(node, mypy.nodes.FloatExpr):
                return chk.named_type("builtins.float")
            if isinstance(node, mypy.nodes.StrExpr):
                return chk.named_type("builtins.str")
            if isinstance(node, mypy.nodes.BytesExpr):
                return chk.named_type("builtins.bytes")
        except KeyError:
            pass

        # (d) Built-in type constructor calls: float(x), int(x), bool(x), complex(x).
        # These always return a scalar type regardless of the argument.
        if isinstance(node, mypy.nodes.CallExpr) and isinstance(node.callee, mypy.nodes.NameExpr):
            if node.callee.name in _BUILTIN_CONSTRUCTORS:
                try:
                    return chk.named_type(_BUILTIN_CONSTRUCTORS[node.callee.name])
                except KeyError:
                    pass

        # (e) and (f) require knowing whether operand types are Tensor — get
        # the tensor type once, bail out if unavailable (e.g. torch not imported).
        try:
            tensor_type = chk.named_type("torch.Tensor")
        except KeyError:
            return None

        # (e) Binary operations where both operands resolve recursively to known
        # non-Tensor types. Stores float as a conservative scalar result so that
        # outer expressions (e.g. math.sqrt(1.0 / float(E))) can see their
        # argument is non-Tensor rather than AnyType from the corrupted binder.
        if isinstance(node, mypy.nodes.OpExpr):
            left_type = self._get_prestored_type(node.left)
            right_type = self._get_prestored_type(node.right)
            if (
                left_type is not None
                and right_type is not None
                and not mypy.subtypes.is_subtype(left_type, tensor_type)
                and not mypy.subtypes.is_subtype(right_type, tensor_type)
            ):
                try:
                    return chk.named_type("builtins.float")
                except KeyError:
                    pass

        # (f) Unary operations where the operand resolves to a known non-Tensor type.
        if isinstance(node, mypy.nodes.UnaryExpr):
            operand_type = self._get_prestored_type(node.expr)
            if operand_type is not None and not mypy.subtypes.is_subtype(operand_type, tensor_type):
                try:
                    return chk.named_type("builtins.float")
                except KeyError:
                    pass

        return None

    def enter_node(self, node: mypy.nodes.Node) -> bool:
        if (code_range := self._code_range_for_node(node)) is None:
            return True  # Still try children

        # Priority 1: read pre-stored types that are immune to binder corruption.
        # These cover NameExpr, typed MemberExpr (self.*), and literal expressions.
        if (typ := self._get_prestored_type(node)) is not None:
            self.range_type_map[code_range] = typ
            return True

        # Priority 2: fall back to expr_checker.accept() for all other expressions.
        try:
            if isinstance(node, mypy.nodes.Expression):
                output_buf = io.StringIO()
                with contextlib.redirect_stdout(output_buf), contextlib.redirect_stderr(output_buf):
                    self.range_type_map[code_range] = self.expr_checker.accept(
                        node, allow_none_return=True, always_allow_any=True
                    )
        except TypeError:
            pass
        except SystemExit:
            # Mypy may raise SystemExit when it encounters certain errors
            # during type checking. This typically happens when we don't
            # properly maintain the type checker state that mypy would normally
            # update (e.g., in visit_func_def). We catch and ignore these exits
            # for now, and can revisit if we find missing type information that
            # should be present.
            pass
        return True


def _get_mypy_base_dir(
    module_name: str, module_path: str | None, python_executable: str | None
) -> str | None:
    """Return a local module's search root without shadowing library stubs."""
    if module_path is None:
        return None

    # Mypy uses base_dir, rather than path, to locate sibling source modules.
    # Derive it from the known identity without parsing package names as Python
    # identifiers: dynamic loaders may use hyphenated directory names.
    path = Path(module_path).absolute()
    package_depth = module_name.count(".") + (path.stem == "__init__")
    if not path.is_file() or package_depth >= len(path.parents):
        return None

    root = path.parents[package_depth]
    _, site_dirs = mypy.modulefinder.get_search_dirs(python_executable)
    library_dirs = [
        *site_dirs,
        sysconfig.get_path("stdlib"),
        sysconfig.get_path("platstdlib"),
    ]
    # Keep library lookup on mypy's normal search paths. Promoting a
    # library directory to user source would shadow bundled stubs.
    if any(root.resolve() == Path(directory).resolve() for directory in library_dirs):
        return None
    return str(root)


def _get_mypy_tree_and_checker(
    code: str,
) -> tuple[mypy.nodes.MypyFile, mypy.checkexpr.ExpressionChecker] | None:
    """Create a mypy AST and expression checker for the given code.

    This function builds a mypy AST from the provided code string and creates
    an expression checker that can be used for type checking. The code is treated
    in its original module context, or a temporary module for standalone code.

    The expression checker is a mypy `NodeVisitor` that can visit arbitrary
    expression nodes in the mypy AST an return the infered type.

    Args:
        code: The Python code to parse and type check

    Returns:
        A tuple containing the mypy AST and an expression checker, or None if
        the AST could not be built
    """
    module_name, module_path = current_module()
    cache_key = (code, module_name, module_path)
    cache = _call_scoped_cache.get()
    if cache is not None and (cached := cache.get(cache_key)) is not None:
        return cached

    opts = mypy.options.Options()
    opts.show_traceback = True
    opts.preserve_asts = True
    # The native parser reads `module_path` from disk and ignores `text`, but `code`
    # can differ from the file on disk.
    opts.native_parser = False

    src = mypy.build.BuildSource(
        path=module_path,
        module=module_name,
        text=code,
        base_dir=_get_mypy_base_dir(module_name, module_path, opts.python_executable),
    )
    result = mypy.build.build(
        sources=[src],
        options=opts,
    )

    state = result.graph[module_name]

    if state.tree is None:
        return None

    type_checker = state.type_checker()
    manager = result.manager
    errors = mypy.errors.Errors(options=opts)
    msgs = mypy.messages.MessageBuilder(errors, manager.modules)

    type_checker.msg = msgs
    expr_checker = mypy.checkexpr.ExpressionChecker(
        type_checker, type_checker.msg, type_checker.plugin, {}
    )

    typed_result = (state.tree, expr_checker)
    if cache is not None:
        cache[cache_key] = typed_result
    return typed_result


_CallScopedCache: TypeAlias = dict[
    tuple[str, str, str | None], tuple[mypy.nodes.MypyFile, mypy.checkexpr.ExpressionChecker]
]

_call_scoped_cache: contextvars.ContextVar[_CallScopedCache | None] = contextvars.ContextVar(
    "fastforward_autoquant_mypy_call_scoped_cache", default=None
)


@contextlib.contextmanager
def mypy_call_scoped_cache() -> Iterator[None]:
    """Activate an in-call cache keyed by source text and module identity.

    While active, `_get_mypy_tree_and_checker(code)` returns a cached result on
    repeated calls with the same code, module name, and path. The cache is
    created empty on enter and discarded on exit, so it cannot outlive the call.
    """
    token = _call_scoped_cache.set({})
    try:
        with _mypy_client_scope():
            yield
    finally:
        _call_scoped_cache.reset(token)
