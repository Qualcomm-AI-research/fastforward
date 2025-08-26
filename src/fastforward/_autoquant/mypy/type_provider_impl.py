# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import contextlib
import dataclasses
import io

from typing import TypeAlias

import libcst
import libcst.metadata
import mypy.build
import mypy.checker
import mypy.checkexpr
import mypy.errors
import mypy.messages
import mypy.nodes
import mypy.options
import mypy.subtypes
import mypy.types

from typing_extensions import override

from fastforward._autoquant.mypy.traverser import TraverserVisitor

CodeRange = libcst.metadata.CodeRange
CodePosition = libcst.metadata.CodePosition
PositionProvider = libcst.metadata.PositionProvider


@dataclasses.dataclass
class TypeInfo:
    """Type information."""

    typ: mypy.types.Type
    _checker: mypy.checker.TypeChecker = dataclasses.field(repr=False)

    def is_subtype(self, qualified_name: str) -> bool:
        """Check if this type is a subtype of the specified qualified type.

        Args:
            qualified_name: The fully qualified name of the potential supertype.

        Returns:
            True if this type is a subtype of the specified type, False otherwise.
        """
        if isinstance(self.typ, mypy.types.AnyType):
            return True
        try:
            typ = self._checker.named_type(qualified_name)
            return mypy.subtypes.is_subtype(self.typ, typ)
        except KeyError:
            return False


_ProvidedType: TypeAlias = TypeInfo


class MypyTypeProvider(libcst.VisitorMetadataProvider[_ProvidedType]):
    """A metadata provider that uses mypy to extract type information from Python code.

    This provider analyzes the code using mypy's type checker and associates type
    information with corresponding CST nodes. It depends on the PositionProvider
    to map between CST nodes and their positions in the source code.

    The provider works by:
    1. Running mypy on the provided code to generate a typed AST
    2. Traversing the mypy AST to extract type information for each node
    3. Mapping the type information back to CST nodes based on code positions

    Type information is made available as metadata that can be retrieved for
    each CST node that has a corresponding type in the mypy analysis.
    """

    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self) -> None:
        super().__init__()
        self._range_type_map: dict[CodeRange, mypy.types.Type] = {}
        self._type_checker: mypy.checker.TypeChecker | None = None

    @override
    def visit_Module(self, node: libcst.Module) -> bool | None:
        if (mypy_state := _get_mypy_tree_and_checker(node.code)) is None:
            # Mypy failed, we cannot provide type information
            return False

        mypy_tree, expr_checker = mypy_state
        self._type_checker = expr_checker.chk

        visitor = _TypeExtractionVisitor(expr_checker, range_type_map=self._range_type_map)
        visitor.visit(mypy_tree)

        return True

    @override
    def on_visit(self, node: libcst.CSTNode) -> bool:
        # If this node's code range has a corresponding type in
        # _range_type_map, associate that type with this node
        coderange = self.get_metadata(PositionProvider, node, None)
        typ = self._range_type_map.get(coderange, None)

        if coderange is not None and typ is not None:  # type: ignore[redundant-expr]
            checker = self._type_checker
            assert checker is not None
            self.set_metadata(node, TypeInfo(typ, checker))

        return super().on_visit(node)


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

    def enter_node(self, node: mypy.nodes.Node) -> bool:
        if (code_range := self._code_range_for_node(node)) is None:
            return True  # Still try children
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


def _get_mypy_tree_and_checker(
    code: str,
) -> tuple[mypy.nodes.MypyFile, mypy.checkexpr.ExpressionChecker] | None:
    """Create a mypy AST and expression checker for the given code.

    This function builds a mypy AST from the provided code string and creates
    an expression checker that can be used for type checking. The code is treated
    as a temporary module within mypy for evaluation purposes.

    The expression checker is a mypy `NodeVisitor` that can visit arbitrary
    expression nodes in the mypy AST an return the infered type.

    Args:
        code: The Python code to parse and type check

    Returns:
        A tuple containing the mypy AST and an expression checker, or None if
        the AST could not be built
    """
    eval_module_name = "_ff_evaluation_module__"
    src = mypy.build.BuildSource(
        path=None,
        module=eval_module_name,
        text=code,
    )
    opts = mypy.options.Options()
    opts.show_traceback = True
    opts.preserve_asts = True

    result = mypy.build.build(
        sources=[src],
        options=opts,
    )

    state = result.graph[eval_module_name]

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

    return state.tree, expr_checker
