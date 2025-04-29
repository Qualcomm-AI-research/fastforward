# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import dataclasses

import libcst

from libcst.metadata import FunctionScope, GlobalScope, Scope
from libcst.metadata import ScopeProvider as _ScopeProvider
from libcst.metadata.scope_provider import ScopeVisitor as _ScopeVisitor

from fastforward._autoquant.cst import nodes


@dataclasses.dataclass(frozen=True)
class ImportSymbol:
    """Representation of a symbol imported from another module/package."""

    name: str
    asname: str | None = None
    module: str | None = None

    def as_node(self) -> libcst.SimpleStatementLine:
        """Create import CST node for symbol."""
        import_str = f"import {self.name}"
        if self.asname is not None:
            import_str = f"{import_str} as {self.asname}"
        if self.module is not None:
            import_str = f"from {self.module} {import_str}"
        statement = libcst.parse_statement(import_str)
        assert isinstance(statement, libcst.SimpleStatementLine)
        return statement


def find_required_imports(
    funcdef: libcst.FunctionDef, function_scope: Scope, module_name: str
) -> set[ImportSymbol]:
    """Find required imports from `function_scope` for `funcdef`.

    This assumes that `function_scope` is the relevant scope for `funcdef` or
    `funcdef` was created based on the scope associated with `function_scope`.
    This means that imports are only found if they occurred in the original
    scope. One exception: the `fastforward` package is always 'found' if
    required.

    Args:
        funcdef: The function CST for which to find required imports.
        function_scope: The scope to consider for imports. Any symbol defined
            in this scope's outer scopes (i.e., assignments or imports) that
            are used in `funcdef` will result in an discovered `ImportSymbol`.
        module_name: The name of the module in which the function associated
            with `function_scope` is defined. This is used to resolve
            relative imports and imports for symbols that are defined in
            `function_scope`'s module.

    Returns:
        A set of `ImportSymbol`s.
    """
    # Steps:
    # 1) Infer a new function scope for `funcdef` in an empty module. This ensures
    #    the function_scope only contains symbols local to the function and the
    #    function itself.
    # 2) For each symbol used in a 'load' context:
    #    - If the symbol is defined in the new local context, it does not require an import
    #    - If is not not defined in the new local context, but it is defined in
    #      `function_scope.parent`, generate an appropriate `ImportSymbol` for it.

    # Find the global scope (i.e., module scope) related to `function_scope`
    global_scope = function_scope
    while not isinstance(global_scope, GlobalScope):
        global_scope = global_scope.parent

    # Add `fastforward` to global scope if it is not there already
    fastforward_in_scope = "fastforward" in global_scope
    if not fastforward_in_scope:
        statement = libcst.parse_statement("import fastforward")
        assert isinstance(statement, libcst.SimpleStatementLine)
        global_scope.record_import_assignment(
            "fastforward", statement.body[0], libcst.Name("fastforward")
        )

    try:
        new_function_scope = _get_function_scope(funcdef)
        funcdef.visit(
            visitor := _RequiredSymbolsVisitor(
                outer_scope=function_scope.parent,
                function_scope=new_function_scope,
                module_name=module_name,
            )
        )
        return visitor.import_symbols
    finally:
        # cleanup fastforward from global scope if it was not part of it
        # originally.
        if not fastforward_in_scope:
            del global_scope._assignments["fastforward"]


def infer_scopes(module: libcst.Module) -> dict[libcst.CSTNode, Scope]:
    """Find scopes for each relevant node in `module`.

    Mapping will hold scope for `Module`, `ClassDef` and `FuncDef` nodes
    in module.
    """
    # unsafe_skip_copy is required because we use node identity for scope
    # lookup in `member_scope`
    scopes: dict[libcst.CSTNode, Scope] = {}
    wrapper = libcst.MetadataWrapper(module, unsafe_skip_copy=True)
    for scope in set(wrapper.resolve(ScopeProvider).values()):
        if scope is None:
            continue
        # Global scope has no assigned node, here we assign the module
        node = getattr(scope, "node", module)
        scopes[node] = scope
    return scopes


def _get_function_scope(funcdef: libcst.FunctionDef) -> FunctionScope:
    scopes = infer_scopes(libcst.Module([funcdef]))
    function_scope = scopes.get(funcdef)
    if function_scope is None or not isinstance(function_scope, FunctionScope):
        raise RuntimeError("Fatal: Unable to infer function scope")
    return function_scope


class _RequiredSymbolsVisitor(libcst.CSTVisitor):
    """Extract symbols that require import from CST."""

    def __init__(self, outer_scope: Scope, function_scope: FunctionScope, module_name: str) -> None:
        super().__init__()
        self._import_symbols: set[ImportSymbol] = set()
        self._outer_scope = outer_scope
        self._function_scope = function_scope
        self._module_name = module_name

    @property
    def import_symbols(self) -> set[ImportSymbol]:
        return set(self._import_symbols)

    def _record_import_symbol(self, symbol: str) -> bool:
        """Record import for `symbol` in `_import_symbols`."""
        if self._symbol_in_function_scope(symbol):
            return False
        if symbol not in self._outer_scope:
            return False

        for assignment in self._outer_scope[symbol]:
            if not (node := getattr(assignment, "node")):
                continue

            match node:
                case libcst.Import() | libcst.ImportFrom():
                    self._record_symbol_from_import_node(node, symbol)
                case _:
                    self._import_symbols.add(ImportSymbol(name=symbol, module=self._module_name))

        return True

    def _record_symbol_from_import_node(
        self, node: libcst.Import | libcst.ImportFrom, symbol: str
    ) -> None:
        names = node.names
        if isinstance(names, libcst.ImportStar):
            return

        for name in names:
            asname = _node_to_str(name.asname.name) if name.asname else None
            target = _node_to_str(name.name)

            local_name = asname or target
            if local_name == symbol:
                module = None
                if isinstance(node, libcst.ImportFrom):
                    import_module = _node_to_str(node.module) if node.module else None
                    relative_levels = len(node.relative)
                    module = _resolve_relative_module_name(
                        self._module_name, relative_levels, import_module
                    )
                self._import_symbols.add(ImportSymbol(name=target, asname=asname, module=module))

    def _symbol_in_function_scope(self, symbol: str) -> bool:
        symbol_parts = _importable_attribute_parts(libcst.parse_expression(symbol))
        return symbol_parts[0] in self._function_scope

    def visit_Name(self, node: libcst.Name) -> bool:
        _ = self._record_import_symbol(node.value)
        return False

    def visit_Attribute(self, node: libcst.Attribute) -> bool:
        str_repr = libcst.Module([]).code_for_node(node)
        if not self._record_import_symbol(str_repr):
            node.value.visit(self)
        return False

    def visit_FunctionDef(self, node: libcst.FunctionDef) -> bool:
        for node_ in [node.params, *node.decorators, node.returns, node.body]:
            if node_ is not None:
                node_.visit(self)
        return False

    def visit_Arg(self, node: libcst.Arg) -> bool:
        node.value.visit(self)
        return False

    def visit_Param(self, node: libcst.Param) -> bool:
        for node_ in [node.annotation, node.default]:
            if node_ is not None:
                node_.visit(self)
        return False

    def visit_Assign(self, node: libcst.Assign) -> bool:
        node.value.visit(self)
        return False

    def visit_AnnAssign(self, node: libcst.AnnAssign) -> bool:
        if node.value:
            node.value.visit(self)
        return False

    def visit_AugAssign(self, node: libcst.AugAssign) -> bool:
        node.value.visit(self)
        return False

    def visit_GeneralAssignment(self, node: nodes.GeneralAssignment) -> bool:
        if node.value is not None:
            node.value.visit(self)
        return False


class ScopeVisitor(_ScopeVisitor):
    """Extension of `_ScopeVisitor` that also included assignments."""

    def __init__(self, provider: _ScopeProvider, global_scope: GlobalScope | None = None) -> None:
        super().__init__(provider=provider)
        if global_scope:
            self.scope = global_scope

    def visit_GeneralAssignment(self, node: nodes.GeneralAssignment) -> bool:
        return self.on_visit(node.original)

    def visit_Assign(self, node: libcst.Assign) -> None:
        for target in node.targets:
            self._record_assignment(target.target, node)

    def visit_AnnAssign(self, node: libcst.AnnAssign) -> None:
        if node.value is not None:
            self._record_assignment(node.target, node)

    def visit_AugAssign(self, node: libcst.AugAssign) -> None:
        self._record_assignment(node.target, node)

    def _record_assignment(
        self,
        target: libcst.BaseAssignTargetExpression,
        assignment: libcst.Assign | libcst.AugAssign | libcst.AnnAssign,
    ) -> None:
        if isinstance(target, libcst.Name):
            self.provider.set_metadata(target, self.scope)
            self.scope.record_assignment(target.value, assignment)


class ScopeProvider(_ScopeProvider):
    """Extension of `_ScopeProvider` that also included assignments."""

    def visit_Module(self, node: libcst.Module) -> None:
        visitor = ScopeVisitor(self)
        node.visit(visitor)
        visitor.infer_accesses()


def _resolve_relative_module_name(
    relative_root: str, relative_levels: int, module_name: str | None
) -> str:
    module_name = module_name or ""
    if relative_levels == 0:
        return module_name

    name_atoms = relative_root.split(".")
    resolved_module = ".".join(name_atoms[:-relative_levels])
    if resolved_module:
        return f"{resolved_module}.{module_name}"
    else:
        return module_name


def _node_to_str(node: libcst.CSTNode) -> str:
    return libcst.Module([]).code_for_node(node)


def _importable_attribute_parts(
    node: libcst.Attribute | libcst.Name | libcst.BaseExpression,
) -> list[str]:
    """Given an attribute node, return a sequence of elements that could have been imported.

    For example: given an attribute node of `ant.bat[5].cat`, return `("ant", "bat")`,
    indicating that the possible imports requires for this expression are
    `ant` and  `ant.bat`.
    """
    match node:
        case libcst.Name(value):
            return [value]
        case libcst.Attribute(value, attr):
            return _importable_attribute_parts(value) + _importable_attribute_parts(attr)
        case _:
            return []
