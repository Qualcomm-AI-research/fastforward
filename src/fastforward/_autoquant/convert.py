# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


import copy

from typing import cast

import libcst
import libcst.helpers

from fastforward._autoquant.cfg.exceptions import CFGError
from fastforward._autoquant.cst.passes import QuantizedCounterpartReplacer
from fastforward._autoquant.pysource.scope import ImportSymbol, find_required_imports
from fastforward._quantops import OperatorTable

from .cfg import block_node_elems, blocks, construct, reconstruct, variable_tracking
from .cst import nodes
from .pybuilder import FunctionBuilder, QuantizedFunctionBuilder, QuantizedModuleBuilder
from .pysource import PySource


def convert_method(
    src: PySource, clsbuilder: QuantizedModuleBuilder, optable: OperatorTable
) -> FunctionBuilder:
    """Convert a single method to a quantized method.

    The function represented by `src` is considered a method of the class built
    using `clsbuilder`. Quantizers required for the newly created quantized
    function will be added to this class. All operator rewriting from
    non-quantized to quantized function calls is based on `optable`.

    Args:
        src: `PySource` object that represents method that will be quantized
        clsbuilder: The `PyBuilder` object for the quantized class of which the
            newly quantized method will be part of.
        optable: The `OperatorTable` that is used as 'ground-truth' for
            operator replacement.

    Returns:
        A CST that represents a quantized method. This method is not yet added
        to `clsbuilder`, but its required quantizers are.
    """
    src_cst = src.cst(NodeType=libcst.FunctionDef)

    src_cst = _rewrite_quantized_operators(src_cst, optable)

    cfg = construct(src_cst)
    _add_input_quantizers(cfg, clsbuilder=clsbuilder)
    dst_cst = reconstruct(cfg)

    assert isinstance(dst_cst, libcst.FunctionDef)
    required_imports = _infer_imports(src, dst_cst)

    return QuantizedFunctionBuilder(dst_cst, required_imports)


def _rewrite_quantized_operators(
    cst: libcst.FunctionDef, optable: OperatorTable
) -> libcst.FunctionDef:
    """Rewrite function call to quantized function calls.

    Replaces all function calls in `cst` that appear in `optable` to a
    quantized function call. Also introduces the appropriate quantizers on
    `clsbuilder`.
    """
    function_replacement = QuantizedCounterpartReplacer(optable=optable)
    new_cst = cast(libcst.FunctionDef, cst.visit(function_replacement))

    return new_cst


def _add_input_quantizers(cfg: blocks.FunctionBlock, clsbuilder: QuantizedModuleBuilder) -> None:
    """Add input quantizers and appropriate calls in `cfg` and `clsbuilder`.

    Add the appropriate input quantizers for symbols that are used in a
    "quantized tensor" place but cannot be determined to be quantized.

    Args:
        cfg: The CFG to which quantizer calls will be added.
        clsbuilder: The builder object for the quantized class to which
            quantizer definitions are added.
    """
    block_trackers = variable_tracking.infer_block_dataflow(cfg)

    for block in cfg.blocks():
        active_vars = copy.copy(block_trackers[block].vars_in)
        vars_local = block_trackers[block].vars_local
        vars_gen = block_trackers[block].vars_gen
        vars_all_gen = vars_local.union(vars_gen)
        for decl_node, assign_or_call in block_node_elems.extract_nodes_from_block(
            block, node_types=(nodes.GeneralAssignment, libcst.Call), include_subclasses=True
        ):
            match assign_or_call:
                case nodes.GeneralAssignment():
                    _process_assignments(
                        assign=assign_or_call,
                        active_vars=active_vars,
                        local_vars=vars_all_gen,
                        declaration_node=decl_node,
                    )
                case nodes.QuantizedCall():
                    _process_quantized_call(assign_or_call, active_vars, clsbuilder)
                case _:
                    pass


def _process_assignments(
    assign: nodes.GeneralAssignment,
    active_vars: variable_tracking.VariableSet,
    local_vars: variable_tracking.VariableSet,
    declaration_node: libcst.CSTNode,
) -> None:
    """Process assignment node `assign` and update `active_vars`.

    Block trackers represent the variable state at the start and conclusion of a block,
    however, during execution the variable state also changes. This function updates
    the `active_vars` set to represent new assignments during the execution of a block.

    Args:
        assign: The CST node that represents the assignment.
        active_vars: Variable set that contains active variables.
        local_vars: A set of variables that is assigned in the current block.
        declaration_node: The node that represents the line on which the
            assignment took place. This is used as `declaration_node` for variables
            in the variable sets and can be used for lookups.
    """
    for target in assign.targets:
        if isinstance(target, libcst.Name):
            name = target.value
            found_variable = False
            for var in local_vars.find_variables_for_node(declaration_node):
                if var.name == target.value:
                    _ = active_vars.remove(name)
                    active_vars.add(var)
                    found_variable = True

            if not found_variable:
                msg = (
                    "There is no corresponding variable in 'local_vars' for assignment 'assign' "
                    "This can happen when the assignment was added after the variable tracking sets "
                    "were created. This is not supported."
                )
                raise RuntimeError(msg)


def _process_quantized_call(
    node: nodes.QuantizedCall,
    active_vars: variable_tracking.VariableSet,
    clsbuilder: QuantizedModuleBuilder,
) -> None:
    """Process a quantized call and ensure appropriate quantizers are added.

    For each argument to the quantized call represented by `node`, make sure
    that their input is quantized. This may be satisfied by the variable
    already being quantized, or by introducing a new quantizer for the
    variable.

    Args:
        node: The quantized call node.
        active_vars: The set of active variables.
        clsbuilder: The builder object for the quantized class.
    """
    args: list[libcst.BaseExpression] = []
    kwargs: dict[str, libcst.BaseExpression] = {}
    seen_kwarg = False
    for arg in node.args:
        if (kw := arg.keyword) is not None:
            kwargs[kw.value] = arg.value
            seen_kwarg = True
        elif seen_kwarg:
            raise SyntaxError("Positional argument follows keyword argument")
        else:
            args.append(arg.value)

    for param, value in node.operator.bind_partial(*args, **kwargs):
        if not param.quantized:
            continue
        match value:
            case libcst.Name(value=name):
                _ensure_quantized(name, active_vars, clsbuilder=clsbuilder)
            case _:
                # Process attributes and other non-Name accessors (#133)
                pass


def _infer_imports(src: PySource, cst: libcst.FunctionDef) -> set[ImportSymbol]:
    if (scope := src.scope()) is None:
        # Cannot infer import if no scope information is available.
        # Generally, this should not happen.
        # Do not error here to ensure autoquant does not completely fail
        # in this case.
        return set()
    return find_required_imports(cst, scope, src.module().qualified_name)


def _ensure_quantized(
    name: str, active_vars: variable_tracking.VariableSet, clsbuilder: QuantizedModuleBuilder
) -> None:
    """Ensure that a variable identified by `name` is quantized.

    The variable identified by `name` may have multiple origins captured in
    `active_vars`. This function ensures that all cases are quantized.

    Args:
        name: The variable name.
        active_vars: A representation of the active scope.
        clsbuilder: The builder object for the quantized class.
    """
    for var in active_vars[name]:
        if var.quantization_status is not variable_tracking.QuantizationStatus.Quantized:
            _insert_quantizer_for(var, clsbuilder)


def _insert_quantizer_for(
    var: variable_tracking.Variable, clsbuilder: QuantizedModuleBuilder
) -> None:
    """Insert a quantizer and quantizer call for `var`."""
    visitor = _InsertQuantizerVisitor(variable=var, clsbuilder=clsbuilder)
    var.declaration_block.visit(visitor)


class _InsertQuantizerVisitor:
    """Visitor that implements block specific quantizer insertion."""

    def __init__(
        self,
        variable: variable_tracking.Variable,
        clsbuilder: QuantizedModuleBuilder,
    ) -> None:
        self.clsbuilder = clsbuilder
        self.variable = variable

    def visit_FunctionBlock(self, block: blocks.FunctionBlock) -> None:
        self._insert_in_referenced(block, "body")

    def visit_IfBlock(self, block: blocks.IfBlock) -> None:
        pass

    def visit_ForBlock(self, block: blocks.ForBlock) -> None:
        self._insert_in_referenced(block, "body")

    def visit_WhileBlock(self, block: blocks.WhileBlock) -> None:
        self._insert_in_referenced(block, "body")

    def visit_WithBlock(self, block: blocks.WithBlock) -> None:
        self._insert_in_referenced(block, "body")

    def visit_MarkerBlock(self, block: blocks.MarkerBlock) -> None:
        if (next_block := block.next_block) is not None:
            next_block.visit(self)
        else:
            msg = "Encountered dangling MarkerBlock in CFG"
            raise CFGError(msg)

    def visit_ExitBlock(self, _block: blocks.ExitBlock) -> None:
        pass

    def visit_SimpleBlock(self, block: blocks.SimpleBlock) -> None:
        statement_idx = block.statements.index(self.variable.declaration_node)
        quantizer_name = nodes.QuantizerReference(self.variable.name)
        block.statements = (
            *block.statements[: statement_idx + 1],
            _create_quantize_statement(self.variable.name, quantizer_name),
            *block.statements[statement_idx + 1 :],
        )
        self.variable.mark_quantized()

    def _insert_in_referenced(self, block: blocks.Block, next_block_attr: str) -> None:
        """Insert quantizer as first statement in block `next_block_attr`.

        If the block referenced by `next_block_attr` is not a `SimpleBlock`,
        create a new `SimpleBlock` between the referenced block and `block` and
        insert the quantizer there.
        """
        next_block = getattr(block, next_block_attr, None)
        if next_block is None or not isinstance(next_block, blocks.Block):
            msg = f"'{type(block).__name__}' does not have a block reference '{next_block_attr}'"
            raise ValueError(msg)
        if not isinstance(next_block, blocks.SimpleBlock):
            blocks.insert_block_between(block, next_block)
            next_block = getattr(block, next_block_attr)

        assert isinstance(next_block, blocks.SimpleBlock)
        quantizer_name = nodes.QuantizerReference(self.variable.name)
        next_block.statements = (
            _create_quantize_statement(self.variable.name, quantizer_name),
            *next_block.statements,
        )
        self.variable.mark_quantized()


def _create_quantize_statement(
    name: str, quantizer_name: nodes.QuantizerReference
) -> libcst.SimpleStatementLine:
    """Create a quantize statement.

    Args:
        name: The name of the variable to be quantized.
        quantizer_name: The name of the quantizer.

    Returns:
        The quantize statement.
    """
    name_node = libcst.Name(name)
    quantize_statement = libcst.helpers.parse_template_statement(
        "{name} = self.{quantizer_name}({name})",
        name=name_node,
        quantizer_name=quantizer_name,
    )
    assert isinstance(quantize_statement, libcst.SimpleStatementLine)
    return quantize_statement
