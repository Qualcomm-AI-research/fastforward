# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


import logging

import libcst

from fastforward._autoquant.cst.passes import QuantizedCounterpartReplacer
from fastforward._autoquant.function_context import FunctionContext
from fastforward._autoquant.pybuilder import QuantizerReferenceCollection
from fastforward._autoquant.pysource.scope import ImportSymbol, find_required_imports
from fastforward._quantops import OperatorTable
from fastforward.type_common import MethodType

from .cst.quantizer_analysis.transformer import QuantizerFunctionTransformer
from .pass_manager import PassManager
from .pybuilder import QuantizedFunctionBuilder
from .pysource import PySource

logger = logging.getLogger(__name__)


class _AnchorBareSuperCalls(libcst.CSTTransformer):
    """Rewrite bare ``super()`` calls to anchored ``super(ClassName, self)`` calls."""

    def __init__(self, owner_class_name: str, self_name: str) -> None:
        self._owner_class_name = owner_class_name
        self._self_name = self_name

    def leave_Call(
        self,
        original_node: libcst.Call,
        updated_node: libcst.Call,
    ) -> libcst.BaseExpression:
        del original_node

        func = updated_node.func
        if not isinstance(func, libcst.Attribute):
            return updated_node

        super_call = func.value
        if not isinstance(super_call, libcst.Call):
            return updated_node

        if not isinstance(super_call.func, libcst.Name) or super_call.func.value != "super":
            return updated_node

        if len(super_call.args) != 0:
            # Already explicit super(...)
            return updated_node

        anchored_super = super_call.with_changes(
            args=(
                libcst.Arg(libcst.Name(self._owner_class_name)),
                libcst.Arg(libcst.Name(self._self_name)),
            )
        )
        return updated_node.with_changes(func=func.with_changes(value=anchored_super))


def _owner_class_name(src: PySource) -> str | None:
    parts = src.qualified_name.split(".")
    if len(parts) < 2:
        return None
    return parts[-2]


def _rewrite_super_calls(
    cst: libcst.FunctionDef,
    src: PySource,
    func_ctx: FunctionContext,
) -> libcst.FunctionDef:
    if func_ctx.method_type is not MethodType.METHOD or func_ctx.instance_var is None:
        return cst

    owner = _owner_class_name(src)
    if owner is None:
        return cst

    rewritten = cst.visit(_AnchorBareSuperCalls(owner, func_ctx.instance_var))
    assert isinstance(rewritten, libcst.FunctionDef)
    return rewritten


def convert_function(
    src: PySource,
    optable: OperatorTable,
    func_ctx: "FunctionContext",
    quantizer_refs: QuantizerReferenceCollection,
) -> QuantizedFunctionBuilder:
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
        method_type: Specifies the type of method, use `MethodType.NO_METHOD`
            for non-method functions.
        func_ctx: Context in which function appears in code.
        quantizer_refs: Quantizer reference collection used for autoquant.

    Returns:
        A CST that represents a quantized method. This method is not yet added
        to `clsbuilder`, but its required quantizers are.
    """
    logger.info(
        "convert_function: converting %s (%s)",
        src.qualified_name,
        func_ctx.method_type.name if func_ctx.method_type is not None else None,
    )
    src_cst = src.cst(NodeType=libcst.FunctionDef)
    required_imports_extra: set[ImportSymbol] = set()
    dst_cst = autoquantize_funcdef(
        src_cst=src_cst,
        optable=optable,
        func_ctx=func_ctx,
        quantizer_refs=quantizer_refs,
        required_imports_extra=required_imports_extra,
    )

    assert isinstance(dst_cst, libcst.FunctionDef)
    dst_cst = _rewrite_super_calls(dst_cst, src=src, func_ctx=func_ctx)
    required_imports = _infer_imports(src, dst_cst)
    required_imports |= required_imports_extra

    return QuantizedFunctionBuilder(dst_cst, required_imports, origin=func_ctx)


def autoquantize_funcdef(
    src_cst: libcst.FunctionDef,
    optable: OperatorTable,
    func_ctx: FunctionContext,
    quantizer_refs: QuantizerReferenceCollection,
    required_imports_extra: set[ImportSymbol] | None = None,
) -> libcst.FunctionDef:
    """Autoquantize a single `FuncDef` with given `optable`."""
    counterpart_replacer = QuantizedCounterpartReplacer(
        optable=optable,
        func_ctx=func_ctx,
        quantizer_refs=quantizer_refs,
    )
    pm = PassManager(
        passes=[
            counterpart_replacer,
            QuantizerFunctionTransformer(quantizer_refs=quantizer_refs),
        ]
    )
    converted = pm(src_cst)

    if required_imports_extra is not None:
        required_imports_extra.update(counterpart_replacer.required_imports_extra)

    return converted


def _infer_imports(src: PySource, cst: libcst.FunctionDef) -> set[ImportSymbol]:
    if (scope := src.scope()) is None:
        # Cannot infer import without scope info. Don't error to avoid autoquant failure
        return set()
    return find_required_imports(cst, scope, src.module().qualified_name)
