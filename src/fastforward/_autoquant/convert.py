# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


import libcst

from fastforward._autoquant.cst.passes import QuantizedCounterpartReplacer
from fastforward._autoquant.function_context import FunctionContext
from fastforward._autoquant.pybuilder import QuantizerReferenceCollection
from fastforward._autoquant.pysource.scope import ImportSymbol, find_required_imports
from fastforward._quantops import OperatorTable

from .cst.quantizer_analysis.transformer import QuantizerFunctionTransformer
from .pass_manager import PassManager
from .pybuilder import QuantizedFunctionBuilder
from .pysource import PySource


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
    src_cst = src.cst(NodeType=libcst.FunctionDef)
    dst_cst = autoquantize_funcdef(
        src_cst=src_cst,
        optable=optable,
        func_ctx=func_ctx,
        quantizer_refs=quantizer_refs,
    )

    assert isinstance(dst_cst, libcst.FunctionDef)
    required_imports = _infer_imports(src, dst_cst)

    return QuantizedFunctionBuilder(dst_cst, required_imports, origin=func_ctx)


def autoquantize_funcdef(
    src_cst: libcst.FunctionDef,
    optable: OperatorTable,
    func_ctx: FunctionContext,
    quantizer_refs: QuantizerReferenceCollection,
) -> libcst.FunctionDef:
    """Autoquantize a single `FuncDef` with given `optable`."""
    pm = PassManager(
        passes=[
            QuantizedCounterpartReplacer(
                optable=optable,
                func_ctx=func_ctx,
                quantizer_refs=quantizer_refs,
            ),
            QuantizerFunctionTransformer(quantizer_refs=quantizer_refs),
        ]
    )
    return pm(src_cst)


def _infer_imports(src: PySource, cst: libcst.FunctionDef) -> set[ImportSymbol]:
    if (scope := src.scope()) is None:
        # Cannot infer import without scope info. Don't error to avoid autoquant failure
        return set()
    return find_required_imports(cst, scope, src.module().qualified_name)
