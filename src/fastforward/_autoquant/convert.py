# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


from typing import Any, Callable

import libcst

from fastforward._autoquant.cst.passes import QuantizedCounterpartReplacer
from fastforward._autoquant.pysource.scope import ImportSymbol, find_required_imports
from fastforward._quantops import OperatorTable

from .cst.quantizer_analysis.transformer import QuantizerFunctionTransformer
from .pass_manager import MetadataTransformer, PassManager
from .pybuilder import FunctionBuilder, QuantizedFunctionBuilder
from .pysource import PySource


def convert_funcdef(
    src: PySource,
    optable: OperatorTable,
    func_ref: Callable[..., Any] | None,
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
        method_type: Specifies the type of method, use `MethodType.NO_METHOD`
            for non-method functions.
        func_ref: A reference to the Python function that is converted.

    Returns:
        A CST that represents a quantized method. This method is not yet added
        to `clsbuilder`, but its required quantizers are.
    """
    src_cst = src.cst(NodeType=libcst.FunctionDef)
    dst_cst = autoquantize_funcdef(src_cst, optable)

    assert isinstance(dst_cst, libcst.FunctionDef)
    required_imports = _infer_imports(src, dst_cst)

    return QuantizedFunctionBuilder(dst_cst, required_imports, origin=func_ref)


def autoquantize_funcdef(src_cst: libcst.FunctionDef, optable: OperatorTable) -> libcst.FunctionDef:
    """Autoquantize a single `FuncDef` with given `optable`."""
    pm = PassManager(
        passes=[
            QuantizedCounterpartReplacer(optable=optable),
            MetadataTransformer(QuantizerFunctionTransformer(), wrap_in_module=True),
        ]
    )
    return pm(src_cst)


def _infer_imports(src: PySource, cst: libcst.FunctionDef) -> set[ImportSymbol]:
    if (scope := src.scope()) is None:
        # Cannot infer import if no scope information is available.
        # Generally, this should not happen.
        # Do not error here to ensure autoquant does not completely fail
        # in this case.
        return set()
    return find_required_imports(cst, scope, src.module().qualified_name)
