# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


from typing import cast

import libcst

from fastforward._autoquant.cst.passes import QuantizedCounterpartReplacer
from fastforward._autoquant.pysource.scope import ImportSymbol, find_required_imports
from fastforward._quantops import OperatorTable

from .cst.quantizer_analysis import introduce_input_quantizers
from .pybuilder import FunctionBuilder, QuantizedFunctionBuilder
from .pysource import PySource


def convert_method(src: PySource, optable: OperatorTable) -> FunctionBuilder:
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
    dst_cst = _add_input_quantizers(src_cst)

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


def _add_input_quantizers(cst: libcst.FunctionDef) -> libcst.FunctionDef:
    return introduce_input_quantizers(cst)


def _infer_imports(src: PySource, cst: libcst.FunctionDef) -> set[ImportSymbol]:
    if (scope := src.scope()) is None:
        # Cannot infer import if no scope information is available.
        # Generally, this should not happen.
        # Do not error here to ensure autoquant does not completely fail
        # in this case.
        return set()
    return find_required_imports(cst, scope, src.module().qualified_name)
