# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


from typing import cast

import libcst
import libcst.display

from fastforward._quantops import OperatorTable
from fastforward.autoquant.cst.passes import QuantizedCounterpartReplacer

from .cfg import construct, reconstruct
from .pybuilder import FunctionBuilder, QuantizedModuleBuilder
from .pysource import PySource


def convert_method(
    src: PySource, clsbuilder: QuantizedModuleBuilder, optable: OperatorTable
) -> FunctionBuilder:
    """Convert existing function."""
    src_cst = src.cst(NodeType=libcst.FunctionDef)

    src_cst = _rewrite_quantized_operators(src_cst, clsbuilder, optable)

    cfg = construct(src_cst)
    dst_cst = reconstruct(cfg)

    assert isinstance(dst_cst, libcst.FunctionDef)

    return FunctionBuilder(dst_cst)


def _rewrite_quantized_operators(
    cst: libcst.FunctionDef, clsbuilder: QuantizedModuleBuilder, optable: OperatorTable
) -> libcst.FunctionDef:
    function_replacement = QuantizedCounterpartReplacer(optable=optable)
    new_cst = cast(libcst.FunctionDef, cst.visit(function_replacement))

    for var in function_replacement.get_quantized_vars():
        clsbuilder.add_quantizer(var)

    return new_cst
