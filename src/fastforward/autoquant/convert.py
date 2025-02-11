# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.


import libcst
import libcst.display

from fastforward._quantops import OperatorTable
from fastforward.autoquant.cst.passes import QuantizedCounterpartReplacer

from .pybuilder import ClassBuilder, FunctionBuilder
from .pysource import PySource


def convert_method(
    src: PySource, clsbuilder: ClassBuilder, optable: OperatorTable
) -> tuple[FunctionBuilder, tuple[str, ...]]:
    """Convert existing function.
    """
    src_cst = src.cst(NodeType=libcst.FunctionDef)
    function_replacement = QuantizedCounterpartReplacer(optable=optable)

    dst_cst = src_cst.visit(function_replacement)
    assert isinstance(dst_cst, libcst.FunctionDef)

    return FunctionBuilder(dst_cst), function_replacement.get_quantized_vars()
