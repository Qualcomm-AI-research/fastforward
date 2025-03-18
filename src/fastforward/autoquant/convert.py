# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


from typing import cast

import libcst
import libcst.display

from fastforward._quantops import OperatorTable
from fastforward.autoquant.cst.passes import QuantizedCounterpartReplacer

from .cfg import construct, reconstruct
from .pybuilder import ClassBuilder, FunctionBuilder
from .pysource import PySource


def convert_method(
    src: PySource, clsbuilder: ClassBuilder, optable: OperatorTable
) -> tuple[FunctionBuilder, tuple[str, ...]]:
    """Convert existing function."""
    src_cst = src.cst(NodeType=libcst.FunctionDef)
    function_replacement = QuantizedCounterpartReplacer(optable=optable)

    src_cst = cast(libcst.FunctionDef, src_cst.visit(function_replacement))

    cfg = construct(src_cst)
    dst_cst = reconstruct(cfg)

    assert isinstance(dst_cst, libcst.FunctionDef)

    return FunctionBuilder(dst_cst), function_replacement.get_quantized_vars()
