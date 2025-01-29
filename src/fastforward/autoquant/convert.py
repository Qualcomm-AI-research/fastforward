# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import libcst
import libcst.display

from fastforward._quantops import OperatorTable

from .pybuilder import ClassBuilder, FunctionBuilder
from .pysource import PySource


def convert_method(
    src: PySource, clsbuilder: ClassBuilder, optable: OperatorTable
) -> FunctionBuilder:
    """
    Convert existing function.
    """
    src_cst = src.cst(NodeType=libcst.FunctionDef)

    dst_cst = src_cst  # Conversion is work in progress
    assert isinstance(dst_cst, libcst.FunctionDef)

    return FunctionBuilder(dst_cst)
