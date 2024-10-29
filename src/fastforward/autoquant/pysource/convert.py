import libcst
import libcst.display

from fastforward._quantops import OperatorTable

from ..cst import passes
from .builder import ClassBuilder, FunctionBuilder
from .node import PySourceNode


def convert_method(
    src_node: PySourceNode, clsbuilder: ClassBuilder, optable: OperatorTable
) -> FunctionBuilder:
    """
    Convert existing function.
    """
    if not isinstance(src_node.cst, libcst.FunctionDef):
        raise TypeError("Can only convert libcst.FunctionDef nodes to quantized methods")

    src_cst = src_node.cst.visit(passes.CandidateRewriter(optable))

    assert isinstance(src_cst, libcst.FunctionDef)

    dst_cst = src_cst  # Conversion is work in progress
    assert isinstance(dst_cst, libcst.FunctionDef)

    return FunctionBuilder(dst_cst)
