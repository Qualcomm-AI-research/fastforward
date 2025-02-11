# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.


import libcst
import libcst.display
import libcst.matchers
import torch

from fastforward._import import fully_qualified_name
from fastforward._quantops import optable
from fastforward.autoquant.pybuilder.builder import InitQuantizationMethod

from . import pybuilder, pysource
from .convert import convert_method
from .cst import passes


def default_source_context() -> pysource.SourceContext:
    """Default source context for Autoquant.

    If no source context is provided, this context is used.
    """
    return pysource.SourceContext(
        preprocessing_passes=[
            passes.ConvertSemicolonJoinedStatements(),
            passes.MarkReplacementCandidates(),
            passes.IsolateReplacementCandidates(),
            passes.WrapAssignments(),
        ]
    )


def autoquant(module: torch.nn.Module, operator_table: optable.OperatorTable | None = None) -> None:
    """Create Python source code for quantized version of `module`.

    Note:
        This functionality is experimental and currently under active
        development.

    Args:
        module: The module to quantize.
        operator_table: The operator table that defines the non-quantized to
            quantized operator mapping.

    """
    operator_table = operator_table or optable.OperatorTable.from_yaml(
        alias_extensions=optable.STR_ALIASES_EXTENSIONS
    )
    print(_autoquant(module, default_source_context(), operator_table))


def _autoquant(
    module: torch.nn.Module,
    source_context: pysource.SourceContext,
    operator_table: optable.OperatorTable,
) -> str:
    ModuleType = type(module)
    src_class = source_context.get(fully_qualified_name(ModuleType))

    dst_class = pybuilder.ClassBuilder(
        f"Quantized{ModuleType.__name__}",
        bases=("fastforward.nn.QuantizedModule", ModuleType.__name__),
    )

    forward_src = src_class.member("forward")
    quantized_forward, quantizer_collection = convert_method(forward_src, dst_class, operator_table)

    dst_class.add_method(InitQuantizationMethod(quantizer_collection))
    dst_class.add_method(quantized_forward)

    return libcst.Module([dst_class.build()]).code
