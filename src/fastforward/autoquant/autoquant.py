# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


import torch

from fastforward._import import fully_qualified_name
from fastforward._quantops import optable

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


def default_optable() -> optable.OperatorTable:
    """Default operator table for autoquant.

    If no operator table is provided this table is used.
    """
    return optable.OperatorTable.from_yaml(alias_extensions=optable.STR_ALIASES_EXTENSIONS)


def autoquantize(
    module: torch.nn.Module, operator_table: optable.OperatorTable | None = None
) -> None:
    """Create Python source code for quantized version of `module`.

    Note:
        This functionality is experimental and currently under active
        development.

    Args:
        module: The module to quantize.
        operator_table: The operator table that defines the non-quantized to
            quantized operator mapping.

    """
    print(_autoquant_with_defaults(module, operator_table).build().code)


def _autoquant_with_defaults(
    module: torch.nn.Module, operator_table: optable.OperatorTable | None = None
) -> pybuilder.ModuleBuilder:
    operator_table = operator_table or default_optable()
    return _autoquant(module, default_source_context(), operator_table)


def _autoquant(
    module: torch.nn.Module,
    source_context: pysource.SourceContext,
    operator_table: optable.OperatorTable,
) -> pybuilder.ModuleBuilder:
    ModuleType = type(module)
    src_class = source_context.get(fully_qualified_name(ModuleType))

    dst_class = pybuilder.QuantizedModuleBuilder(
        f"Quantized{ModuleType.__name__}",
        bases=(ModuleType.__name__,),
    )

    dst_module = pybuilder.ModuleBuilder()

    forward_src = src_class.member("forward")
    quantized_forward = convert_method(forward_src, dst_class, operator_table)

    dst_class.add_method(quantized_forward)
    dst_module.add_class(dst_class)

    return dst_module
