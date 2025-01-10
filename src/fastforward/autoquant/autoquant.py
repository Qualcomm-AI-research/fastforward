# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import libcst
import libcst.display
import libcst.matchers
import torch

from fastforward._import import fully_qualified_name
from fastforward._quantops import optable

from . import pybuilder, pysource
from .convert import convert_method
from .cst import passes


def default_source_context() -> pysource.SourceContext:
    return pysource.SourceContext(
        preprocessing_passes=[
            passes.ConvertSemicolonJoinedStatements(),
            passes.MarkReplacementCandidates(),
            passes.IsolateReplacementCandidates(),
            passes.WrapAssignments(),
        ]
    )


def autoquant(module: torch.nn.Module, operator_table: optable.OperatorTable | None = None) -> None:
    operator_table = operator_table or optable.OperatorTable.from_yaml(
        alias_extensions=optable.STR_ALIASES_EXTENSIONS
    )
    _autoquant(module, default_source_context(), operator_table)


def _autoquant(
    module: torch.nn.Module,
    source_context: pysource.SourceContext,
    operator_table: optable.OperatorTable,
) -> None:
    ModuleType = type(module)
    src_class = source_context.get(fully_qualified_name(ModuleType))

    dst_class = pybuilder.ClassBuilder(
        f"Quantized{ModuleType.__name__}",
        bases=("fastforward.nn.QuantizedModule", ModuleType.__name__),
    )

    forward_src = src_class.member("forward")
    dst_class.add_method(convert_method(forward_src, dst_class, operator_table))

    print(libcst.Module([dst_class.build()]).code)
