# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


from collections.abc import Iterator

import torch

from fastforward._import import fully_qualified_name
from fastforward._quantops import optable
from fastforward.nn.quantized_module import QuantizedModule

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


def autoquant_with_defaults(
    module: torch.nn.Module, operator_table: optable.OperatorTable | None = None
) -> pybuilder.ModuleBuilder:
    operator_table = operator_table or default_optable()
    return autoquant(module, default_source_context(), operator_table)


def codeformat_with_defaults(
    code: str, code_formatter: pybuilder.CodeFormatter | None = None
) -> str:
    code_formatter = code_formatter or pybuilder.RuffFormatter()
    return code_formatter.format(code)


def autoquant(
    module: torch.nn.Module,
    source_context: pysource.SourceContext,
    operator_table: optable.OperatorTable,
) -> pybuilder.ModuleBuilder:
    pre_quantized_modules: set[type[torch.nn.Module]] = _find_known_quantized_modules()
    dst_module = pybuilder.ModuleBuilder()

    for mod in _find_unquantized_submodules(module, pre_quantized_modules):
        mod_type = type(mod)
        src_class = source_context.get(fully_qualified_name(mod_type))
        dst_class = pybuilder.QuantizedModuleBuilder(
            f"Quantized{mod_type.__name__}",
            bases=(mod_type.__name__,),
        )

        forward_src = src_class.member("forward")
        quantized_forward = convert_method(forward_src, dst_class, operator_table)

        dst_class.add_method(quantized_forward)
        dst_module.add_class(dst_class)

    return dst_module


def _find_unquantized_submodules(
    torch_module: torch.nn.Module, pre_quantized_modules: set[type[torch.nn.Module]]
) -> Iterator[torch.nn.Module]:
    """Yield submodules of `torch_module` that are not quantized yet.

    Multiple instances of a module type that is not quantized may be part of
    `torch_module` in this case, only the first occurrence is yielded from this
    function. Any submodule whose type is a member of `pre_quantized_modules`
    is considered quantized and is not yielded.
    """
    discovered_modules = set(pre_quantized_modules)
    for module in torch_module.modules():
        module_type = type(module)
        if module_type not in discovered_modules:
            discovered_modules.add(module_type)
            yield module


def _find_known_quantized_modules() -> set[type[torch.nn.Module]]:
    """Find the modules that are manually quantized in FastForward."""
    subclasses = QuantizedModule.__subclasses__()
    immediate_superclasses: set[type[torch.nn.Module]] = set()
    for cls in subclasses:
        for base in cls.__bases__:
            if not issubclass(base, QuantizedModule):
                assert issubclass(base, torch.nn.Module), f"Expected a torch.nn.Module, got: {base}"
                immediate_superclasses.add(base)

    return immediate_superclasses
