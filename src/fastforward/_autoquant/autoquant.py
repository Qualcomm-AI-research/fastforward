# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


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


def autoquant(
    module: torch.nn.Module,
    source_context: pysource.SourceContext,
    operator_table: optable.OperatorTable,
) -> pybuilder.ModuleBuilder:
    module_queue: list[torch.nn.Module] = [module]
    seen_modules: set[type[torch.nn.Module]] = _find_known_quantized_modules()
    dst_module = pybuilder.ModuleBuilder()

    while len(module_queue) > 0:
        current = module_queue.pop()
        current_cls = type(current)

        seen_modules.add(current_cls)
        unseen_submodules = _find_unseen_submodules(current, seen_modules)

        module_queue.extend(unseen_submodules)

        src_class = source_context.get(fully_qualified_name(current_cls))
        dst_class = pybuilder.QuantizedModuleBuilder(
            f"Quantized{current_cls.__name__}",
            bases=(current_cls.__name__,),
        )

        forward_src = src_class.member("forward")
        quantized_forward = convert_method(forward_src, dst_class, operator_table)

        dst_class.add_method(quantized_forward)
        dst_module.add_class(dst_class)

    return dst_module


def _find_unseen_submodules(
    torch_module: torch.nn.Module, seen_modules: set[type[torch.nn.Module]]
) -> tuple[torch.nn.Module, ...]:
    unseen_submodules = tuple(
        module for module in torch_module.modules() if type(module) not in seen_modules
    )
    return unseen_submodules


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
