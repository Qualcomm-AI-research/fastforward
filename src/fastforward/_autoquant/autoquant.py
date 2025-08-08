# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


import collections
import pathlib

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from types import ModuleType

import libcst
import torch

import fastforward as ff
import fastforward._autoquant.cst.nodes as nodes

from fastforward._autoquant.cst.filter import filter_nodes_by_type
from fastforward._autoquant.pysource.scope import ImportSymbol
from fastforward._import import fully_qualified_name
from fastforward._quantops import optable
from fastforward.nn.quantized_module import QuantizedModule

from . import pybuilder, pysource
from .convert import convert_funcdef
from .cst import passes


def default_source_context(use_type_inference: bool = True) -> pysource.SourceContext:
    """Default source context for Autoquant.

    If no source context is provided, this context is used.
    """
    return pysource.SourceContext(
        preprocessing_passes=default_preprocessing_passes(use_type_inferece=use_type_inference)
    )


def default_preprocessing_passes(use_type_inferece: bool = True) -> Sequence[libcst.CSTTransformer]:
    MarkReplacementCandidatesPass = (
        passes.ExtendedMarkReplacementCandidates()
        if use_type_inferece
        else passes.MarkReplacementCandidates()
    )
    return [
        passes.ConvertSemicolonJoinedStatements(),
        MarkReplacementCandidatesPass,
        passes.IsolateReplacementCandidates(),
        passes.WrapAssignments(),
    ]


def default_optable() -> optable.OperatorTable:
    """Default operator table for autoquant.

    If no operator table is provided this table is used.
    """
    return optable.OperatorTable.from_yaml(alias_extensions=optable.STR_ALIASES_EXTENSIONS)


def autoquant_with_defaults(
    module: torch.nn.Module,
    operator_table: optable.OperatorTable | None = None,
    use_type_inference: bool = True,
) -> str:
    return autoquant(
        module=module,
        source_context=default_source_context(use_type_inference=use_type_inference),
        operator_table=operator_table or default_optable(),
    )


def codeformat_with_defaults(
    code: str, code_formatter: pybuilder.CodeFormatter | None = None
) -> str:
    code_formatter = code_formatter or pybuilder.RuffFormatter()
    return code_formatter.format(code)


def emit_code_of_module(
    module: str,
    output_path: pathlib.Path | str | None,
    code_writer: pybuilder.BasicCodeWriter | None,
    force_overwrite: bool,
) -> str:
    """Emits code via a CodeWriter."""
    if (output_path is None) + (code_writer is None) != 1:
        raise ValueError("Specify exactly one of `output_path` and `code_writer`.")

    if code_writer is not None and force_overwrite:
        raise ValueError(
            "Cannot force overwrite when using a CodeWriter. "
            + "Instead, pass it as argument to the `CodeWriter`."
        )
    if output_path is not None:
        code_writer = pybuilder.FileWriter(
            output_path=pathlib.Path(output_path), force_overwrite=force_overwrite
        )
    assert code_writer is not None
    code_writer.write(module)
    return code_writer.module_name


def autoquant(
    module: torch.nn.Module,
    source_context: pysource.SourceContext,
    operator_table: optable.OperatorTable,
) -> str:
    """Autoquantizes a `torch.nn.Module`."""
    pre_quantized_modules = _find_known_quantized_modules()
    dst_module = pybuilder.ModuleBuilder(origin=type(module))

    for mod in _find_unquantized_submodules(module, pre_quantized_modules):
        dst_module.add_class(_autoquantize_pytorch_module(mod, source_context, operator_table))

    return dst_module.build().code


def _autoquantize_pytorch_module(
    module: torch.nn.Module,
    source_context: pysource.SourceContext,
    operator_table: optable.OperatorTable,
) -> pybuilder.ClassBuilder:
    mod_type = type(module)
    qualified_class_name = fully_qualified_name(mod_type)
    src_class = source_context.get(qualified_class_name)

    base_module_name, base_class_name = qualified_class_name.rsplit(".", 1)

    dst_class = pybuilder.QuantizedModuleBuilder(
        f"Quantized{mod_type.__name__}",
        bases=(mod_type.__name__,),
        required_imports=(ImportSymbol(name=base_class_name, module=base_module_name),),
        origin=mod_type,
    )
    method_queue = collections.deque[str](["forward"])

    while method_queue:
        func_name = method_queue.popleft()
        if dst_class.has_method(func_name):
            continue

        method_src = src_class.member(func_name)
        method_ref = getattr(mod_type, func_name, None)
        method_queue.extend(_find_dependent_methods(method_src, mod_type))
        quantized_forward = convert_funcdef(
            src=method_src,
            optable=operator_table,
            func_ref=method_ref,
        )
        dst_class.add_method(quantized_forward)

    return dst_class


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


MethodType = ff.type_common.MethodType


def _find_dependent_methods(
    func_src: pysource.PySource, ModuleType: type[torch.nn.Module]
) -> Iterator[str]:
    """Find methods that are called by the given function within the same module.

    This function analyzes the source code of a method to identify other methods of the same
    class that are called within it.

    Args:
        func_src: The source code representation of the function to analyze.
        ModuleType: The PyTorch module class that contains the function.

    Returns:
        An iterator of method names that are called by the given function.

    Raises:
        ValueError: If the function is expected to have a 'self' parameter but doesn't.
    """
    funcdef = func_src.cst(NodeType=libcst.FunctionDef)
    func_name = funcdef.name.value
    self_name = None

    method_type = ff.type_common.method_type(ModuleType, func_name)
    if method_type in (MethodType.CLASS_METHOD, MethodType.METHOD):
        try:
            self_name = funcdef.params.params[0].name.value
        except IndexError:
            msg = f"Expected function '{funcdef.name.value}' to have at least one parameter"
            raise ValueError(msg)

    for candidate in filter_nodes_by_type(funcdef, nodes.ReplacementCandidate):
        if not isinstance(call_expr := candidate.original, libcst.Call):
            continue

        match call_expr.func:
            case libcst.Attribute(value=libcst.Name(obj), attr=libcst.Name(attr)):
                if obj != self_name and obj != ModuleType.__name__:
                    continue

                # Only instance methods are currently supported. Class and
                # static methods require additional analysis and will be added
                # in the future. See GitHub issue #359 for tracking progress on
                # this feature.
                if ff.type_common.method_type(ModuleType, attr) is MethodType.METHOD:
                    yield attr


def _all_subclasses(cls: type[torch.nn.Module]) -> set[type[torch.nn.Module]]:
    return set(cls.__subclasses__()).union([
        c for subcls in cls.__subclasses__() for c in _all_subclasses(subcls)
    ])


def _find_known_quantized_modules() -> set[type[torch.nn.Module]]:
    """Find the modules that are manually quantized in FastForward."""
    subclasses = _all_subclasses(QuantizedModule)
    immediate_superclasses: set[type[torch.nn.Module]] = set()
    for cls in subclasses:
        for base in cls.__bases__:
            if not issubclass(base, QuantizedModule):
                assert issubclass(base, torch.nn.Module), f"Expected a torch.nn.Module, got: {base}"
                immediate_superclasses.add(base)

    return immediate_superclasses


@dataclass
class AutoQuantizedCode:
    """Contains the generated code and the corresponding Python module."""

    code: str
    pymodule: ModuleType | None
    pymodule_name: str
