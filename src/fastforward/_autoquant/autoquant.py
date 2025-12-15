# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


import collections
import contextlib
import dataclasses
import inspect
import itertools
import logging
import pathlib
import sys
import types

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Callable, Iterable, TypeAlias

import libcst
import torch

import fastforward as ff
import fastforward._autoquant.cst.nodes as nodes

from fastforward._autoquant import pybuilder, pysource
from fastforward._autoquant.convert import convert_function
from fastforward._autoquant.cst import node_creation, node_processing, passes
from fastforward._autoquant.cst.filter import filter_nodes_by_type
from fastforward._autoquant.cst.pattern import PatternRule, _PatternRuleTransformer
from fastforward._autoquant.function_context import FunctionContext
from fastforward._autoquant.pybuilder import QuantizerReferenceCollection
from fastforward._autoquant.pysource.scope import ImportSymbol
from fastforward._import import fully_qualified_name
from fastforward._quantops import optable
from fastforward.nn.quantized_module import QuantizedModule

_FuncRef: TypeAlias = Callable[..., Any]
MethodType = ff.type_common.MethodType

logger = logging.getLogger(__name__)


def autoquant(
    module: torch.nn.Module,
    source_context: pysource.SourceContext,
    operator_table: optable.OperatorTable,
) -> str:
    """Autoquantizes a torch.nn.Module and its submodules by quantizing all methods.

    Args:
        module: The PyTorch module to quantize
        source_context: Source code context for accessing module definitions
        operator_table: Table of quantization operators to use

    Returns:
        Generated Python code for the quantized module
    """
    quantizer_refs = QuantizerReferenceCollection()
    module_builder = pybuilder.ModuleBuilder(origin=type(module))
    class_builders: dict[type, pybuilder.QuantizedModuleBuilder] = {}

    # Skip modules that are already quantized
    pre_quantized_modules = _find_known_quantized_modules()
    for mod in _find_unquantized_submodules(module, pre_quantized_modules):
        mod_type = type(mod)
        class_builders[mod_type] = _cls_builder_for_module(mod_type)

    # Queue all forward methods for processing
    func_queue = collections.deque[_AqTask]()
    for mod in _find_unquantized_submodules(module, pre_quantized_modules):
        mod_type = type(mod)
        func_queue.append(_AqTask(module=mod_type, function=mod_type.forward))

    # Process each method and its dependencies
    while func_queue:
        task = func_queue.popleft()

        func_name = task.alias or task.function.__name__
        if any(b.origin.func is task.function for b in module_builder.functions()):
            continue

        if isinstance(task.function, type):
            logger.warning(
                "Skipping '%s' because it is a class and class constructors are not supported. "
                + "This may require further manual conversion for correct quantization.",
                fully_qualified_name(task.function),
            )
        elif inspect.ismodule(task.module):
            # If task.module is a Python module (in contrast to a PyTorch
            # module), treat the function as a 'helper' function. Create a new
            # quantized version of the function in the quantized (Python)
            # module.

            qualified_module_name = fully_qualified_name(task.module)
            func_ctx = FunctionContext.from_function_reference(task.function, task.module)
            module_src = source_context.get(qualified_module_name)
            func_src = module_src.member(func_name)
            with quantizer_refs.push_context(func_ctx):
                func_builder = convert_function(
                    src=func_src,
                    optable=operator_table,
                    func_ctx=func_ctx,
                    quantizer_refs=quantizer_refs,
                )
            if task.alias is not None:
                func_builder.name = task.alias
            module_builder.add_function(func_builder)

            # Queue dependent functions for processing
            for new_task in _find_dependent_functions(func_src, func_ctx):
                func_queue.append(new_task)

        elif issubclass(task.module, torch.nn.Module):
            # If task.module is a PyTorch module (in contrast to a Python module), create
            # a new member function on the quantized module. This can be an instance, class, or
            # static method.
            if task.module not in class_builders:
                class_builders[task.module] = _cls_builder_for_module(task.module)
            cls_builder = class_builders[task.module]
            if cls_builder.has_method(func_name):
                continue

            # Convert method to quantized version
            qualified_class_name = fully_qualified_name(task.module)
            src_class = source_context.get(qualified_class_name)
            method_src = src_class.member(func_name)
            method_ctx = FunctionContext.from_method(task.module, func_name)
            with quantizer_refs.push_context(method_ctx):
                func_builder = convert_function(
                    src=method_src,
                    optable=operator_table,
                    func_ctx=method_ctx,
                    quantizer_refs=quantizer_refs,
                )
                cls_builder.add_method(func_builder)

            # Queue dependent methods for processing
            for new_task in itertools.chain(
                _find_dependent_methods(method_src, method_ctx),
                _find_dependent_functions(method_src, method_ctx),
            ):
                if new_task.function in operator_table:
                    # If function is in operator_table, it will be converted
                    # directly and no further analysis is required.
                    continue
                func_queue.append(new_task)

        else:
            msg = (  # type: ignore[unreachable]
                f"Failed to quantize '{task.function.__name__}' of '{task.module}' because "
                + f"'{task.module}' is not a Python or Pytorch module."
            )
            logger.warning(msg)

    for class_builder in class_builders.values():
        module_builder.add_class(class_builder)

    _resolve_all_quantized_calls(module_builder, quantizer_refs)

    return module_builder.build(quantizer_refs).code


@dataclasses.dataclass
class _AqTask:
    """Autoquant task element for task queue."""

    function: Callable[..., Any]
    module: type[torch.nn.Module] | types.ModuleType
    alias: str | None = None


def _cls_builder_for_module(module_type: type[torch.nn.Module]) -> pybuilder.QuantizedModuleBuilder:
    qualified_class_name = fully_qualified_name(module_type)
    base_module_name, base_class_name = qualified_class_name.rsplit(".", 1)
    return pybuilder.QuantizedModuleBuilder(
        f"Quantized{module_type.__name__}",
        bases=(module_type.__name__,),
        required_imports=(ImportSymbol(name=base_class_name, module=base_module_name),),
        origin=module_type,
    )


def default_source_context(
    use_type_inference: bool = True, replacement_patterns: Iterable[PatternRule] = ()
) -> pysource.SourceContext:
    """Default source context for Autoquant.

    If no source context is provided, this context is used.
    """
    passes = default_preprocessing_passes(use_type_inference=use_type_inference)
    patterns = tuple(replacement_patterns)
    if len(patterns) > 0:
        # Ensure that patterns are applied first, this way the patterns are matched
        # against the actual input.
        passes = (_PatternRuleTransformer(patterns),) + tuple(passes)
    return pysource.SourceContext(preprocessing_passes=passes)


def default_preprocessing_passes(
    use_type_inference: bool = True,
) -> Sequence[libcst.CSTTransformer | type[libcst.CSTTransformer]]:
    MarkReplacementCandidatesPass = (
        passes.ExtendedMarkReplacementCandidates()
        if use_type_inference
        else passes.MarkReplacementCandidates()
    )
    return [
        passes.ConvertSemicolonJoinedStatements(),
        MarkReplacementCandidatesPass,
        passes.IsolateReplacementCandidates,
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
    replacement_patterns: Iterable[PatternRule] = (),
) -> str:
    return autoquant(
        module=module,
        source_context=default_source_context(
            use_type_inference=use_type_inference,
            replacement_patterns=replacement_patterns,
        ),
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
    subclasses = _all_subclasses(QuantizedModule)
    immediate_superclasses: set[type[torch.nn.Module]] = set()
    for cls in subclasses:
        for base in cls.__bases__:
            if not issubclass(base, QuantizedModule):
                assert issubclass(base, torch.nn.Module), f"Expected a torch.nn.Module, got: {base}"
                immediate_superclasses.add(base)

    return immediate_superclasses


def _all_subclasses(cls: type[torch.nn.Module]) -> set[type[torch.nn.Module]]:
    """Used in _find_known_quantized_modules."""
    return set(cls.__subclasses__()).union([
        c for subcls in cls.__subclasses__() for c in _all_subclasses(subcls)
    ])


@dataclasses.dataclass(frozen=True)
class _QuantizerRefTrace:
    """Tracks the propagation path of a quantizer reference through function calls.

    This class represents a quantizer reference along with its source path, which is used
    during the quantizer propagation phase to track how quantizers flow through the call
    graph and to prevent circular dependencies.

    Attributes:
        src: Tuple of function references representing the propagation path of the quantizer.
             The first element is the original function where the quantizer was defined,
             and subsequent elements represent the call chain through which it propagated.
             For example, (func_a, func_b, func_c) means the quantizer originated in func_a,
             was passed to func_b, and then to func_c.
        ref: The quantizer reference object that is being tracked through the call graph.
    """

    src: tuple[_FuncRef, ...]
    ref: nodes.QuantizerReference


@dataclasses.dataclass
class _QuantizedFunctionSpec:
    """Specification for a quantized function including its quantizer dependencies and calls.

    This class represents the quantization specification for a single function, tracking:
    1. Local quantizers that are defined within the function
    2. Calls to other quantized functions
    3. The complete set of quantizers needed by this function (local + propagated)

    The class is used during the quantizer propagation phase to build a dependency graph
    of quantizer usage across function boundaries and determine which quantizer arguments
    need to be passed between functions.

    Attributes:
        func_ref: Reference to the function this specification describes
        calls: Mapping of unresolved quantized calls to their required quantizer arguments.
               Each call maps to a list of quantizer traces that represent the quantizers
               that need to be passed as arguments to that call.
        local_quantizers: List of quantizers that are locally defined within this function.
                         Each quantizer is wrapped in a trace that tracks its source path.
    """

    func_ref: _FuncRef
    calls: dict[nodes.UnresolvedQuantizedCall, list[_QuantizerRefTrace]] = dataclasses.field(
        default_factory=dict
    )
    local_quantizers: list[_QuantizerRefTrace] = dataclasses.field(default_factory=list)

    def quantizers(self, skip_forwarded: bool = False) -> Iterator[_QuantizerRefTrace]:
        """Iterate over all quantizers needed by this function.

        Args:
            skip_forwarded: If True, skip quantizers that passed to other functions and not
                used locally within this function.
        """
        yield from self.local_quantizers
        for quant_args in self.calls.values():
            for arg in quant_args:
                if not skip_forwarded or len(arg.src) > 1:
                    yield arg


@dataclasses.dataclass
class _CallArg:
    """Represents a keyword argument for a quantized function call.

    This class encapsulates the mapping between a quantizer reference (used as the
    parameter name) and its corresponding value expression (the actual argument to pass).
    It is used during the call resolution phase to construct concrete function calls
    with the appropriate quantizer arguments.
    """

    keyword: nodes.QuantizerReference
    value: libcst.BaseExpression


def _resolve_all_quantized_calls(
    builder: pybuilder.ModuleBuilder, quantizer_refs: QuantizerReferenceCollection
) -> None:
    """Resolve quantized function calls.

    Resolve remaining unresolved quantized function calls by creating a complete mapping of
    quantizer arguments across function boundaries.
    """
    # The algorithm builds a dependency graph of quantizer usage across functions and then
    # "threads" the quantizer arguments through the call chain.
    # See the inline annotations for more detailed explanation.

    # 1. Discovery Phase: Collect all quantized functions and identify their local quantizers
    #    and unresolved calls to other quantized functions
    func_builder_map, func_contexts, func_specs = _discover_quantized_functions(
        builder, quantizer_refs
    )

    # 2. Propagation Phase: Use _propagate_quantizers() to determine which quantizer arguments
    #    need to flow between functions based on the call graph and propagate quantizers across
    #    function call boundaries.
    _propagate_quantizers(func_specs)

    # 3. Signature Generation Phase: For each function, create a signature that includes all
    #    quantizers it needs (both local and propagated from callers), generating unique
    #    parameter names for propagated quantizers using prefixed naming.
    signatures, signature_ref_map = _generate_signatures(quantizer_refs, func_contexts, func_specs)

    # 4. Call Resolution Phase: For each unresolved call, determine the concrete arguments
    #    to pass by mapping the caller's quantizer references to the callee's expected
    #    parameters (regular functions use signature references, instance methods use
    #    quantizer expressions)
    #
    calls: dict[nodes.UnresolvedQuantizedCall, tuple[_CallArg, ...]] = {}
    _resolve_calls(quantizer_refs, func_contexts, func_specs, signatures, signature_ref_map, calls)

    # 5. Cleanup phase: remove functions that were initially identified as
    #    quantization candidates but have empty quantization signatures,
    #    indicating they do not require any quantization operations and should
    #    be excluded from the generated module.
    for func, signature in list(signatures.items()):
        if not signature:
            builder.remove_function(func_builder_map[func])
            del func_builder_map[func]
            del func_specs[func]
            del signatures[func]

    # 6. Resolve function name phase: Rename helper functions to avoid naming
    #    conflicts For functions that are not methods (NO_METHOD type), convert
    #    their fully qualified names into valid Python identifiers prefixed
    #    with "quantized_" and update both the function definition and the
    #    mapping for later reference updates.
    #    For example, `mod.submod.helper`` becomes `quantized_mod_submod_helper`.
    helper_function_names: dict[_FuncRef, libcst.BaseExpression] = {}
    used_helper_names: dict[str, int] = {}
    for func_ref, func_builder in func_builder_map.items():
        if func_builder.origin.method_type is MethodType.NO_METHOD:
            new_name = f"quantized_{func_builder.name}"
            used_helper_names[new_name] = used_helper_names.get(new_name, 0) + 1
            if used_helper_names[new_name] > 1:
                new_name = f"{new_name}_{used_helper_names[new_name]}"
            func_builder.cst = func_builder.cst.with_changes(name=libcst.Name(new_name))
            helper_function_names[func_ref] = libcst.Name(new_name)

    # 7. Transformation Phase: Apply a CST transformer to replace all unresolved calls
    #    with concrete function calls containing the resolved quantizer arguments.
    #    Update the function builder `quantizer_signature` list based on the inferred
    #    signature.
    call_transformer = _ResolveQuantizedCallsTransformer(calls, helper_function_names)
    for func_ref, func_builder in func_builder_map.items():
        new_funcdef = func_builder.cst.visit(call_transformer)
        assert isinstance(new_funcdef, libcst.FunctionDef)
        func_builder.cst = new_funcdef
        func_builder.quantizer_signature = signatures.get(func_ref, ())


def _discover_quantized_functions(
    builder: pybuilder.ModuleBuilder, quantizer_refs: QuantizerReferenceCollection
) -> tuple[
    dict[_FuncRef, pybuilder.QuantizedFunctionBuilder],
    dict[_FuncRef, FunctionContext],
    dict[_FuncRef, _QuantizedFunctionSpec],
]:
    """Discover and collect information about quantized functions in the module.

    Args:
        builder: The module builder containing quantized functions
        quantizer_refs: Collection of quantizer references used across functions

    Returns:
        A tuple containing three dictionaries:
        - Mapping from function references to their quantized function builders
        - Mapping from function references to their function contexts
        - Mapping from function references to their quantized function specifications
          that track local quantizers and unresolved calls
    """
    # Mappings of function references to their builders, contexts, and specs
    func_builder_map: dict[_FuncRef, pybuilder.QuantizedFunctionBuilder] = {}
    func_contexts: dict[_FuncRef, FunctionContext] = {}
    func_specs: dict[_FuncRef, _QuantizedFunctionSpec] = {}

    for funcbuilder in builder.quantized_functions():
        if funcbuilder.origin.func is None:
            continue

        func_ref = funcbuilder.origin.func
        func_builder_map[func_ref] = funcbuilder
        func_contexts[func_ref] = funcbuilder.origin

    # Initialize function specs with local quantizers and unresolved calls
    for func_ref, funcbuilder in func_builder_map.items():
        spec = _QuantizedFunctionSpec(func_ref)
        func_specs[func_ref] = spec

        # Add local quantizers for this function
        for ref in quantizer_refs.local_quantizers_for_func(func_ref):
            spec.local_quantizers.append(_QuantizerRefTrace(src=(func_ref,), ref=ref))

        # Collect unresolved calls (excluding method calls)
        for unresolved_call in filter_nodes_by_type(funcbuilder.cst, nodes.UnresolvedQuantizedCall):
            if unresolved_call.func_ref not in func_contexts:
                continue
            call_func_context = func_contexts[unresolved_call.func_ref]
            if call_func_context.method_type != MethodType.METHOD:
                spec.calls[unresolved_call] = []

    return func_builder_map, func_contexts, func_specs


def _propagate_quantizers(func_specs: dict[_FuncRef, _QuantizedFunctionSpec]) -> None:
    """Propagate quantizer dependencies through the function call graph.

    This function implements an iterative algorithm that ensures each function in the
    call graph has access to all quantizers it needs, either directly or through
    functions it calls.

    The algorithm works by:
    1. Iterating through all function specifications until no changes occur
    2. For each function, examining all the functions it calls
    3. Collecting quantizers from called functions and adding them as dependencies
    4. Tracking the source path of each quantizer to handle circular dependencies:
       - If a quantizer originates from the current function, truncate the path
       - Otherwise, extend the path to include the current function
    5. Updating call arguments when new quantizer dependencies are discovered

    Circular dependency resolution:
    When function A calls function B, and B (directly or indirectly) calls A,
    the algorithm prevents infinite propagation by recognizing when a quantizer's
    source path would create a cycle and truncating it appropriately.

    Args:
        func_specs: Dictionary mapping function references to their quantized
                   specifications, including local quantizers and function calls

    Side effects:
        Modifies the `calls` attribute of function specifications in-place to
        include all required quantizer arguments for each function call.
    """
    # Iterate until no more quantizers need propagation
    changed = True
    while changed:
        changed = False

        for func_ref, spec in func_specs.items():
            # Check each function call to see if it needs additional quantizer args
            for unresolved_call, call_args in spec.calls.items():
                call_func_ref = unresolved_call.func_ref
                new_call_args = []

                # Collect all quantizers needed by the called function
                for quant_arg in func_specs[call_func_ref].quantizers(skip_forwarded=True):
                    # Break circular dependencies by truncating the source path.
                    # This marks the quantizer as 'forwarded' since it's passed
                    # to another function, causing it to be skipped in subsequent
                    # iterations due to `skip_forwarded=True`, effectively
                    # preventing infinite loops in the dependency graph.
                    if quant_arg.src[0] is func_ref:
                        new_call_args.append(
                            _QuantizerRefTrace(src=quant_arg.src[:1], ref=quant_arg.ref)
                        )
                    else:
                        # Extend source path to track propagation chain
                        new_call_args.append(
                            _QuantizerRefTrace(src=quant_arg.src + (func_ref,), ref=quant_arg.ref)
                        )

                # Update call args if new quantizers were discovered
                if len(new_call_args) != len(call_args):
                    spec.calls[unresolved_call] = new_call_args
                    changed = True


_SignatureRefMap: TypeAlias = dict[
    _FuncRef,
    dict[tuple[nodes.UnresolvedQuantizedCall, nodes.QuantizerReference], nodes.QuantizerReference],
]


def _generate_signatures(
    quantizer_refs: QuantizerReferenceCollection,
    func_contexts: Mapping[_FuncRef, FunctionContext],
    func_specs: Mapping[_FuncRef, _QuantizedFunctionSpec],
) -> tuple[
    dict[_FuncRef, tuple[nodes.QuantizerReference, ...]],
    _SignatureRefMap,
]:
    """Generate signatures for quantized functions based on their quantizer dependencies.

    This function analyzes the quantizer dependencies of each function and creates appropriate
    signatures that include all required quantizer parameters. For local quantizers, it uses
    the original reference. For propagated quantizers from other functions, it creates new
    references with prefixed names to avoid naming conflicts.

    Note:
        Only non-method functions (regular functions, static methods, class methods) receive
        updated signatures, as instance methods handle quantizer propagation differently.

    Args:
        quantizer_refs: Collection of quantizer references used across functions
        func_contexts: Mapping from function references to their function contexts
        func_specs: Mapping from function references to their quantized function specifications

    Returns:
        A tuple containing three dictionaries:
        - Mapping from function references to their quantizer parameter signatures
        - Mapping from function references to dictionaries that map original quantizer
          references to their corresponding signature references
    """
    signatures: dict[_FuncRef, tuple[nodes.QuantizerReference, ...]] = {}
    signature_ref_map: _SignatureRefMap = {}

    for func_ref, spec in func_specs.items():
        if func_contexts[func_ref].method_type == MethodType.METHOD:
            # Instance functions don't require an updated signature.
            continue

        # Create function signature from quantizer arguments
        signature = []
        refs = {}

        for arg in spec.local_quantizers:
            signature.append(arg.ref)

        for unresolved_call, quant_args in spec.calls.items():
            for arg in quant_args:
                assert len(arg.src) > 1  # Assert that arg is not a local quantizer
                with quantizer_refs.push_context(func_contexts[func_ref]):
                    prefix = "_".join(fn.__name__ for fn in reversed(arg.src[:-1]))
                    name = f"{prefix}_{arg.ref.value}"
                    sigref = quantizer_refs.create_reference(name)
                signature.append(sigref)
                refs[(unresolved_call, arg.ref)] = sigref

        signatures[func_ref] = tuple(signature)
        signature_ref_map[func_ref] = refs

    return signatures, signature_ref_map


def _resolve_calls(
    quantizer_refs: QuantizerReferenceCollection,
    func_contexts: Mapping[_FuncRef, FunctionContext],
    func_specs: Mapping[_FuncRef, _QuantizedFunctionSpec],
    signatures: dict[_FuncRef, tuple[nodes.QuantizerReference, ...]],
    signature_ref_map: _SignatureRefMap,
    calls: dict[nodes.UnresolvedQuantizedCall, tuple[_CallArg, ...]],
) -> None:
    """Resolve quantized function calls by mapping quantizer references between functions.

    This function processes each unresolved quantized call and determines the concrete
    arguments that need to be passed based on the quantizer dependencies identified
    during the propagation phase. It handles two cases differently:

    1. For instance methods: Creates quantizer expressions that reference the instance's
       quantizers directly
    2. For regular functions: Uses the signature references to map between caller and
       callee parameter names

    The resolved call arguments are stored in the `calls` dictionary for later use by
    the _ResolveQuantizedCallsTransformer.

    Args:
        quantizer_refs: Collection of quantizer references used across functions
        func_contexts: Mapping from function references to their function contexts
        func_specs: Mapping from function references to their quantized function specs
        signatures: Mapping from function references to their quantizer parameter signatures
        signature_ref_map: Mapping from function references to dictionaries that map
                          original quantizer references to their signature references
        calls: Output dictionary that will be populated with resolved call arguments
              for each unresolved quantized call
    """
    for func_ref, spec in func_specs.items():
        is_instance_method = func_contexts[func_ref].method_type == MethodType.METHOD

        for unresolved_call, args in spec.calls.items():
            call_args = []
            for param, arg in zip(signatures[unresolved_call.func_ref], args):
                if is_instance_method:
                    with quantizer_refs.push_context(func_contexts[func_ref]):
                        prefix = "_".join(fn.__name__ for fn in reversed(arg.src[:-1]))
                        name = f"{prefix}_{arg.ref.value}"
                        value = quantizer_refs.create_quantizer_expression(name)
                else:
                    value = signature_ref_map[func_ref][(unresolved_call, arg.ref)]
                call_args.append(_CallArg(keyword=param, value=value))
            calls[unresolved_call] = tuple(call_args)


class _ResolveQuantizedCallsTransformer(libcst.CSTTransformer):
    """CST transformer that resolves unresolved quantized function calls.

    This transformer replaces UnresolvedQuantizedCall nodes in the CST with concrete
    QuantizedCall nodes that include the appropriate quantizer arguments.

    Arguments:
        call_args: Mapping from UnresolvedQuantizedCall nodes to their resolved
                   quantizer arguments.
    """

    def __init__(
        self,
        call_args: dict[nodes.UnresolvedQuantizedCall, tuple[_CallArg, ...]],
        func_rename_map: dict[_FuncRef, libcst.BaseExpression],
    ):
        self._call_args = call_args
        self._func_rename_map = func_rename_map

    def leave_UnresolvedQuantizedCall(
        self,
        original_node: nodes.UnresolvedQuantizedCall,
        updated_node: nodes.UnresolvedQuantizedCall,
    ) -> nodes.QuantizedCall | nodes.UnresolvedQuantizedCall:
        if (quantizer_args := self._call_args.get(original_node, None)) is None:
            return updated_node

        new_args = []
        for arg in quantizer_args:
            new_args.append(node_creation.get_keyword_argument_node(arg.keyword, arg.value))

        params = nodes.node_asdict(updated_node)
        params["args"] = tuple(updated_node.args) + tuple(new_args)
        if original_node.func_ref in self._func_rename_map:
            params["func"] = self._func_rename_map[original_node.func_ref]

        return nodes.QuantizedCall(**params)


def _find_dependent_methods(func_src: pysource.PySource, ctx: FunctionContext) -> Iterator[_AqTask]:
    """Find methods that are called by the given function within the same module.

    This function analyzes the source code of a method to identify other methods of the same
    class that are called within it. It handles instance methods, class methods, and static
    methods, tracking references through 'self', class references, and closure variables.

    Args:
        func_src: The source code representation of the function to analyze.
        ctx: The `FunctionContext` of `func_src`.

    Returns:
        An iterator of method names that are dependencies of the given function.
        Only yields method names that exist as actual methods on the ModuleType.

    Example:
        For a method that calls `self.forward()` and `<ClassName>.helper()`, this would
        yield both "forward" and "helper" if they are methods on the module.

    Exceptions:
        Dynamic resolutions to the class are not detected. For example, `type(self).helper()`
        will not be identified as a call to the class method `helper`.
    """
    funcdef = func_src.cst(NodeType=libcst.FunctionDef)
    func_name = funcdef.name.value

    ModuleType = ctx.torch_module
    method_type = ctx.method_type

    assert ModuleType is not None
    assert method_type is not None

    module_refs = {ctx.class_var} if ctx.class_var else set[str]()
    try:
        _, extended_module_refs = _scope_vars_and_module_refs(ModuleType, func_name)
        module_refs.update(extended_module_refs)
    except ValueError:
        pass

    for candidate in filter_nodes_by_type(funcdef, nodes.ReplacementCandidate):
        if not isinstance(call_expr := candidate.original, libcst.Call):
            continue

        match call_expr.func:
            case libcst.Attribute(value=libcst.Name(obj), attr=libcst.Name(attr)):
                if obj != ctx.instance_var and obj not in module_refs:
                    continue

                if ff.type_common.method_type(ModuleType, attr) is not MethodType.NO_METHOD:
                    yield _AqTask(module=ModuleType, function=getattr(ModuleType, attr))

            case libcst.Name(call_func_name):
                if method_type == MethodType.METHOD and call_func_name == ctx.instance_var:
                    # When calling self() in a torch.nn.Module subclass, this invokes
                    # __call__ which delegates to forward(). Track as "forward" call.
                    yield _AqTask(module=ModuleType, function=getattr(ModuleType, "forward"))


def _find_dependent_functions(
    func_src: pysource.PySource, ctx: FunctionContext
) -> Iterator[_AqTask]:
    """Find Python functions that are called by the given function.

    This function analyzes the source code to identify calls to other Python functions
    (not methods) that are referenced within the function. It handles both direct function
    calls and calls through attributes, tracking references through closure variables.

    Args:
        func_src: The source code representation of the function to analyze.
        ctx: The `FunctionContext` of `func_src`.

    Returns:
        An iterator of `_AqTask` objects representing Python functions that are called
        by the given function. Only yields functions that can be inspected (Python-defined,
        not built-ins).

    Note:
        Method calls (e.g., `self.method()`) are filtered out as they are handled separately
        by `_find_dependent_methods()`.
    """
    funcdef = func_src.cst(NodeType=libcst.FunctionDef)
    func_name = funcdef.name.value

    method_type = ctx.method_type
    ModuleType = ctx.torch_module or ctx.py_module
    is_method = method_type is not MethodType.NO_METHOD  # whether func_src represents a method

    assert ModuleType is not None

    try:
        scope_vars, module_refs = _scope_vars_and_module_refs(ModuleType, func_name)
    except ValueError:
        # Skip function: no reference available for source inspection and quantization rewriting
        return

    def _is_method_call(call: libcst.Call) -> bool:
        if not is_method:
            return False

        match call.func:
            case libcst.Attribute(value=libcst.Name(obj), attr=libcst.Name(attr)):
                if obj != ctx.instance_var and obj not in module_refs:
                    return False
                return ff.type_common.method_type(ModuleType, attr) is not MethodType.NO_METHOD

        return False

    for candidate in filter_nodes_by_type(funcdef, nodes.ReplacementCandidate):
        if not isinstance(call_expr := candidate.original, libcst.Call):
            continue
        if _is_method_call(call_expr):
            continue

        alias: str | None
        match call_expr.func:
            # Direct function calls: `func_name(args)`
            case libcst.Name(func_name):
                ref = scope_vars.get(func_name)
                module = sys.modules.get(ref.__module__) if ref is not None else None
                if ref is not None and module is not None:
                    alias = None
                    if getattr(module, func_name, None) is ref:
                        alias = func_name
                    yield _AqTask(module=module, function=ref, alias=alias)

            # Attribute access: `module.submodule.func(args)`
            case libcst.Attribute():
                try:
                    module, ref = _resolve_attribute(scope_vars, call_expr.func)
                except AttributeError:
                    continue

                alias = None
                for name, obj in vars(module).items():
                    if obj is ref:
                        alias = name
                        break

                # Only queue functions that are inspectable (Python-defined).
                # Builtin function calls are replaced to quantized versions based on
                # the operator table during function conversion.
                if not inspect.isbuiltin(ref) and inspect.isfunction(ref):
                    yield _AqTask(
                        module=module,
                        function=ref,
                        alias=alias,
                    )


def _scope_vars_and_module_refs(
    ModuleType: type[torch.nn.Module] | types.ModuleType, func_name: str
) -> tuple[dict[str, Any], set[str]]:
    """Extract closure variables and module reference names from a function's scope."""
    scope_vars: dict[str, Any] = {}
    module_refs: set[str] = set()
    if ref := getattr(ModuleType, func_name, None):
        with contextlib.suppress(TypeError):  # getclosurevars raises TypeError on builtins
            # Extract variable names that reference ModuleType from function closure
            closure_vars = inspect.getclosurevars(inspect.unwrap(ref))
            scope_vars = {**closure_vars.nonlocals, **closure_vars.globals}
            for name, value in scope_vars.items():
                if value is ModuleType:
                    module_refs.add(name)
    else:
        msg = f"'{func_name}' is not a members of '{ModuleType}"
        raise ValueError(msg)

    return scope_vars, module_refs


def _resolve_attribute(scope: dict[str, Any], attr: libcst.Attribute) -> tuple[Any, Any]:
    """Resolves an attribute expression to its actual object using the given scope."""
    attr_parts = list(node_processing.iter_attribute(attr))
    full_attr_name = libcst.Module([]).code_for_node(attr)
    if not attr_parts or not isinstance(attr_parts[0], libcst.Name):
        msg = f"Cannot resolve attribute {full_attr_name}"
        raise AttributeError(msg)

    obj = scope.get(attr_parts[0].value)
    parent = obj
    current_path = attr_parts[0].value

    for attr_part in attr_parts[1:]:
        parent = obj
        if parent is None or not isinstance(attr_part, libcst.Name):
            msg = f"Cannot resolve attribute {full_attr_name} (failed at '{current_path}')"
            raise AttributeError(msg)
        obj = getattr(parent, attr_part.value, None)
        current_path += f".{attr_part.value}"
    return parent, obj
