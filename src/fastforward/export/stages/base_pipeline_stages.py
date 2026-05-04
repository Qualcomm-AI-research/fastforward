# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Any, TypeAlias, cast

import torch
import torch.fx

from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.infra.pass_manager import PassManager

import fastforward as ff

from fastforward.export.stages.passes import AnnotateFFQuantSpecs, PropagateFFQuantSpecs

_SampleInputsT: TypeAlias = list[tuple[tuple[Any, ...], dict[str, Any]]]
_EvaluableModuleT: TypeAlias = torch.nn.Module | torch.fx.GraphModule


def _collect_quantizer_module_paths(module: torch.nn.Module) -> set[str]:
    quantizer_paths: set[str] = set()
    for name, submodule in module.named_modules(remove_duplicate=False):
        if name and isinstance(submodule, ff.nn.Quantizer):
            quantizer_paths.add(name)
    return quantizer_paths


def _module_path_prefixes(path: str, include_full_path: bool) -> set[str]:
    parts = path.split(".")
    max_length = len(parts) if include_full_path else len(parts) - 1
    if max_length <= 0:
        return set()
    return {".".join(parts[:idx]) for idx in range(1, max_length + 1)}


def _module_prefixes_from_get_attr_target(module: torch.fx.GraphModule, target: str) -> set[str]:
    prefixes: set[str] = set()
    parts = target.split(".")
    for idx in range(1, len(parts) + 1):
        prefix = ".".join(parts[:idx])
        try:
            module.get_submodule(prefix)
        except AttributeError:
            continue
        prefixes.add(prefix)
    return prefixes


def _collect_graph_referenced_module_paths(module: torch.fx.GraphModule) -> set[str]:
    referenced_module_paths: set[str] = set()

    for node in module.graph.nodes:
        if not isinstance(node.target, str):
            continue
        if node.op == "call_module":
            referenced_module_paths.update(
                _module_path_prefixes(node.target, include_full_path=True)
            )
        elif node.op == "get_attr":
            referenced_module_paths.update(
                _module_prefixes_from_get_attr_target(module, node.target)
            )

    return referenced_module_paths


class PruneUnusedGetAttrsPass:
    """Pass that removes unused ``get_attr`` nodes from an FX graph module."""

    def __call__(self, module: torch.fx.GraphModule) -> PassResult:
        """Remove unused ``get_attr`` nodes and recompile the graph module."""
        nodes_to_remove: list[torch.fx.Node] = []
        for node in module.graph.nodes:
            if node.op == "get_attr" and len(node.users) == 0:
                nodes_to_remove.append(node)

        for node in nodes_to_remove:
            module.graph.erase_node(node)

        module.graph.eliminate_dead_code()
        module.recompile()
        return PassResult(module, True)


class FinalizeGraphModulePass:
    """Pass that finalizes graph cleanup after quantizer artifact pruning."""

    def __call__(self, module: torch.fx.GraphModule) -> PassResult:
        """Delete unused submodules, run dead-code elimination, and recompile."""
        module.delete_all_unused_submodules()
        module.graph.eliminate_dead_code()
        module.recompile()
        return PassResult(module, True)


def stage_capture_impl_ff(
    modules: tuple[ff.nn.QuantizedModule, ...],
    sample_inputs: _SampleInputsT,
    context: dict[str, Any],
) -> torch.fx.GraphModule:
    """Capture FastForward quantized module and convert to quantization-free graph.

    Export a FastForward QuantizedModule using torch.export, then applies passes to
    annotate and propagate quantization specifications while removing the actual
    quantization operations from the graph.

    Args:
        modules: Tuple containing a single FastForward QuantizedModule to be captured
        sample_inputs: List of sample input tuples (args, kwargs) for module execution
        context: Pipeline context dictionary (unused in this stage)

    Returns:
        A quantization-free graph module with FF quantization specs annotated on nodes
        but quantization operations removed

    Raises:
        ValueError: If sample_inputs is empty
    """
    del context
    (module,) = modules
    if len(sample_inputs) == 0:
        raise ValueError("sample_inputs cannot be empty")

    sample_args, sample_kwargs = sample_inputs[0]

    with ff.export_mode(True), torch.no_grad(), ff.strict_quantization(False):
        exported = torch.export.export(module, sample_args, sample_kwargs)
    exported = exported.run_decompositions()
    captured = exported.module()

    pasman = PassManager([
        AnnotateFFQuantSpecs(),
        PropagateFFQuantSpecs(),
    ])
    quant_free_module: torch.fx.GraphModule = pasman(captured).graph_module
    quant_free_module.graph.eliminate_dead_code()
    quant_free_module.recompile()

    return quant_free_module


def stage_passthrough_ff_module(
    modules: tuple[ff.nn.QuantizedModule, ...],
    sample_inputs: _SampleInputsT,
    context: dict[str, Any],
) -> ff.nn.QuantizedModule:
    """Pass through the original FF module for downstream multi-input stages."""
    del sample_inputs, context
    (module,) = modules
    return module


def stage_cleanup_ff_quantizer_artifacts(
    modules: tuple[torch.fx.GraphModule, ff.nn.QuantizedModule],
    sample_inputs: _SampleInputsT,
    context: dict[str, Any],
) -> torch.fx.GraphModule:
    """Remove FF quantizer artifacts from a captured graph module.

    This stage runs a dedicated cleanup flow after capture:
    prune unused get_attrs, validate no live quantizer references remain,
    then finalize graph/module cleanup.
    """
    del sample_inputs, context
    module, source_module = modules
    quantizer_module_paths = _collect_quantizer_module_paths(source_module)

    prune_pass_manager = PassManager([
        PruneUnusedGetAttrsPass(),
    ])
    pruned_module = prune_pass_manager(module).graph_module

    referenced_module_paths = _collect_graph_referenced_module_paths(pruned_module)
    live_quantizer_refs = quantizer_module_paths.intersection(referenced_module_paths)
    if live_quantizer_refs:
        msg = (
            "Captured graph still references quantizer submodules after FF cleanup passes: "
            f"{sorted(live_quantizer_refs)}"
        )
        raise RuntimeError(msg)

    finalize_pass_manager = PassManager([
        FinalizeGraphModulePass(),
    ])
    return cast(torch.fx.GraphModule, finalize_pass_manager(pruned_module).graph_module)


def stage_fp_eval(
    modules: tuple[_EvaluableModuleT, ...],
    sample_inputs: _SampleInputsT,
    context: dict[str, Any],
) -> list[Any]:
    """Evaluate module in floating-point mode with quantization disabled.

    Args:
        modules: Tuple containing a single module to evaluate
            (``torch.nn.Module`` or ``torch.fx.GraphModule``)
        sample_inputs: List of sample input tuples (args, kwargs) for evaluation
        context: Pipeline context dictionary (unused in this stage)

    Returns:
        List of floating-point outputs corresponding to each sample input
    """
    del context
    module = modules[0]
    with ff.disable_quantization(module), torch.no_grad():
        return [module(*args, **kwargs) for args, kwargs in sample_inputs]


def stage_quantized_eval(
    modules: tuple[_EvaluableModuleT, ...],
    sample_inputs: _SampleInputsT,
    context: dict[str, Any],
) -> list[Any]:
    """Evaluate module in quantized mode and dequantize outputs.

    Args:
        modules: Tuple containing a single module to evaluate
            (``torch.nn.Module`` or ``torch.fx.GraphModule``)
        sample_inputs: List of sample input tuples (args, kwargs) for evaluation
        context: Pipeline context dictionary (unused in this stage)

    Returns:
        List of dequantized outputs corresponding to each sample input
    """
    del context
    module = modules[0]
    with torch.no_grad():
        return [module(*args, **kwargs).dequantize() for args, kwargs in sample_inputs]
