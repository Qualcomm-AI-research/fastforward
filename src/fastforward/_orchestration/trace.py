# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from __future__ import annotations

import inspect
import re

from typing import Any, Callable, cast

import torch
import torch.nn as nn
import torch.utils._pytree as pytree

from packaging.version import Version
from torch import fx
from torch.export import unflatten as _unflatten_exported

from fastforward._orchestration.graph_module import Const, GraphModule, NodeRef, Op, _BaseRef

_MIN_TORCH_VERSION = Version("2.6.0")


def _make_tuple(*items: Any) -> tuple[Any, ...]:
    return items


def _make_list(*items: Any) -> list[Any]:
    return list(items)


def _make_slice(start: Any = None, stop: Any = None, step: Any = None) -> slice:
    return slice(start, stop, step)


def _make_dict(**items: Any) -> dict[str, Any]:
    return items


def _make_method_caller(method_name: str) -> Callable[..., Any]:
    """Bridge an FX `call_method` (target = method-name string) into a callable the engine can invoke."""

    def _call(self: Any, *args: Any, **kwargs: Any) -> Any:
        return getattr(self, method_name)(*args, **kwargs)

    _call.__name__ = method_name
    return _call


def _make_get_attr(root: nn.Module, attr_path: str) -> Callable[[], Any]:
    """Bridge an FX `get_attr` into a callable the engine can invoke."""

    def _get() -> Any:
        obj: Any = root
        for part in attr_path.split("."):
            obj = getattr(obj, part)
        return obj

    _get.__name__ = attr_path
    return _get


def _strip_hardcoded_device_placements(module: nn.Module) -> None:
    """Make the unflattened graph device-agnostic.

    torch.export bakes the tracing device into dtype casts (e.g. ``.float()``
    becomes ``aten.to.dtype_layout(dtype=float32, device='cpu')``). We strip the
    device kwarg so the cast preserves whatever device the tensor is already on.
    """
    for submod in module.modules():
        graph = getattr(submod, "graph", None)
        if not isinstance(graph, fx.Graph):
            continue
        modified = False
        for node in graph.nodes:
            if node.op == "call_function" and "device" in node.kwargs and "dtype" in node.kwargs:
                node.kwargs = {k: v for k, v in node.kwargs.items() if k != "device"}
                modified = True
        if modified and isinstance(submod, fx.GraphModule):
            submod.recompile()


def _prepare_output(spec: pytree.TreeSpec) -> Callable[..., Any]:
    """Reconstruct the model's original return structure from the flat leaves dynamo emits."""

    def _unflatten(*leaves: Any) -> Any:
        return pytree.tree_unflatten(list(leaves), spec)

    _unflatten.__name__ = "unflatten"
    return _unflatten


def _contains_fx_node(x: Any) -> bool:
    """Distinguish an arg subtree that references traced nodes from one that's a pure constant."""
    match x:
        case fx.Node():
            return True
        case slice(start=s, stop=t, step=p):
            return any(_contains_fx_node(i) for i in (s, t, p))
        case tuple() | list():
            return any(_contains_fx_node(i) for i in x)
        case dict():
            return any(_contains_fx_node(v) for v in x.values())
        case _:
            return False


def _capture_out_spec(
    module: nn.Module, exported: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> pytree.TreeSpec:
    """Recover whatever structure the module's return statement has after export."""
    if (spec := getattr(exported, "_out_spec", None)) is not None:
        assert isinstance(spec, pytree.TreeSpec)
        if not spec.is_leaf():
            return spec

    codegen = getattr(exported.graph, "_codegen", None)
    if (info := getattr(codegen, "pytree_info", None)) is not None:
        if (spec := getattr(info, "out_spec", None)) is not None:
            assert isinstance(spec, pytree.TreeSpec)
            return spec

    with torch.no_grad():
        sample = module(*args, **kwargs)

    return pytree.tree_structure(sample)


def _unmangle_placeholder(name: str, positional_names: list[str]) -> str:
    """Restore the caller's parameter name on a placeholder.

    Dynamo internally renames inputs to `l_a_<i>_` / `l_kw_<name>_`; undoing the
    mangling lets the traced graph accept the same args/kwargs as the source module.
    """
    if m := re.fullmatch(r"l_kw_(.+)_", name):
        return m.group(1)
    if m := re.fullmatch(r"l_a_(\d+)_", name):
        idx = int(m.group(1))
        return positional_names[idx] if idx < len(positional_names) else f"arg_{idx}"
    return name


def _is_non_leaf_graph_module(mod: nn.Module) -> bool:
    """True if mod has an fx graph containing call_module nodes (a user-defined wrapper).

    Unflatten wraps every submodule in InterpreterModule (NOT a subclass of fx.GraphModule),
    so we check for `.graph` directly. The presence of call_module nodes inside the graph
    distinguishes a user wrapper (e.g. a transformer block — should recurse) from a
    decomposed library leaf (e.g. nn.Linear, whose graph is placeholder + get_attr +
    aten.linear + output — should be treated as a leaf).
    """
    graph = getattr(mod, "graph", None)
    if not isinstance(graph, fx.Graph):
        return False
    return any(node.op == "call_module" for node in graph.nodes)


def _can_call_original(original: nn.Module, submod: nn.Module, fx_node: fx.Node) -> bool:
    """Check if calling original.forward matches the captured FX call.

    The unflattened InterpreterModule's forward graph reflects exactly what was captured;
    the original nn.Module may differ when torch.export:

    1. Stripped args that were only used for metadata (e.g. dtype/device extracted
       from a tensor) — the call site has fewer args than the original requires.
    2. Fused adjacent ops into the call (e.g. residual capture in a layernorm),
       making the captured call return a tuple where the original returns a tensor.

    When either holds, fall back to the InterpreterModule as the runtime callable.
    """
    try:
        sig = inspect.signature(original.forward)
    except (ValueError, TypeError):
        return False
    n_provided = len(fx_node.args) + len(fx_node.kwargs)
    n_required = sum(
        1
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        and p.default is inspect.Parameter.empty
    )
    if n_provided < n_required:
        return False

    sub_graph = getattr(submod, "graph", None)
    if isinstance(sub_graph, fx.Graph):
        output_node = next((n for n in sub_graph.nodes if n.op == "output"), None)
        if output_node is not None and output_node.args:
            decomposed_outputs = output_node.args[0]
            if isinstance(decomposed_outputs, (list, tuple)) and len(decomposed_outputs) > 1:
                return False
    return True


class _GraphBuilder:
    """Converts an unflattened fx.GraphModule tree into a FastForward GraphModule.

    Holds the mutable state (graph under construction, fx-to-ff node mapping) that
    otherwise has to be threaded through every helper function.
    """

    def __init__(
        self,
        original_root: nn.Module,
        path_prefix: str = "",
        *,
        is_root: bool = False,
        positional_names: list[str] | None = None,
        out_spec: pytree.TreeSpec | None = None,
    ) -> None:
        self._graph = GraphModule()
        self._fx_to_ff: dict[fx.Node, _BaseRef] = {}
        self._original_root = original_root
        self._path_prefix = path_prefix
        self._is_root = is_root
        self._positional_names = positional_names or []
        self._out_spec = out_spec

    def _absolute_path(self, target: str) -> str:
        return f"{self._path_prefix}.{target}" if self._path_prefix else target

    def _convert_arg(self, x: Any, name: str) -> _BaseRef:
        """Convert an FX argument into a `_BaseRef`, creating container nodes where needed."""
        match x:
            case fx.Node():
                return self._fx_to_ff[x]
            case _ if not _contains_fx_node(x):
                return Const(x)
            case tuple() | list() as seq:
                ctor = _make_tuple if isinstance(seq, tuple) else _make_list
                return self._graph.add_node(
                    f"{name}_{ctor.__name__}",
                    ctor,
                    args=tuple(self._convert_arg(v, f"{name}_{i}") for i, v in enumerate(seq)),
                    op=Op.call_function,
                )
            case slice(start=s, stop=t, step=p):
                return self._graph.add_node(
                    f"{name}_make_slice",
                    _make_slice,
                    args=(),
                    kwargs={
                        "start": self._convert_arg(s, f"{name}_start"),
                        "stop": self._convert_arg(t, f"{name}_stop"),
                        "step": self._convert_arg(p, f"{name}_step"),
                    },
                    op=Op.call_function,
                )
            case dict() as d:
                return self._graph.add_node(
                    f"{name}_make_dict",
                    _make_dict,
                    args=(),
                    kwargs={k: self._convert_arg(v, f"{name}_{k}") for k, v in d.items()},
                    op=Op.call_function,
                )
            case _:
                raise TypeError("Unsupported FX argument type: {type(x)!r}")

    def _convert_args(self, fx_node: fx.Node) -> tuple[tuple[_BaseRef, ...], dict[str, _BaseRef]]:
        """Convert all positional and keyword FX args of a node into `_BaseRef`."""
        name = fx_node.name
        return (
            tuple(self._convert_arg(a, f"{name}_arg{i}") for i, a in enumerate(fx_node.args)),
            {k: self._convert_arg(v, f"{name}_{k}") for k, v in fx_node.kwargs.items()},
        )

    def _add_op_node(
        self,
        fx_node: fx.Node,
        name: str,
        module: nn.Module | Callable[..., Any],
        op: Op,
    ) -> NodeRef:
        """Convert args and add a single operation node to the graph."""
        args, kwargs = self._convert_args(fx_node)
        return self._graph.add_node(name, module, args=args, kwargs=kwargs, op=op)

    def _handle_call_module(self, fx_node: fx.Node, fx_gm: nn.Module) -> None:
        assert isinstance(fx_node.target, str)
        target = fx_node.target
        submod = fx_gm.get_submodule(target)
        absolute_path = self._absolute_path(target)

        if _is_non_leaf_graph_module(submod):
            sub_builder = _GraphBuilder(self._original_root, absolute_path)
            sub_ff = sub_builder.build(submod)
            args, kwargs = self._convert_args(fx_node)
            output_refs = self._graph.add_subgraph(target, sub_ff, args, kwargs)
            if len(output_refs) == 1:
                self._fx_to_ff[fx_node] = output_refs[0]
            else:
                self._fx_to_ff[fx_node] = self._graph.add_node(
                    f"{target}_pack_outputs",
                    _make_tuple,
                    args=output_refs,
                    op=Op.call_function,
                )
        else:
            original = self._original_root.get_submodule(absolute_path)
            if not _can_call_original(original, submod, fx_node):
                original = submod
            self._fx_to_ff[fx_node] = self._add_op_node(fx_node, target, original, Op.torch_module)

    def _handle_output(self, fx_node: fx.Node) -> None:
        raw: tuple[Any, ...] | list[Any] = fx_node.args
        if len(raw) == 1 and isinstance(raw[0], (list, tuple)):
            raw = raw[0]
        flat_refs = tuple(self._fx_to_ff[a] for a in raw if isinstance(a, fx.Node))

        if self._is_root:
            assert self._out_spec is not None
            assert all(isinstance(r, NodeRef) for r in flat_refs)
            unflatten_ref = self._graph.add_node(
                "_prepare_output",
                _prepare_output(self._out_spec),
                args=flat_refs,
                op=Op.call_function,
            )
            self._graph.add_output(unflatten_ref)
        else:
            for ref in flat_refs:
                assert isinstance(ref, NodeRef)
                self._graph.add_output(ref)

    def build(self, fx_gm: nn.Module) -> GraphModule:
        """Walk the fx graph and populate the FastForward GraphModule."""
        skip_targets = {
            torch._C._set_grad_enabled,
            torch.amp.autocast_mode._enter_autocast,
            torch.amp.autocast_mode._exit_autocast,
        }
        graph = cast(fx.Graph, fx_gm.graph)
        for fx_node in graph.nodes:
            if fx_node.target in skip_targets:
                continue

            match fx_node.op:
                case "placeholder":
                    name = (
                        _unmangle_placeholder(fx_node.name, self._positional_names)
                        if self._is_root
                        else fx_node.name
                    )
                    self._fx_to_ff[fx_node] = self._graph.add_input(name)

                case "get_attr":
                    assert isinstance(fx_node.target, str)
                    absolute_path = self._absolute_path(fx_node.target)
                    self._fx_to_ff[fx_node] = self._graph.add_node(
                        fx_node.name + "_get_attr",
                        _make_get_attr(self._original_root, absolute_path),
                        args=(),
                        kwargs={},
                        op=Op.get_attr,
                    )

                case "call_module":
                    self._handle_call_module(fx_node, fx_gm)

                case "call_function":
                    assert callable(fx_node.target)
                    self._fx_to_ff[fx_node] = self._add_op_node(
                        fx_node,
                        fx_node.name + "_call_function",
                        fx_node.target,
                        Op.call_function,
                    )

                case "call_method":
                    assert isinstance(fx_node.target, str)
                    self._fx_to_ff[fx_node] = self._add_op_node(
                        fx_node,
                        fx_node.name + "_call_method",
                        _make_method_caller(fx_node.target),
                        Op.call_method,
                    )

                case "output":
                    self._handle_output(fx_node)

                case _:
                    msg = f"Unsupported fx op: {fx_node.op!r}"
                    raise ValueError(msg)

        return self._graph


def trace(module: nn.Module, *args: Any, **kwargs: Any) -> GraphModule:
    """Trace `module` with example inputs and return a `GraphModule`.

    Uses `torch.export` to produce an exported program, then `torch.export.unflatten`
    to reconstruct the module hierarchy. The unflattened fx.GraphModule is recursively
    converted into a FastForward `GraphModule` where leaf modules are preserved as
    `Op.torch_module` nodes (enabling mpath search, offloading, and SubgraphSpec).

    Leaf modules are recovered from `module` via dotted-path lookup
    (``module.get_submodule(path)``), so the returned graph shares weights with
    `module` by identity.

    Args:
        module: Model to trace. Must be in a mode consistent with the desired
            output (e.g. `module.eval()` for inference-mode tracing). Passing
            a `torch.compile`d model is not supported.
        *args: Positional example inputs forwarded to `module` during tracing.
            Dynamo uses these to resolve control flow.
        **kwargs: Keyword example inputs forwarded to `module` during tracing.

    Returns:
        A `GraphModule` whose topology mirrors the module hierarchy, with leaf
        modules as `Op.torch_module` nodes and intermediate modules as inlined
        subgraphs. Produces the same return structure as `module(*args, **kwargs)`.
    """
    if Version(torch.__version__.split("+", 1)[0]) < _MIN_TORCH_VERSION:
        msg = f"fastforward._orchestration.trace requires PyTorch >= {_MIN_TORCH_VERSION}; got {torch.__version__}."
        raise RuntimeError(msg)

    positional_names = [
        p.name
        for p in inspect.signature(module.forward).parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]

    exported_program = torch.export.export(module, args=args, kwargs=kwargs, strict=False)
    unflattened = _unflatten_exported(exported_program)
    _strip_hardcoded_device_placements(unflattened)
    out_spec = _capture_out_spec(module, unflattened, args, kwargs)

    builder = _GraphBuilder(
        module,
        "",
        is_root=True,
        positional_names=positional_names,
        out_spec=out_spec,
    )
    return builder.build(unflattened)
