# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from __future__ import annotations

import contextlib
import inspect
import operator
import re
import warnings
import weakref

from typing import Any, Callable, Iterator, cast

import torch
import torch.nn as nn
import torch.utils._pytree as pytree

from packaging.version import Version
from torch import fx

import fastforward as ff

from fastforward._orchestration.graph_module import Const, GraphModule, NodeRef, Op, _BaseRef

_MIN_TORCH_VERSION = Version("2.6.0")


@contextlib.contextmanager
def _placeholder_uninitialized_params(module: nn.Module) -> Iterator[None]:
    """Temporarily replace uninitialized parameters/buffers with empty placeholders.

    `torch.export` walks every parameter and buffer on a module, which raises on
    `UninitializedParameter`/`UninitializedBuffer` (lazy quantizer scale/offset that
    are only materialized during calibration). Since tracing runs with quantization
    disabled, these tensors are never read, so we swap them for tiny real tensors for
    the duration of the export and restore the originals afterwards.
    """
    uninitialized = (
        torch.nn.parameter.UninitializedParameter,
        torch.nn.parameter.UninitializedBuffer,
    )
    saved: list[tuple[nn.Module, str, Any, bool]] = []
    for submodule in module.modules():
        for name, param in list(submodule.named_parameters(recurse=False)):
            if isinstance(param, uninitialized):
                saved.append((submodule, name, param, True))
                delattr(submodule, name)
                submodule.register_parameter(
                    name, nn.Parameter(torch.empty(0), requires_grad=False)
                )
        for name, buf in list(submodule.named_buffers(recurse=False)):
            if isinstance(buf, uninitialized):
                saved.append((submodule, name, buf, False))
                delattr(submodule, name)
                submodule.register_buffer(name, torch.empty(0))

    try:
        yield
    finally:
        for submodule, name, original, is_param in saved:
            delattr(submodule, name)
            if is_param:
                submodule.register_parameter(name, original)
            else:
                submodule.register_buffer(name, original)


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


_EXPORT_GUARD_OPS = frozenset({
    # every `aten.to.*`: dtype cast, device move, memory-format change
    "aten._assert_tensor_metadata",
    # unbacked symint (`nonzero`, `.item()`, ...), or an explicit `torch._check`
    "aten._assert_scalar",
    "aten.sym_constrain_range",
    "aten.sym_constrain_range_for_size",
    # an explicit `torch._assert_async`; never emitted by export itself
    "aten._assert_async",
})


def _is_export_guard(node: fx.Node) -> bool:
    """True for a side-effect-only guard that records a trace-time assumption.

    A guard returns nothing, so it has no users and cannot contribute to the module's output.
    Ops are matched on their overload packet (`aten._assert_scalar`), so every overload of a
    listed op counts.
    """
    if node.op != "call_function" or node.users:
        return False
    packet = getattr(node.target, "overloadpacket", None)
    return packet is not None and str(packet) in _EXPORT_GUARD_OPS


def _strip_export_artifacts(graph: fx.Graph) -> None:
    """Remove what `torch.export` records about the trace-time environment.

    Export traces one example input, so the graph keeps the properties of that input. Two
    such records are removed:

    * Side-effect-only guards (`aten._assert_*`, `aten.sym_constrain_range_*`) that assert a
      later input still matches the traced one. FF orchestration uses the eager modules, so
      they guard nothing, and they do harm: a guard inside a submodule belongs to the top-level
      model, so `unflatten` rebuilds the submodule as two partial calls with no children left
      inside, and the trace reads it as a leaf.
    * A `device` kwarg on a cast, which pins the cast to the trace device. Without the kwarg,
      the cast keeps the device that the tensor already has.

    Args:
        graph: The flat graph of an exported program, modified in place.
    """
    for node in list(graph.nodes):
        if _is_export_guard(node):
            graph.erase_node(node)
        elif node.op == "call_function" and "device" in node.kwargs and "dtype" in node.kwargs:
            node.kwargs = {k: v for k, v in node.kwargs.items() if k != "device"}


def _op_schema(node: fx.Node) -> torch._C.FunctionSchema | None:
    """Return the schema of an ATen op node, or None for any other node."""
    if node.op != "call_function":
        return None
    schema = getattr(node.target, "_schema", None)
    return schema if isinstance(schema, torch._C.FunctionSchema) else None


def _schema_arguments(
    node: fx.Node, schema: torch._C.FunctionSchema
) -> Iterator[tuple[torch._C.Argument, Any]]:
    """Pair every argument of `node` with its entry in the op schema."""
    yield from zip(schema.arguments, node.args)
    by_name = {argument.name: argument for argument in schema.arguments}
    for name, value in node.kwargs.items():
        yield by_name[name], value


def _written_nodes(node: fx.Node) -> list[fx.Node]:
    """Return the tensors that `node` overwrites in place."""
    schema = _op_schema(node)
    if schema is None or not schema.is_mutable:
        return []
    return [
        value
        for argument, value in _schema_arguments(node, schema)
        if argument.alias_info is not None
        and argument.alias_info.is_write
        and isinstance(value, fx.Node)
    ]


def _aliased_node(node: fx.Node) -> fx.Node | None:
    """Return the direct base of node: the tensor it views, writes in place, or takes a part of.

    This is the tensor written by an in-place op, or the tensor viewed by a view op
    (`slice`, `view`, `unbind`, ...). The result of `getitem` shares the storage of its
    container.
    """
    if node.op == "call_function" and node.target is operator.getitem:
        container = node.args[0]
        return container if isinstance(container, fx.Node) else None
    schema = _op_schema(node)
    if schema is None or not any(ret.alias_info is not None for ret in schema.returns):
        return None
    if schema.is_mutable:
        candidates = _written_nodes(node)
    else:
        candidates = [
            value
            for argument, value in _schema_arguments(node, schema)
            if argument.alias_info is not None and isinstance(value, fx.Node)
        ]
    return candidates[0] if candidates else None


def _make_fused_call(fused: fx.GraphModule, name: str) -> Callable[..., Any]:
    """Bridge a fused fx graph into a plain function, so that it is not a module node."""

    def _call(*inputs: Any) -> Any:
        return fused(*inputs)

    _call.__name__ = name
    return _call


def _in_place_clusters(graph: fx.Graph) -> list[tuple[fx.Node, list[fx.Node]]]:
    """Group each tensor that is overwritten in place with the nodes that must stay in order.

    An in-place write has no consumers, so the graph has no edge from the write to the nodes
    that read the tensor afterwards. For example:

        x = torch.zeros_like(h)
        x[..., :] = a
        y = x * 2

    The assignment becomes a `slice` and a `copy_`. The graph has the edges
    `zeros_like -> slice -> copy_` and `zeros_like -> mul`. Nothing links `copy_` to `mul`, so
    a scheduler can run `zeros_like`, `mul`, and only then `copy_`. Then `y` is computed from
    the zeros and not from `a`.

    This function ensures that the in-place ops are clustered correctly in-order.

    Returns:
        One `(root, cluster)` pair per overwritten tensor. `root` is the node that creates the
            tensor. `cluster` is sorted in graph order

    Raises:
        NotImplementedError: If a node that is not a plain tensor op (for example a child module)
            reads the tensor between two writes, or if two clusters overlap.
    """
    nodes = list(graph.nodes)
    position = {node: index for index, node in enumerate(nodes)}

    # Figure out the data root of each node, if possible. Views share the same root,
    # other objects are its own roots.
    root_of: dict[fx.Node, fx.Node] = {}
    for node in nodes:
        base = _aliased_node(node)
        root_of[node] = root_of.get(base, base) if base is not None else node

    # Given the data roots, figure out what nodes write in-place to it.
    writes: dict[fx.Node, list[fx.Node]] = {}
    for node in nodes:
        for target in _written_nodes(node):
            writes.setdefault(root_of[target], []).append(node)

    clusters: list[tuple[fx.Node, list[fx.Node]]] = []
    claimed: set[fx.Node] = set()
    for root, writers in writes.items():
        # We cannot write in-place to graph inputs / buffers or child modules.
        if root.op != "call_function":
            kind = {
                "placeholder": "a graph input",
                "get_attr": "a buffer",
                "call_module": "a child module output",
            }[root.op]
            warnings.warn(
                f"Node {writers[0].name!r} writes in place to {kind} ({root.name!r}). The trace does "
                "not order it against the reads of that tensor.",
            )
            continue

        # Init the clusters with the root node and all "viewing" nodes.
        last = max(position[writer] for writer in writers)
        cluster = {node for node in nodes[: last + 1] if root_of[node] is root}
        # A reader of the shared storage must stay in order. The consumers of a reader's fresh
        # output (for example a clone) only stay in order if they feed a later write.
        views = set(cluster)
        feeds_write = set(writers)
        for node in reversed(nodes[position[root] : last + 1]):
            if any(user in feeds_write for user in node.users):
                feeds_write.add(node)
        for node in nodes[position[root] : last + 1]:
            if node in cluster:
                continue
            reads_storage = any(arg in views for arg in node.all_input_nodes)
            if not (
                reads_storage
                or (node in feeds_write and any(a in cluster for a in node.all_input_nodes))
            ):
                continue
            # anything that will end up reading from the cluster is also pulled in.
            # (except if it is part of another cluster or a child module).
            if node.op != "call_function" or node in claimed or _written_nodes(node):
                msg = (
                    f"Cannot keep the order of the in-place writes to {root.name!r}: "
                    f"node {node.name!r} reads it between two writes and cannot be fused."
                )
                raise NotImplementedError(msg)
            cluster.add(node)

        # Clusters cannot overlap since we fuse each to a distinct node.
        if overlap := cluster & claimed:
            msg = f"In-place writes to {root.name!r} overlap with {next(iter(overlap)).name!r}."
            raise NotImplementedError(msg)
        claimed |= cluster
        clusters.append((root, sorted(cluster, key=position.__getitem__)))
    return clusters


def _fuse_in_place_writes(graph: fx.Graph) -> None:
    """Replace each cluster of in-place writes with one node that runs the cluster in order.

    `_in_place_clusters` finds the nodes that must stay together. This function turns each
    cluster into a single `call_function` node named `<root>_writes`:

    - The inputs are the nodes outside the cluster that the cluster reads.
    - The outputs are the cluster nodes that nodes outside the cluster use.
    - The body is a copy of the cluster nodes in graph order.

    Every user outside the cluster reads its value from the fused node through
    `operator.getitem`. This gives it a data edge to the fused node, so it runs after the last
    write. The original cluster nodes are then erased.

    Args:
        graph: A graph with ATen ops, modified in place.
    """
    for root, cluster in _in_place_clusters(graph):
        members = set(cluster)
        inputs = list(
            dict.fromkeys(
                arg for node in cluster for arg in node.all_input_nodes if arg not in members
            )
        )
        outputs = [node for node in cluster if any(user not in members for user in node.users)]

        # Given cluster inputs and outputs, create a new fx.Graph.
        body = fx.Graph()
        env: dict[fx.Node, fx.Node] = {arg: body.placeholder(arg.name) for arg in inputs}
        for node in cluster:
            env[node] = body.node_copy(node, env.__getitem__)
        body.output(tuple(env[node] for node in outputs))
        fused = _make_fused_call(fx.GraphModule(nn.Module(), body), f"{root.name}_writes")

        # Insert the fx.Graph as a fused call after the cluster runs.
        with graph.inserting_after(cluster[-1]):
            call = graph.create_node(
                "call_function", fused, tuple(inputs), name=f"{root.name}_writes"
            )

        # Nodes that need to access cluster-internal nodes will be rewired through a `getitem` call.`
        anchor = call
        items: list[fx.Node] = []
        for index, node in enumerate(outputs):
            with graph.inserting_after(anchor):
                item = graph.call_function(operator.getitem, (call, index))
            items.append(item)
            node.replace_all_uses_with(item, delete_user_cb=lambda user: user not in members)
            anchor = item
        # Users that ran before the last write now depend on the fused node, so move them after it.
        late = set(items)
        for node in list(graph.nodes):
            if node is call:
                break
            if node not in members and any(arg in late for arg in node.all_input_nodes):
                late.add(node)
                anchor.append(node)
                anchor = node

        # Now erase the original cluster nodes in reverse order (remove users before their inputs).
        for node in reversed(cluster):
            graph.erase_node(node)


def _export_and_unflatten(
    module: nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    preserve: tuple[str, ...] = (),
) -> nn.Module:
    """Export `module`, remove the trace-time artefacts, and rebuild the module tree."""
    exported_program = torch.export.export(
        module, args=args, kwargs=kwargs, preserve_module_call_signature=preserve
    )
    _strip_export_artifacts(exported_program.graph_module.graph)
    exported_program.graph_module.recompile()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Attempted to insert a get_attr Node with no underlying reference",
            category=UserWarning,
        )
        unflattened = torch.export.unflatten(exported_program)
    for submodule in unflattened.modules():
        if _is_non_leaf_graph_module(submodule):
            _fuse_in_place_writes(cast(fx.Graph, submodule.graph))
    return unflattened


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


def _fold_needs_preservation(mod: nn.Module) -> bool:
    """True if the fold's call site has multiple positional inputs that need signature recovery.

    Without `preserve_module_call_signature`, torch.export.unflatten collapses all
    module call kwargs into positional args. Folds with a single positional input
    don't need this (order is unambiguous). Folds with multiple inputs (e.g. a
    decoder layer receiving hidden_states + attention_mask + position_embeddings)
    need preservation to recover the correct args/kwargs split.
    """
    graph = getattr(mod, "graph", None)
    if not isinstance(graph, fx.Graph):
        return False
    n_placeholders = sum(1 for n in graph.nodes if n.op == "placeholder")
    return n_placeholders > 1


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

    def _has_attr_on_original(self, absolute_path: str) -> bool:
        """Check if the attribute path exists on the original module."""
        obj = self._original_root
        for part in absolute_path.split("."):
            if not hasattr(obj, part):
                return False
            obj = getattr(obj, part)
        return True

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
            original = self._original_root.get_submodule(absolute_path)
            output_refs = self._graph.add_subgraph(
                target, sub_ff, args, kwargs, original_module=original
            )
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
                    if hasattr(fx_gm, fx_node.target) and not self._has_attr_on_original(
                        absolute_path
                    ):
                        self._fx_to_ff[fx_node] = Const(getattr(fx_gm, fx_node.target))
                    else:
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


# We keep track of a GraphModule cache to ensure we don't re-trace the same model twice
# under different orchestration needs (e.g. PTQ vs inference).
_graph_cache: weakref.WeakKeyDictionary[torch.nn.Module, GraphModule] = weakref.WeakKeyDictionary()


@ff.flags.context(ff.strict_quantization, False)
@torch.no_grad()
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

    if module in _graph_cache:
        return _graph_cache[module]

    positional_names = [
        p.name
        for p in inspect.signature(module.forward).parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]

    with ff.disable_quantization(module), _placeholder_uninitialized_params(module):
        unflattened = _export_and_unflatten(module, args, kwargs)

        fold_fqns = tuple(
            name
            for name, mod in unflattened.named_modules()
            if _is_non_leaf_graph_module(mod) and name and _fold_needs_preservation(mod)
        )
        if fold_fqns:
            unflattened = _export_and_unflatten(module, args, kwargs, preserve=fold_fqns)

        out_spec = _capture_out_spec(module, unflattened, args, kwargs)

    builder = _GraphBuilder(
        module,
        "",
        is_root=True,
        positional_names=positional_names,
        out_spec=out_spec,
    )

    _graph_cache[module] = builder.build(unflattened)
    return _graph_cache[module]
