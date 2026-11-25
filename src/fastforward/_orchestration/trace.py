# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from __future__ import annotations

import collections
import contextlib
import dataclasses
import inspect
import operator
import re
import warnings
import weakref

from typing import Any, Callable, Iterator, cast

import torch
import torch.fx._pytree as fx_pytree
import torch.nn as nn
import torch.utils._pytree as pytree

from packaging.version import Version
from torch import fx
from torch.export.graph_signature import InputKind
from torch.fx.experimental import proxy_tensor
from torch.fx.proxy import GraphAppendingTracer

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


def _is_plain_pytree(spec: pytree.TreeSpec) -> bool:
    """Whether argument conversion can reconstruct every container in a tree.

    A dict must have string keys because it can become the keyword arguments of a call.
    """
    if spec.is_leaf():
        return True
    if spec.type not in (tuple, list, dict):
        return False
    if spec.type is dict and not all(isinstance(key, str) for key in spec.context):
        return False
    children = spec.children() if hasattr(spec, "children") else spec.children_specs
    return all(_is_plain_pytree(child) for child in children)


# The ops that export records for `torch.no_grad`, `torch.enable_grad` and `torch.autocast`.
_GRAD_AND_AUTOCAST_OPS = frozenset({
    torch._C._set_grad_enabled,
    torch.amp.autocast_mode._enter_autocast,
    torch.amp.autocast_mode._exit_autocast,
})


@dataclasses.dataclass
class _Tree:
    """A value that a module call takes or returns, flattened by `pytree`.

    Each tensor leaf is replaced by the graph node that made the tensor. The other leaves are
    constants.

    Args:
        leaves: The leaves of the value, in `pytree` order.
        spec: The structure of the value.
        paths: The key path of each leaf, or None if a container type has no key paths.
    """

    leaves: list[Any]
    spec: pytree.TreeSpec
    paths: list[pytree.KeyPath] | None

    @property
    def nodes(self) -> list[fx.Node]:
        """The leaves that are graph nodes, in order."""
        return [leaf for leaf in self.leaves if isinstance(leaf, fx.Node)]


@dataclasses.dataclass(eq=False)
class _Call:
    """One call of a module during the export, with the graph nodes that the call made.

    Args:
        module: The module that was called.
        name: The node name of the call: the path of the module relative to the module of the
            parent call. The n-th later call of the same path in one parent call gets `@<n>`.
        parent: The call that made this call, or None for the call of the model.
        args: The positional arguments, bound to the signature of `forward`.
        kwargs: The keyword-only arguments, and the arguments of a `**kwargs` parameter.
        output: The return value.
        body: The nodes that the call made itself and its child calls, in the order of the trace.
        markers: The nodes of `body` that only mark an output that the call did not make.
        relinked: The node of each argument, parameter or buffer tensor whose link the call
            changed to a node of `body`, by that node.
    """

    module: nn.Module
    name: str
    parent: _Call | None
    args: list[_Tree]
    kwargs: dict[str, _Tree]
    output: _Tree | None = None
    body: list[fx.Node | _Call] = dataclasses.field(default_factory=list)
    markers: set[fx.Node] = dataclasses.field(default_factory=set)
    relinked: dict[fx.Node, fx.Node] = dataclasses.field(default_factory=dict)

    @property
    def children(self) -> list[_Call]:
        """The child calls, in call order."""
        return [item for item in self.body if isinstance(item, _Call)]


class _CallRecorder:
    """Record each module call of a non-strict `torch.export` as a tree of `_Call`.

    Non-strict export runs the model on fake tensors under a proxy mode. The mode appends each op
    to one fx graph and links each tensor to the node that made it. A hook on each module reads
    these links, so it gets the exact nodes that a call reads and returns. The nodes that the graph
    gets between two hooks belong to the innermost open call.

    A call can return a tensor that it did not make, for example a no-op quantizer that returns
    its input. The forward hook then returns an `aten.alias` of the tensor. The caller reads the
    alias, so the call stays in the data flow of the caller.

    A tensor that the trace did not make, such as a plain tensor attribute, has no node. It stays
    a constant leaf of the call.

    An op can return its input itself, for example `x.to(torch.float32)` on a float32 tensor. The
    trace then links the tensor to the node of the op, also for the code that reads the tensor
    after the call. For each argument, parameter and buffer of the call, the call records the node
    of the tensor before the call in `relinked`.
    """

    def __init__(self, model: nn.Module) -> None:
        self._model = model
        self._paths = {module: path for path, module in model.named_modules()}
        self._graph: fx.Graph | None = None
        self._seen: set[fx.Node] = set()
        self._owner: dict[fx.Node, _Call] = {}
        self._open: list[_Call] = []
        self._links_before: dict[_Call, list[tuple[torch.Tensor, fx.Node]]] = {}
        self._name_counts: collections.defaultdict[_Call, collections.Counter[str]] = (
            collections.defaultdict(collections.Counter)
        )
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self.root: _Call | None = None

    def __enter__(self) -> _CallRecorder:
        for module in self._model.modules():
            # The pre-hook runs before the pre-hooks of the user, so that the call records the
            # arguments that the module gets. A leaf node runs the module, with its hooks.
            self._handles.append(
                module.register_forward_pre_hook(self._enter_call, with_kwargs=True, prepend=True)
            )
            self._handles.append(module.register_forward_hook(self._exit_call, with_kwargs=True))
        return self

    def __exit__(self, *exc_info: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _tracer(self) -> Any:
        """Return the tracer of the export, which appends each op to the exported graph."""
        mode = proxy_tensor.get_proxy_mode()
        if mode is None:
            msg = "A module was called outside the trace of `torch.export`."
            raise RuntimeError(msg)
        if self._graph is not None and mode.tracer.graph is not self._graph:
            msg = "A module was called inside a nested trace, such as the branch of `torch.cond`."
            raise NotImplementedError(msg)
        return mode.tracer

    def _node_of(self, tensor: torch.Tensor) -> fx.Node | None:
        """Return the node that made `tensor`, or None if the trace did not make it."""
        slot = proxy_tensor.get_proxy_slot(tensor, self._tracer(), None)
        if slot is None:
            return None
        node: fx.Node = slot.proxy.node
        return node

    def _flatten(self, value: Any) -> _Tree:
        try:
            keyed_leaves, spec = pytree.tree_flatten_with_path(value)
        except ValueError:
            # A type that is registered without `flatten_with_keys_fn`, e.g. `ModelOutput`.
            leaves, spec = pytree.tree_flatten(value)
            return _Tree([self._leaf(x) for x in leaves], spec, None)
        paths = [path for path, _ in keyed_leaves]
        return _Tree([self._leaf(x) for _, x in keyed_leaves], spec, paths)

    def _leaf(self, value: Any) -> Any:
        """Return the node that made a tensor leaf, or the leaf itself if it is a constant."""
        if not isinstance(value, torch.Tensor):
            return value
        node = self._node_of(value)
        return value if node is None else node

    def _claim_new_nodes(self, owner: _Call | None) -> list[fx.Node]:
        """Give the nodes that the graph got since the last hook to `owner`."""
        assert self._graph is not None
        new_nodes: list[fx.Node] = []
        for node in reversed(self._graph.nodes):
            if node in self._seen:
                break
            new_nodes.append(node)
        new_nodes.reverse()
        self._seen.update(new_nodes)
        if owner is not None:
            owner.body.extend(new_nodes)
            self._owner.update(dict.fromkeys(new_nodes, owner))
        return new_nodes

    def _call_name(self, parent: _Call, module: nn.Module) -> str:
        parent_path, path = self._paths[parent.module], self._paths[module]
        relative_path = path.removeprefix(f"{parent_path}.") if parent_path else path
        earlier_calls = self._name_counts[parent][relative_path]
        self._name_counts[parent][relative_path] += 1
        return f"{relative_path}@{earlier_calls}" if earlier_calls else relative_path

    def _enter_call(self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        graph = self._tracer().graph
        # A call records its arguments bound to the signature, so that `f(x)` and `f(input=x)`
        # give the same node: only keyword-only arguments stay keyword arguments.
        bound = inspect.signature(module.forward).bind(*args, **kwargs)
        flat_args = [self._flatten(arg) for arg in bound.args]
        flat_kwargs = {key: self._flatten(value) for key, value in bound.kwargs.items()}
        if not self._open:
            if self.root is not None or module is not self._model:
                msg = "`trace` needs exactly one call of the model in the export."
                raise RuntimeError(msg)
            self._graph = graph
            self._claim_new_nodes(None)
            call = self.root = _Call(module, "", None, flat_args, flat_kwargs)
        else:
            parent = self._open[-1]
            self._claim_new_nodes(parent)
            call = _Call(module, self._call_name(parent, module), parent, flat_args, flat_kwargs)
            parent.body.append(call)
        self._open.append(call)
        # The ops of the call can relink its arguments and the parameters and buffers of its module.
        tensors = pytree.tree_leaves((bound.args, bound.kwargs))
        tensors += [*module.parameters(), *module.buffers()]
        self._links_before[call] = [
            (tensor, node)
            for tensor in tensors
            if isinstance(tensor, torch.Tensor) and (node := self._node_of(tensor)) is not None
        ]

    def _exit_call(
        self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any
    ) -> Any:
        del module, args, kwargs
        call = self._open.pop()
        self._claim_new_nodes(call)
        for tensor, node in self._links_before.pop(call):
            relinked = self._node_of(tensor)
            # `_CallSplitter` orders the reads of a tensor that the call writes in place.
            if relinked is not None and relinked is not node and not _written_nodes(relinked):
                call.relinked[relinked] = node
        if call is self.root:
            self._erase_grad_and_autocast_nodes()
        # A tensor that the call returns but did not make (for example its input) becomes an
        # alias, so that the caller reads it from this call.
        leaves, spec = pytree.tree_flatten(output)
        aliased = [
            self._alias_passthrough(call, x) if isinstance(x, torch.Tensor) else x for x in leaves
        ]
        if all(new is old for new, old in zip(aliased, leaves)):
            call.output = self._flatten(output)
            # The hook returns None to keep the output of the module.
            return None
        output = pytree.tree_unflatten(aliased, spec)
        call.output = self._flatten(output)
        return output

    def _erase_grad_and_autocast_nodes(self) -> None:
        """Remove the grad-mode and autocast nodes from the exported graph.

        If the graph has such nodes, export replaces each region between them with a higher-order
        op: it splits the graph and inlines the parts again, which makes new nodes. The recorded
        nodes are then not in the exported graph. Without these nodes, the graph stays the
        recorded graph. A leaf node runs its module, so it keeps the grad mode and the autocast
        state of the module. The ops of a fold run in the grad mode and the autocast state of the
        caller, so a fold whose own forward changes them gets a warning.
        """
        assert self._graph is not None
        changes: dict[_Call, set[str]] = {}
        for node in reversed(self._graph.nodes):
            if node.op == "call_function" and node.target in _GRAD_AND_AUTOCAST_OPS:
                if (owner := self._owner.pop(node, None)) is not None:
                    owner.body.remove(node)
                    grad_mode = node.target is torch._C._set_grad_enabled
                    changes.setdefault(owner, set()).add(
                        "grad mode" if grad_mode else "autocast state"
                    )
                self._graph.erase_node(node)
        live = set(self._graph.nodes)
        for call, kinds in changes.items():
            if _is_fold(call, live):
                warnings.warn(
                    f"The call {call.name or 'of the model'!r} changes the "
                    f"{' and the '.join(sorted(kinds))} in its forward. The trace runs the ops of "
                    "this call in the grad mode and the autocast state of its caller. Move the "
                    "region into a child module to keep it.",
                )

    def _alias_passthrough(self, call: _Call, tensor: torch.Tensor) -> torch.Tensor:
        """Return `tensor` if `call` or one of its child calls made it, else an alias of it.

        A tensor that the trace did not make stays a constant, so it is returned as it is.
        """
        node = self._node_of(tensor)
        if node is None:
            return tensor
        owner = self._owner.get(node)
        while owner is not None and owner is not call:
            owner = owner.parent
        if owner is call:
            return tensor
        marker: torch.Tensor = torch.ops.aten.alias.default(tensor)
        call.markers.update(self._claim_new_nodes(call))
        return marker


def _has_ops(call: _Call, live: set[fx.Node]) -> bool:
    """Whether a call or one of its child calls runs an op of the exported graph."""
    return any(
        _has_ops(item, live)
        if isinstance(item, _Call)
        else item in live and item not in call.markers
        for item in call.body
    )


def _is_fold(call: _Call, live: set[fx.Node]) -> bool:
    """Whether the trace runs the ops of a call itself, and not the module of the call."""
    return call.parent is None or any(_has_ops(child, live) for child in call.children)


def _input_names(call: _Call) -> list[str]:
    """Name each argument of a call: the parameter name of a positional one, else its keyword."""
    parameter_names: list[str] = []
    for parameter in inspect.signature(call.module.forward).parameters.values():
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD):
            parameter_names.append(parameter.name)
        elif parameter.kind is parameter.VAR_POSITIONAL:
            extra = range(len(call.args) - len(parameter_names))
            parameter_names.extend(f"{parameter.name}_{index}" for index in extra)
    positional_names = [
        parameter_names[index] if index < len(parameter_names) else f"arg{index}"
        for index in range(len(call.args))
    ]
    return [*positional_names, *call.kwargs]


def _bind_parts(graph: fx.Graph, base: fx.Node, tree: _Tree, env: dict[fx.Node, fx.Node]) -> None:
    """Map each node leaf of `tree` to the part of `base` that holds the leaf.

    A part is read at the key path of the leaf, so it needs no op. A container type without key
    paths, such as `transformers.ModelOutput`, is flattened once, and each leaf is read at its
    `pytree` index.
    """
    if tree.spec.is_leaf():
        if tree.nodes:
            env.setdefault(tree.nodes[0], base)
        return
    if tree.paths is None:
        leaves = graph.call_function(pytree.tree_leaves, (base,))
        for index, leaf in enumerate(tree.leaves):
            if isinstance(leaf, fx.Node) and leaf not in env:
                env[leaf] = graph.call_function(operator.getitem, (leaves, index))
        return
    for path, leaf in zip(tree.paths, tree.leaves):
        if not isinstance(leaf, fx.Node) or leaf in env:
            continue
        part = base
        for key in path:
            match key:
                case pytree.SequenceKey(idx=index):
                    part = graph.call_function(operator.getitem, (part, index))
                case pytree.MappingKey(key=mapping_key):
                    part = graph.call_function(operator.getitem, (part, mapping_key))
                case pytree.GetAttrKey(name=name):
                    part = graph.call_function(getattr, (part, name))
                case _:
                    msg = f"Unsupported pytree key: {key!r}"
                    raise NotImplementedError(msg)
        env[leaf] = part


def _after(value: Any, *_: Any) -> Any:
    """Return `value`. The other arguments only make the node run after them."""
    return value


def _in_place_writes(call: _Call) -> Iterator[fx.Node]:
    """Yield the nodes that write a tensor in place, in `call` and in its child calls."""
    for item in call.body:
        if isinstance(item, _Call):
            yield from _in_place_writes(item)
        elif _written_nodes(item):
            yield item


@dataclasses.dataclass
class _SplitTrace:
    """The exported graph, split into one fx graph per fold call.

    Args:
        graphs: The graph of each fold call. Each child call is a `call_module` node.
        calls: The call of each `call_module` node.
        lifted: The constant tensors that export lifted out of the model, by `get_attr` target.
    """

    graphs: dict[_Call, fx.Graph]
    calls: dict[fx.Node, _Call]
    lifted: dict[str, Any]


class _CallSplitter:
    """Split the flat exported graph into one fx graph per fold call.

    A call is a fold if one of its child calls runs an op, and the call of the model is always a
    fold. Every other call is a leaf: its graph is not needed, because the leaf node runs the
    module itself.

    The graph of a call has one placeholder per argument as called. The nodes that the call made
    itself are copied. Each child call is one `call_module` node with the arguments of the call.
    The parameters and buffers that the call reads become `get_attr` nodes.

    A child call can write a tensor of the call in place. The call reads that tensor after the
    child call through an `_after` node, which takes the child call node as an argument, so the
    read runs after the write. A node that a call relinked to a tensor reads as that tensor, in
    every graph.

    The call can also write a parameter or buffer in place before a child call whose module holds
    it. The child call then takes its first tensor argument through an `_after` node that takes
    the write as an argument, so the child call runs after the write.
    """

    def __init__(self, exported: torch.export.ExportedProgram) -> None:
        self._graph = exported.graph
        self._live = set(self._graph.nodes)
        placeholders = {node.name: node for node in self._graph.nodes if node.op == "placeholder"}
        state_kinds = (InputKind.PARAMETER, InputKind.BUFFER, InputKind.CONSTANT_TENSOR)
        self._state: dict[fx.Node, str] = {}
        lifted: dict[str, Any] = {}
        for spec in exported.graph_signature.input_specs:
            if spec.kind in state_kinds and spec.target is not None:
                self._state[placeholders[spec.arg.name]] = spec.target
            if spec.kind is InputKind.CONSTANT_TENSOR and spec.target is not None:
                lifted[spec.target] = exported.constants[spec.target]
        self._result = _SplitTrace({}, {}, lifted)

    def split(self, root: _Call) -> _SplitTrace:
        """Build the graph of `root` and of each fold call inside it."""
        self._tensors = {
            **dict(root.module.named_parameters()),
            **dict(root.module.named_buffers()),
        }
        self._relinked: dict[fx.Node, fx.Node] = {}
        calls = [root]
        while calls:
            call = calls.pop()
            self._relinked.update(call.relinked)
            calls.extend(call.children)
        self._check_every_op_has_a_call(root)
        self._add_graph(root)
        return self._result

    def _check_every_op_has_a_call(self, root: _Call) -> None:
        made: set[fx.Node] = set()
        calls = [root]
        while calls:
            call = calls.pop()
            made.update(item for item in call.body if isinstance(item, fx.Node))
            calls.extend(call.children)
        lost = [n.name for n in self._graph.nodes if n.op == "call_function" and n not in made]
        if lost:
            msg = f"No module call made the nodes {lost}, so `trace` cannot place them."
            raise RuntimeError(msg)

    def _written_state(self, node: fx.Node) -> list[fx.Node]:
        """Return the parameters and buffers that `node` writes in place, also through a view."""
        written: list[fx.Node] = []
        for target in _written_nodes(node):
            base: fx.Node | None = target
            while base is not None and base not in self._state:
                base = _aliased_node(base)
            if base is not None:
                written.append(base)
        return written

    def _holds_state(self, module: nn.Module, state: fx.Node) -> bool:
        """Whether `module` or one of its submodules holds the parameter or buffer `state`."""
        tensor = self._tensors.get(self._state[state])
        held = [*module.parameters(), *module.buffers()]
        return tensor is not None and any(other is tensor for other in held)

    def _add_graph(self, call: _Call) -> None:
        graph = self._call_graph(call)
        _fuse_in_place_writes(graph)
        self._result.graphs[call] = graph
        for child in call.children:
            if _is_fold(child, self._live):
                self._add_graph(child)

    def _call_graph(self, call: _Call) -> fx.Graph:
        graph = fx.Graph()
        env: dict[fx.Node, fx.Node] = {}

        def lookup(node: fx.Node) -> fx.Node:
            if node not in env and (target := self._state.get(node)) is not None:
                env[node] = graph.get_attr(target)
            if node not in env and (original := self._relinked.get(node)) is not None:
                env[node] = lookup(original)
            if node in env:
                return env[node]
            msg = f"The call {call.name or 'of the model'!r} reads {node.name!r}, not an input."
            raise NotImplementedError(msg)

        def read_after_writes(child: _Call, child_node: fx.Node) -> None:
            for write in _in_place_writes(child):
                base = _aliased_node(write)
                while base is not None and base not in env and base not in self._state:
                    base = _aliased_node(base)
                if base is None:
                    continue
                after = graph.call_function(_after, (lookup(base), child_node))
                env[base] = after
                if _aliased_node(write) is base:
                    env[write] = after

        def argument(tree: _Tree) -> Any:
            leaves = [lookup(leaf) if isinstance(leaf, fx.Node) else leaf for leaf in tree.leaves]
            if _is_plain_pytree(tree.spec):
                return pytree.tree_unflatten(leaves, tree.spec)
            unflatten_args: tuple[Any, ...] = (leaves, tree.spec)
            return graph.call_function(pytree.tree_unflatten, unflatten_args)

        # The last in-place write of the call to each parameter or buffer, by its node.
        state_writes: dict[fx.Node, fx.Node] = {}

        def after_state_writes(child: _Call, arguments: Any) -> Any:
            writes = [
                write
                for state, write in state_writes.items()
                if self._holds_state(child.module, state)
            ]
            if not writes:
                return arguments
            leaves, spec = pytree.tree_flatten(arguments)
            first = next((i for i, leaf in enumerate(leaves) if isinstance(leaf, fx.Node)), None)
            if first is None:
                msg = (
                    f"The call {child.name!r} reads a parameter or buffer that is written in place "
                    "before it, but it has no tensor argument to order it after the write."
                )
                raise NotImplementedError(msg)
            leaves[first] = graph.call_function(_after, (leaves[first], *writes))
            return pytree.tree_unflatten(leaves, spec)

        for name, tree in zip(_input_names(call), [*call.args, *call.kwargs.values()]):
            _bind_parts(graph, graph.placeholder(name), tree, env)

        for item in call.body:
            if isinstance(item, _Call):
                args = tuple(argument(tree) for tree in item.args)
                kwargs = {key: argument(tree) for key, tree in item.kwargs.items()}
                args, kwargs = after_state_writes(item, (args, kwargs))
                node = graph.call_module(item.name, args, kwargs)
                self._result.calls[node] = item
                assert item.output is not None
                _bind_parts(graph, node, item.output, env)
                read_after_writes(item, node)
            elif item in self._live:
                if item.op != "call_function":
                    msg = f"Unsupported node {item.name!r} with op {item.op!r} in a module call."
                    raise NotImplementedError(msg)
                env[item] = graph.node_copy(item, lookup)
                state_writes.update(dict.fromkeys(self._written_state(item), env[item]))

        # The graph returns the output as the module returns it, so the fold and its module
        # give the same value.
        assert call.output is not None
        graph.output(argument(call.output))
        for node in reversed(graph.nodes):
            if (
                node.target in (operator.getitem, getattr, pytree.tree_leaves, _after)
                and not node.users
            ):
                graph.erase_node(node)
        return graph


def _resolve_pytree_glue(gm: fx.GraphModule) -> None:
    """Remove the pytree glue that `torch.export.unflatten` adds around preserved calls.

    The glue is the `tree_unflatten`, `tree_flatten_spec` and `getitem` nodes with which the
    caller packs all inputs of a preserved call into one tree and unpacks them again, and the
    callee does the same for its inputs and outputs. Each input of the call then depends on all
    of them: the gradient of a decoder layer output flows into the attention mask of the next
    layer.

    The glue only packs and unpacks values, so it is evaluated here instead of copied:
    `getitem(tree_unflatten([a, b], spec), 0)` is `a`. A `getitem` on a node, an item of a
    value whose structure is not known, is kept. Trees containing typed containers
    stay packed at runtime because argument conversion only rebuilds plain containers.

    Args:
        gm: The graph module to rewrite in place.
    """
    graph = fx.Graph()
    tracer = GraphAppendingTracer(graph)
    values: dict[fx.Node, Any] = {}

    def value(node: fx.Node) -> Any:
        return values[node]

    def to_proxy(arg: Any) -> Any:
        return fx.Proxy(arg, tracer) if isinstance(arg, fx.Node) else arg

    def to_node(arg: Any) -> Any:
        return arg.node if isinstance(arg, fx.Proxy) else arg

    for node in gm.graph.nodes:
        args = fx.node.map_arg(node.args, value)
        if node.op == "get_attr" and isinstance(getattr(gm, node.target, None), pytree.TreeSpec):
            values[node] = getattr(gm, node.target)
        elif node.target in (pytree.tree_unflatten, fx_pytree.tree_flatten_spec) and not (
            _is_plain_pytree(args[1])
        ):
            # Keep typed containers out of FX argument mapping and `_GraphBuilder`,
            # which can otherwise erase their types or fail to find their node leaves.
            # `tree_flatten_spec` also cannot flatten a proxy of a type, such as
            # `transformers.ModelOutput`, that has no `fx_pytree` flatten function.
            spec = graph.get_attr(node.args[1].target)
            values[node] = graph.call_function(node.target, (args[0], spec))
        elif node.target in (pytree.tree_unflatten, fx_pytree.tree_flatten_spec):
            # Proxies let `tree_flatten_spec` read an item of a node of unknown structure.
            proxied = fx.node.map_aggregate(args, to_proxy)
            result: Any = fx.node.map_aggregate(node.target(*proxied), to_node)
            if isinstance(result, fx.Node) or all(user.op != "output" for user in node.users):
                values[node] = result
            else:
                # An output that is a container stays packed by its `tree_unflatten`.
                spec = graph.get_attr(node.args[1].target)
                values[node] = graph.call_function(node.target, (args[0], spec))
        elif node.target is operator.getitem and not isinstance(args[0], fx.Node):
            item = args[0][args[1]]
            if isinstance(item, fx.Node) and item.target is operator.getitem and not item.meta:
                item.meta = node.meta
            values[node] = item
        else:
            values[node] = graph.node_copy(node, value)
    for node in reversed(graph.nodes):
        if node.target is operator.getitem and not node.users:
            graph.erase_node(node)
    graph.owning_module = gm
    gm.graph = graph


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
            _resolve_pytree_glue(cast(fx.GraphModule, submodule))
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
                # `unflatten` makes its modules in training mode. The fallback runs in place
                # of the original, so each of its modules takes the mode of the original.
                original_modes = {name: m.training for name, m in original.named_modules()}
                for name, module in submod.named_modules():
                    module.training = original_modes.get(name, original.training)
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
