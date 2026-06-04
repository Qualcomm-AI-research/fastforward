# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
from __future__ import annotations

import collections
import dataclasses
import enum
import functools
import itertools
import uuid

from collections.abc import Collection, Mapping
from contextlib import nullcontext
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ContextManager,
    Iterator,
    Sequence,
    TypeAlias,
)

import torch

if TYPE_CHECKING:
    from fastforward._orchestration.instruction_engine import (
        InstructionEngine,
        InstructionProgram,
        OffloadingStrategy,
    )


@dataclasses.dataclass(frozen=True)
class _BaseRef:
    """Base class for all references - provides attribute access."""

    def __getattr__(self, key: str) -> AttributeRef:
        return AttributeRef(reference=self, attribute=key)

    def __getitem__(self, key: int | str) -> AttributeRef:
        return AttributeRef(reference=self, attribute=key)

    def unwrap_ref(self) -> _BaseRef:
        """Extract the base NodeRef or InputRef."""
        return self


@dataclasses.dataclass(frozen=True)
class NamedRef(_BaseRef):
    """Reference with both an `id` and a `name` for identification."""

    id: uuid.UUID = dataclasses.field(repr=False)
    name: str = dataclasses.field(hash=False, compare=False)


@dataclasses.dataclass(frozen=True)
class NodeRef(NamedRef):
    """Reference to a node inside a GraphModule.

    Args:
        name: Fully-qualified name inside the GraphModule
    """

    def __repr__(self) -> str:
        return f"NodeRef({self.name})"


@dataclasses.dataclass(frozen=True)
class InputRef(NamedRef):
    """Reference to a GraphModule input.

    Args:
        name: Input name of the current GraphModule
    """

    def __repr__(self) -> str:
        return f"InputRef({self.name})"


@dataclasses.dataclass(frozen=True)
class Const(_BaseRef):
    """Literal argument for a node.

    Args:
        value: Any compile-time value we want to hold constant.
    """

    value: Any = dataclasses.field(compare=False, hash=False)
    id: uuid.UUID = dataclasses.field(default_factory=uuid.uuid4, repr=False)

    def __repr__(self) -> str:
        return f"Const({self.value})"


@dataclasses.dataclass(frozen=True)
class AttributeRef(_BaseRef):
    """Reference to an item / attribute of another `Node`.

    If an output of an nn.Module is a tuple, an `AttributeRef` can refer to these with indexing (e.g. output[0]).
    If an output is instead some structured (data)class, then an `AttributeRef` can refer to
    values inside of this class (e.g. output.attention_weights).

    Args:
        reference: The base reference this attribute is attached to
        attribute: The attribute or index being accessed.
    """

    reference: _BaseRef
    attribute: str | int

    def unwrap_ref(self) -> _BaseRef:
        """Recursively unwrap the base NodeRef, InputRef, or Const."""
        if isinstance(self.reference, AttributeRef):
            return self.reference.unwrap_ref()
        return self.reference


def remap_subgraph_reference(
    old_reference: _BaseRef,
    new_context: Mapping[str, _BaseRef],
    subgraph_nodes: dict[str, uuid.UUID],
    scope_name: str = "",
) -> _BaseRef:
    """Remap an old reference for the new context by scoping internal nodes and mapping external ones.

    When adding a GraphModule as subgraph or creating a new subgraph from nodes in an existing
    GraphModule, we need to distinguish between internal nodes (belonging to the subgraph, `subgraph_nodes`)
    and external nodes (existing outside of the subgraph). Internal nodes should be validated
    and prefixed with the new scope name (`name_prefix`). External nodes should be
    re-mapped to inputs via the `input_binding` dict.

    Args:
        old_reference: Reference to remap given the subgraph context.
        new_context: Mapping from external node/input names to new references.
        subgraph_nodes : Subgraph nodes with their IDs. Nodes in this dict stay
                         as NodeRefs; others become inputs.
        scope_name: Optional scope prefix to add to internal node names

    Returns:
        A new reference with updated names and IDs.
    """
    match old_reference:
        # NodeRef is an external input to the subgraph.
        case NodeRef(name=node_name) if node_name not in subgraph_nodes:
            return new_context[node_name]
        # NodeRef is an internal input in the subgraph.
        case NodeRef(id=ref_id, name=node_name):
            new_name = f"{scope_name}.{node_name}" if scope_name else node_name
            return NodeRef(ref_id, new_name)
        case AttributeRef(reference=ref, attribute=attr):
            if isinstance(base := ref.unwrap_ref(), (NodeRef, InputRef)):
                # Node/Input outside of subgraph that only requires attribute access
                # (e.g. we do not need to remap 'base' only 'base[attr]').
                # Here it is expected that 'base' is not present in context but 'base[attr]' is.
                lifted_key = f"{base.name}[{attr}]"
                is_external_node = isinstance(base, NodeRef) and base.name not in subgraph_nodes
                is_external_input = isinstance(base, InputRef)
                if (is_external_node or is_external_input) and lifted_key in new_context:
                    return new_context[lifted_key]

            # AttributeRef wraps another reference, so remap the wrapped reference.
            remapped_ref = remap_subgraph_reference(ref, new_context, subgraph_nodes, scope_name)
            return dataclasses.replace(old_reference, reference=remapped_ref)
        case InputRef(name=input_name):
            return new_context[input_name]
        case Const():
            return old_reference
        case _BaseRef:
            assert False, f"Unexpected reference type {type(old_reference)}"


# Contexts should be ordered to match delegate function signature.
Contexts: TypeAlias = Sequence[ContextManager[None]]


@dataclasses.dataclass(frozen=True)
class Delegate:
    """Defines a function to execute on a node and the contexts needed to generate its inputs.

    Args:
        fn: Function that receives the module and input activations as positional arguments.
        contexts: Sequence of ContextManagers that generate input activations.
    """

    fn: Callable[..., None]
    contexts: Contexts


class Op(enum.Enum):
    """Operation type for node dispatching.

    Each node has an optional module. this can be a function, a method, a torch module, or
    something that gets an attribute from another Node.
    """

    torch_module = enum.auto()
    call_function = enum.auto()
    call_method = enum.auto()
    get_attr = enum.auto()


@dataclasses.dataclass(frozen=True)
class Node:
    """Single operation in a static call graph.

    A Node declares its dependencies through symbolic references
    (`args`/`kwargs`) and its behavior through `op`:

    - `torch_module`: calls an nn.Module
    - `call_function`: calls a standalone function (e.g. F.scaled_dot_product_attention)
    - `call_method`: calls a method on a tensor
    - `get_attr`: retrieves a module attribute (e.g. a buffer)

    Nodes form a hierarchy via `parent`. A **leaf node** is an executable
    operation with no nested structure. A **fold** marks an enclosing region
    (e.g. an entire attention block) — its `target` is the original nn.Module,
    callable as a shortcut for the whole subtree. Leaves point to their
    enclosing fold via `parent`; folds themselves have no data-flow edges.
    A coarser resolution is obtained by leaving folds intact; a finer
    resolution is obtained by unfolding them into their children.

    A node may optionally carry a `delegate`: an optimization function with
    context managers that supply calibration data during quantization.

    Args:
        id: Unique identifier within the graph.
        name: Human-readable name within the graph (e.g. `"layer_0.attn.q_proj"`).
        target: The thing this node calls or accesses — an nn.Module, function,
            method, or attribute owner, depending on `op`.
        args: Positional dependencies as symbolic references (NodeRef, InputRef,
            or Const), resolved to actual values at execution time.
        op: Selects how `target` is invoked (see above).
        kwargs: Keyword dependencies, same reference types as `args`.
        delegate: Optional optimization function with calibration contexts,
            invoked instead of `target` during local optimization.
        parent: Enclosing fold, or None for top-level nodes.
    """

    id: uuid.UUID
    name: str
    target: torch.nn.Module | Callable[..., Any]
    args: Collection[_BaseRef]
    op: Op = Op.torch_module
    kwargs: Mapping[str, _BaseRef] = dataclasses.field(default_factory=dict)
    delegate: Delegate | None = None
    parent: NodeRef | None = None


def _sanitize_inputs(
    input_names: Sequence[str],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Project user-supplied inputs onto a graph's named-input contract.

    The engine binds values to inputs either positionally (in `input_names` order) or
    by keyword. Users often have data in a more natural shape — a single iterable
    yielding `dict`, `tuple`, or dataclass-like batches. This helper recognizes those
    shapes and rewrites them to per-input lists the engine already accepts. All other
    shapes (a tensor, parallel kwarg lists, declared-arity positional args) pass
    through unchanged.

    Recognized shape (single positional iterable, no kwargs):
      - element is a `Mapping` containing `input_names` as keys
      - element is a tuple/list of length `len(input_names)`
      - element exposes `input_names` as attributes (dataclass, namedtuple, ...)
    """
    if len(args) != 1 or kwargs:
        return args, kwargs

    (single,) = args
    if isinstance(single, torch.Tensor):
        return args, kwargs

    try:
        iterator = iter(single)
    except TypeError:
        return args, kwargs

    try:
        first = next(iterator)
    except StopIteration:
        return args, kwargs

    batches = [first, *iterator]
    names = tuple(input_names)

    if isinstance(first, Mapping) and all(name in first for name in names):
        return (), {name: [b[name] for b in batches] for name in names}

    if isinstance(first, (list, tuple)) and len(first) == len(names):
        return (), {name: [b[i] for b in batches] for i, name in enumerate(names)}

    if not isinstance(first, torch.Tensor) and all(hasattr(first, name) for name in names):
        return (), {name: [getattr(b, name) for b in batches] for name in names}

    # Iterable of values that don't match any known per-batch shape — leave as a
    # single positional arg so the engine treats it as N batches of one input.
    return (batches,), {}


class GraphModule(torch.nn.Module):
    """Multi-resolution static call graph over PyTorch modules.

    A GraphModule represents a model as an explicit DAG of Nodes at multiple
    resolutions. Folds represent coarser regions (e.g. a full transformer
    layer) that can be unfolded into their children (e.g. individual linear
    projections) for finer-grained execution.

    Take as an example a simplified transformer decoder layer with three
    resolutions::

        (1)  ┌──────────────── decoder_layer ────────────────┐
        (2)   ┌──── attn ──────────┐       ┌───── mlp ──────┐
        (3)    q → k → v → sdpa → o → norm → gate → up → down

    The first resolution is the coarsest: calling the decoder layer directly
    (`decoder_layer(x)`). The second traverses the direct children of
    decoder_layer: `mlp(norm(attn(x)))`. The third traverses every leaf node
    in the graph. These are all valid execution strategies for the same model.

    Resolutions can also be mixed — for instance, unfolding attn into its
    leaves while keeping mlp folded::

        (mix) q → k → v → sdpa → o → norm → mlp

    Runtime behavior (how a forward pass is actually executed) depends on how
    the user schedules instructions for the InstructionEngine. By default,
    no unfolding takes place, and the coarsest resolution is used (`decoder_layer(x)`).

    Example::

        graph = GraphModule()
        x = graph.add_input("x")

        # Leaf-level construction (resolution 3):
        q = graph.add_node("attn.q", nn.Linear(64, 64), [x])
        k = graph.add_node("attn.k", nn.Linear(64, 64), [x])
        v = graph.add_node("attn.v", nn.Linear(64, 64), [x])
        sdpa = graph.add_node("attn.sdpa", F.scaled_dot_product_attention, [q, k, v])
        graph.add_output(sdpa)

        # Or compose pre-built subgraphs (resolution 2):
        attn_out, = graph.add_subgraph("attn", attn_graph, [x], original_module=attn)

        # Executes like a normal nn.Module:
        assert torch.allclose(graph(input_tensor), attn(input_tensor))
    """

    def __init__(self) -> None:
        super().__init__()
        self._nodes: dict[uuid.UUID, Node] = {}
        self._node_refs: dict[uuid.UUID, NodeRef] = {}
        self._inputs: dict[str, InputRef] = dict()
        self._outputs: list[NodeRef | AttributeRef] = []
        self._fold_ids: set[uuid.UUID] = set()

        # Binding represents fold "args", in the order its forward() expects them.
        self._fold_bindings: dict[uuid.UUID, tuple[_BaseRef, ...]] = {}

        self._program: Any = None
        self._engine: InstructionEngine | None = None

    @property
    def input_names(self) -> list[str]:
        """Return the list of graph input names in definition order."""
        return list(self._inputs.keys())

    def node_inputs(self, node_ref: NodeRef) -> Iterator[NodeRef]:
        """Return nodes that are inputs to `node_ref`.

        Args:
            node_ref: Node reference to get inputs for. This must be a valid node in the graph.

        Returns:
            Iterator yielding NodeRef objects for each input node.
        """
        node = self._nodes[node_ref.id]
        seen: set[uuid.UUID] = set()
        for arg in (*node.args, *node.kwargs.values()):
            if isinstance(arg_base := arg.unwrap_ref(), NodeRef):
                if arg_base.id not in seen:
                    seen.add(arg_base.id)
                    yield arg_base

    def node_outputs(self, node_ref: NodeRef) -> Iterator[NodeRef]:
        """Return nodes that use `node_ref` as input.

        Args:
            node_ref: Node reference to get outputs for. This must be a valid node in the graph.

        Returns:
            Iterator yielding NodeRef objects for each output node.
        """
        node = self._nodes[node_ref.id]
        seen: set[uuid.UUID] = set()
        for other_node in self._nodes.values():
            for arg in (*other_node.args, *other_node.kwargs.values()):
                if isinstance(arg_base := arg.unwrap_ref(), NodeRef) and arg_base.id == node.id:
                    if other_node.id not in seen:
                        seen.add(other_node.id)
                        yield self._node_refs[other_node.id]
                    break

    def node_ref(self, module: torch.nn.Module) -> NodeRef:
        """Return the NodeRef for the given module instance.

        Args:
            module: The torch.nn.Module instance to find in the graph.

        Returns:
            NodeRef for the first node containing this module instance.

        Raises:
            ValueError: If the module is not found in the graph.

        Note:
            If the same module instance is added multiple times to the graph,
            this method will only return the first matching NodeRef.
        """
        for id, node in self._nodes.items():
            if node.target is module:
                return NodeRef(id=id, name=node.name)

        msg = f"Module {module} not found in Graphmodule."
        raise ValueError(msg)

    def leaf_nodes(self, fold_ref: NodeRef) -> set[NodeRef]:
        """Return leaf (non-fold) nodes under a fold."""
        result: set[NodeRef] = set()
        stack = [
            self._node_refs[nid] for nid, node in self._nodes.items() if node.parent == fold_ref
        ]

        while stack:
            if (child := stack.pop()) in result:
                continue

            result.add(child)
            stack.extend(
                self._node_refs[nid] for nid, node in self._nodes.items() if node.parent == child
            )
        return {r for r in result if r.id not in self._fold_ids}

    def fold_outputs(self, fold_ref: NodeRef) -> list[NodeRef]:
        """Return leaf nodes inside the fold whose output is used externally or as graph output."""
        fold_leaf_ids = {n.id for n in self.leaf_nodes(fold_ref)}
        internal = fold_leaf_ids | self._fold_ids
        output_ids = {
            r.unwrap_ref().id for r in self._outputs if isinstance(r.unwrap_ref(), NodeRef)
        }

        outputs: list[NodeRef] = []
        for nid in self._nodes:
            if nid not in fold_leaf_ids:
                continue
            leaf_ref = self._node_refs[nid]
            feeds_external = any(c.id not in internal for c in self.node_outputs(leaf_ref))
            is_graph_output = nid in output_ids
            if feeds_external or is_graph_output:
                outputs.append(leaf_ref)
        return outputs

    def node(self, node_ref: NodeRef) -> Node:
        """Return the Node for the given NodeRef.

        Args:
            node_ref: Reference to the node.

        Returns:
            The Node object.

        Raises:
            KeyError: If the node reference is not found in the graph.
        """
        return self._nodes[node_ref.id]

    def add_input(self, name: str) -> InputRef:
        """Add an input name to the graph."""
        if name in self._inputs:
            msg = f"Duplicate input name: {name}"
            raise ValueError(msg)
        ref = InputRef(uuid.uuid4(), name)
        self._inputs[name] = ref
        return ref

    def add_output(self, *nodes: NodeRef | AttributeRef) -> None:
        """Add a sequence of NodeRef / AttributeRef to the graph's output.

        A NodeRef is eligible if it is contained inside the GraphModule, an
        AttributeRef is eligible if it resolves to a NodeRef inside the GraphModule.

        Args:
            nodes: Sequence of NodeRef or AttributeRef to be added as outputs.
        """
        for node in nodes:
            match base_ref := node.unwrap_ref():
                case NodeRef(id=node_id, name=node_name):
                    if node_id not in self._nodes:
                        msg = f"Unknown node name: {node_name}"
                        raise ValueError(msg)

                    # Add the original node (not the resolved value) to the outputs.
                    self._outputs.append(node)

                case _:
                    msg = f"An AttributeRef has to resolve to a NodeRef for outputs, found {type(base_ref)}"
                    raise ValueError(msg)

    def add_node(
        self,
        name: str,
        target: torch.nn.Module | Callable[..., Any],
        args: Collection[_BaseRef],
        kwargs: Mapping[str, _BaseRef] | None = None,
        *,
        op: Op = Op.torch_module,
        node_id: uuid.UUID | None = None,
        parent: NodeRef | None = None,
    ) -> NodeRef:
        """Add a node to this 'GraphModule'.

        Creates a new Node wrapping the given target and adds it to the graph. The node
        can reference other nodes' outputs, external inputs, or constant values through
        its arguments.

        Args:
            name: Unique identifier of the node within its fold
            target: Callable to execute (nn.Module, function, method wrapper, etc.)
            args: Positional arguments (NodeRef/InputRef/Const)
            kwargs: Keyword arguments (NodeRef/InputRef/Const)
            op: Discriminator for the kind of callable. Defaults to Op.torch_module.
            node_id: Optional node ID. If not provided, a new one will be generated.
            parent: Optional fold that this node belongs to in the hierarchy.

        Returns:
            NodeRef: Reference to the created node

        Raises:
            ValueError: If node already exists in graph
        """
        node_id = node_id or uuid.uuid4()
        if node_id in self._nodes:
            msg = f"Duplicate node id: {node_id}"
            raise ValueError(msg)

        # Reset execution engine if it existed since the graph will be altered.
        self._engine = None
        node = Node(node_id, name, target, args, op=op, kwargs=kwargs or {}, parent=parent)
        self._nodes[node_id] = node
        ref = NodeRef(node_id, name)
        self._node_refs[node_id] = ref

        if isinstance(node.target, torch.nn.Module):
            *module_path, leaf_name = name.split(".")
            parent_module: torch.nn.Module = self
            for attr in module_path:
                if not hasattr(parent_module, attr):
                    parent_module.add_module(attr, torch.nn.Module())
                parent_module = getattr(parent_module, attr)

            # Don't overwrite children of original modules registered as folds —
            # those belong to the user's model and must not be mutated.
            is_fold_original = any(
                parent_module is self._nodes[fid].target for fid in self._fold_ids
            )
            if not is_fold_original:
                parent_module.add_module(leaf_name, node.target)

        return ref

    def add_subgraph(
        self,
        name: str,
        subgraph: GraphModule,
        args: Collection[_BaseRef],
        kwargs: Mapping[str, _BaseRef] | None = None,
        *,
        original_module: torch.nn.Module | None = None,
    ) -> tuple[NodeRef | AttributeRef, ...]:
        """Inline a subgraph into this 'GraphModule' as a folded collection of nodes.

        When `original_module` is provided, a fold (parent) node is created that
        represents the original module in the hierarchy. Inlined leaves get their
        `parent` field set to this fold. The fold has no data-flow edges — it is
        a hierarchy marker only. Coarse execution of the fold is handled as a
        scheduler transform.

        Args:
            name: Fold name for the inlined subgraph
            subgraph: GraphModule to inline into this graph
            args: Positional arguments (NodeRef/InputRef/Const)
            kwargs: Keyword arguments (NodeRef/InputRef/Const). Kwargs are allowed but are
                    expected to be inputs to the subgraph.
            original_module: When provided, creates a fold holding the original
                nn.Module for coarse execution.

        Returns:
            A tuple of node references (NodeRef) to the output node(s) of the inlined sub-graph

        Raises:
            ValueError: If subgraph has no output nodes defined
            ValueError: If number of args and kwargs does not match subgraph's input count
            ValueError: If a keyword argument is not an input of the subgraph
            ValueError: If a keyword argument is already defined as an argument
        """
        if not subgraph._outputs:
            msg = "Subgraph has no output nodes defined"
            raise ValueError(msg)

        # Reset execution engine if it existed since the graph will be altered.
        self._engine = None

        kwargs = kwargs or {}

        if len(args) + len(kwargs) != len(subgraph.input_names):
            msg = (
                f"Subgraph expects {len(subgraph.input_names)} inputs {subgraph.input_names}"
                f"got {len(args)} positional and {len(kwargs)} keyword args"
            )
            raise ValueError(msg)

        # Map old subgraph input names and kwargs -> new NodeRef
        input_binding = dict(zip(subgraph.input_names, args))

        for key, value in kwargs.items():
            if key not in subgraph.input_names:
                msg = f"Subgraph does not have input '{key}'"
                raise ValueError(msg)
            if key in input_binding:
                msg = f"Subgraph input has duplicate binding for '{key}'"
                raise ValueError(msg)
            input_binding[key] = value

        # Create the fold (parent) node — a hierarchy marker whose args/kwargs
        # mirror the original module's call convention for coarse execution.
        fold_ref: NodeRef | None = None
        if original_module is not None:
            fold_id = uuid.uuid4()
            fold_kwargs = {k: input_binding[k] for k in (kwargs or {})}
            fold_args = tuple(
                input_binding[n] for n in subgraph.input_names if n not in fold_kwargs
            )
            fold_node = Node(fold_id, name, original_module, args=fold_args, kwargs=fold_kwargs)
            self._nodes[fold_id] = fold_node
            fold_ref = NodeRef(fold_id, name)
            self._node_refs[fold_id] = fold_ref
            self._fold_ids.add(fold_id)
            self._fold_bindings[fold_id] = tuple(input_binding[n] for n in subgraph.input_names)

        remap_reference = functools.partial(
            remap_subgraph_reference,
            new_context=input_binding,
            subgraph_nodes={
                node_ref.name: node_ref.id for node_ref in subgraph._node_refs.values()
            },
            scope_name=name,
        )

        # Build a mapping from old subgraph fold refs to newly created (folded) refs.
        old_to_new_fold: dict[uuid.UUID, NodeRef] = {}

        # Process folds first so leaf nodes can reference them via parent.
        sorted_nodes = sorted(
            subgraph._nodes.values(),
            key=lambda n: 0 if n.id in subgraph._fold_ids else 1,
        )

        for node in sorted_nodes:
            node_args = [remap_reference(old_reference=arg) for arg in node.args]
            node_kwargs = {
                key: remap_reference(old_reference=arg) for key, arg in node.kwargs.items()
            }
            node_name = f"{name}.{node.name}"

            node_parent = (
                old_to_new_fold.get(node.parent.id, node.parent) if node.parent else fold_ref
            )

            ref = self.add_node(
                node_name,
                node.target,
                node_args,
                node_kwargs,
                op=node.op,
                node_id=node.id,
                parent=node_parent,
            )
            if node.id in subgraph._fold_ids:
                self._fold_ids.add(node.id)
                old_to_new_fold[node.id] = ref
                # The binding lives on the parent graph, so we have to copy it across manually.
                if (inner_binding := subgraph._fold_bindings.get(node.id)) is not None:
                    self._fold_bindings[node.id] = tuple(
                        remap_reference(old_reference=b) for b in inner_binding
                    )

        # Return the in-lined subgraph outputs in stored order to preserve positional semantics.
        new_output_refs = []
        for out_ref in subgraph._outputs:
            remapped = remap_reference(old_reference=out_ref)
            assert isinstance(remapped, (NodeRef, AttributeRef))
            new_output_refs.append(remapped)

        return tuple(new_output_refs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward inputs through the GraphModule.

        An internal execution engine is created on the first run, which executes nodes
        in a topologically sorted order.

        Args:
            *args: Positional inputs for the GraphModule, corresponding to the names defined with
                `add_input`.
            **kwargs: Key-word inputs for the GraphModule, corresponding to the names defined with
                `add_input`.

        Returns:
            Output tensor(s) from the graph's output nodes
        """
        args, kwargs = _sanitize_inputs(self.input_names, args, kwargs)

        if self._engine is None:
            from fastforward._orchestration.instruction_engine import (
                InstructionEngine,
                InstructionScheduler,
            )

            scheduler = InstructionScheduler()
            self._program = scheduler.schedule(self)
            self._engine = InstructionEngine()

        # The engine produces per-context results.  When the program has no
        # ReturnOutputs instruction (e.g. optimization-only schedules) the
        # engine returns None.
        outputs: dict[ContextManager[None], tuple[Any]] | None = self._engine.run(
            self._program, *args, **kwargs
        )

        if outputs is None:
            return None

        # If the output is a single value (non-batched) and produced by a single
        # context, we return directly that value, mimicing the torch module behavior.
        if len(outputs) == 1 and len(results := next(iter(outputs.values()))) == 1:
            return results[0]

        # Otherwise, return the dictionary itself.
        return outputs


def find_nodes_on_path(graph: GraphModule, start: NodeRef, end: NodeRef) -> set[NodeRef]:
    """Return the set of nodes on the path from `start` to `end` (inclusive).

    Args:
        graph: GraphModule to analyze.
        start: Reference to the first node on the path.
        end: Reference to the last node on the path.

    Returns:
        A set with the NodeRefs of all nodes that participate in at least one
        direct dependency path from `start` to `end`.

    Raises:
        ValueError: If either node is missing or if no path exists.
    """
    if start not in graph._node_refs.values():
        msg = f"Start node '{start.name}' not found in graph"
        raise ValueError(msg)

    if end not in graph._node_refs.values():
        msg = f"End node '{end.name}' not found in graph"
        raise ValueError(msg)

    nodes_on_path: set[NodeRef] = set()
    visited: set[NodeRef] = set()

    def _can_reach_end(current: NodeRef, target: NodeRef, visited: set[NodeRef]) -> bool:
        """Depth-First recursive search that returns True if `target` is reachable from `current`."""
        if current == target:
            nodes_on_path.add(current)
            return True

        if current in visited:
            return False

        visited.add(current)
        found_path = False

        for dependent in graph.node_outputs(current):
            if _can_reach_end(dependent, target, visited):
                found_path = True

        if found_path:
            nodes_on_path.add(current)

        visited.remove(current)
        return found_path

    if not _can_reach_end(start, end, visited):
        msg = f"Node {start.name} is not reachable from {end.name}"
        raise ValueError(msg)

    return nodes_on_path


def create_subgraph(graph: GraphModule, path_nodes: set[NodeRef]) -> GraphModule:
    """Build and return a standalone GraphModule that contains exactly `path_nodes`.

    Args:
        graph: GraphModule to extract subgraph from.
        path_nodes: Names of nodes that must appear in the subgraph.

    Returns:
        A new GraphModule representing the extracted sub-graph.

    Raises:
        ValueError: If proposed subgraph has no external inputs.
    """
    node_ids: set[uuid.UUID] = {ref.id for ref in path_nodes}

    def _external_dependencies(node: Node) -> list[str]:
        external_dependencies = []
        for arg in (*node.args, *node.kwargs.values()):
            base = arg.unwrap_ref()
            is_external = isinstance(base, InputRef) or (
                isinstance(base, NodeRef) and base.id not in node_ids
            )
            if not is_external:
                continue

            if isinstance(arg, AttributeRef):
                external_dependencies.append(f"{base.name}[{arg.attribute}]")
            else:
                external_dependencies.append(str(base.name))

        return external_dependencies

    external_inputs = []
    seen = set()
    for node in path_nodes:
        for external_dependency in _external_dependencies(graph._nodes[node.id]):
            if external_dependency not in seen:
                external_inputs.append(external_dependency)
                seen.add(external_dependency)

    # Determine which nodes must be exposed as subgraph outputs.
    # These are either (1) nodes that are outside of the original graph or
    # (2) nodes required by external nodes outside of the subgraph.
    external_outputs = []
    seen_outputs: set[NodeRef | AttributeRef] = set()

    # (1) Include original graph outputs that are produced within this subgraph.
    for ref in graph._outputs:
        if (base := ref.unwrap_ref()) in path_nodes:
            if ref not in seen_outputs:
                external_outputs.append(ref)
                seen_outputs.add(ref)

    # (2) Include internal nodes that are used by nodes outside the subgraph.
    for internal_ref in path_nodes:
        for dependent_ref in graph.node_outputs(internal_ref):
            if dependent_ref in path_nodes:
                continue

            # External nodes (outside the subgraph) that depend on internal nodes (inside the subgraph).
            external_node = graph._nodes[dependent_ref.id]
            for arg in itertools.chain(external_node.args, external_node.kwargs.values()):
                base = arg.unwrap_ref()
                if isinstance(base, NodeRef) and base.id == internal_ref.id:
                    if base not in seen_outputs:
                        external_outputs.append(base)
                        seen_outputs.add(base)

    # Connect the inputs, nodes, and outputs to a new subgraph.
    subgraph = GraphModule()

    remap_reference = functools.partial(
        remap_subgraph_reference,
        new_context={name: subgraph.add_input(name) for name in external_inputs},
        subgraph_nodes={ref.name: ref.id for ref in path_nodes},
    )

    for node_id in node_ids:
        source_node = graph._nodes[node_id]
        subgraph.add_node(
            name=source_node.name,
            target=source_node.target,
            args=[remap_reference(old_reference=arg) for arg in source_node.args],
            kwargs={
                key: remap_reference(old_reference=arg) for key, arg in source_node.kwargs.items()
            },
            op=source_node.op,
            node_id=source_node.id,
        )

    remapped_outputs: list[NodeRef | AttributeRef] = []
    for ref in external_outputs:
        remapped = remap_reference(old_reference=ref)
        assert isinstance(remapped, (NodeRef, AttributeRef))
        remapped_outputs.append(remapped)
    subgraph.add_output(*remapped_outputs)

    return subgraph


def topological_sort(graph: GraphModule) -> list[NodeRef]:
    """Return leaf (non-fold) nodes from a GraphModule in topological order.

    Raises:
        ValueError: If graph contains circular dependencies among leaf nodes.
    """
    leaf_refs = {nid: ref for nid, ref in graph._node_refs.items() if nid not in graph._fold_ids}

    order = []
    in_degree = {
        node_ref: len(list(graph.node_inputs(node_ref))) for node_ref in leaf_refs.values()
    }

    queue = collections.deque([name for name, degree in in_degree.items() if degree == 0])

    while queue:
        current = queue.popleft()
        order.append(current)

        for dependent in graph.node_outputs(current):
            if dependent.id in leaf_refs:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

    if len(order) != len(leaf_refs):
        msg = "Circular dependency detected in graph!"
        raise ValueError(msg)

    return order


DEFAULT_CONTEXT = nullcontext()


@dataclasses.dataclass
class SubgraphSpec:
    """A specification for targeting a subgraph with a function.

    When the `input` and `output` modules of a GraphModule are selected, we can create a
    subgraph that includes all layers on the path (inclusive).

    Args:
        input: The start module of the subgraph.
        output: The end module of the subgraph.
        fn: Function to execute on the subgraph.
        contexts: Sequence of ContextManagers that generate input activations. If
            not specified, inputs are computed in a single default execution context.
    """

    input: torch.nn.Module
    output: torch.nn.Module

    fn: dataclasses.InitVar[Callable[..., None]]
    contexts: dataclasses.InitVar[Contexts | None] = None
    delegate: Delegate = dataclasses.field(init=False)

    def __post_init__(self, fn: Callable[..., None], contexts: Contexts | None = None) -> None:
        self.delegate = Delegate(fn, contexts or (DEFAULT_CONTEXT,))


OutputNode = NodeRef | GraphModule


def node_order(graph: GraphModule, nodes: list[OutputNode]) -> list[OutputNode]:
    """Sort nodes into a topologically-valid sequence relative to `graph`."""
    topo = {ref: i for i, ref in enumerate(topological_sort(graph))}

    def _sort_key(node: OutputNode) -> int:
        match node:
            case NodeRef() if node.id in graph._fold_ids:
                leaves = graph.fold_outputs(node)
                return max(topo[leaf] for leaf in leaves) if leaves else 0
            case NodeRef():
                return topo[node]
            case GraphModule() as subgraph:
                bases = [ref.unwrap_ref() for ref in subgraph._outputs]
                return max(
                    topo[graph._node_refs[base.id]] for base in bases if isinstance(base, NodeRef)
                )

    return sorted(nodes, key=_sort_key)


def _coarse_complement(graph: GraphModule, targets: set[NodeRef]) -> list[NodeRef]:
    """Return the maximally-coarse set of nodes that fills in everything not in `targets`."""
    target_ids = {ref.id for ref in targets}

    def has_target_descendant(fold_id: uuid.UUID) -> bool:
        for nid, node in graph._nodes.items():
            if nid in target_ids:
                ancestor = node.parent
                while ancestor is not None:
                    if ancestor.id == fold_id:
                        return True
                    ancestor = graph._nodes[ancestor.id].parent
        return False

    complement: list[NodeRef] = []

    def collect(fold_id: uuid.UUID | None = None) -> None:
        children = [
            nid
            for nid, node in graph._nodes.items()
            if (node.parent.id if node.parent is not None else None) == fold_id
        ]
        for child_id in children:
            if child_id in target_ids:
                continue
            if child_id in graph._fold_ids and has_target_descendant(child_id):
                collect(child_id)
            else:
                complement.append(graph._node_refs[child_id])

    collect()
    return complement


def _resolve_subgraph_input(name: str, context: dict[str, _BaseRef]) -> _BaseRef:
    """Look up a subgraph input name (possibly ``'name[attr]'``) in the context."""
    base, sep, rest = name.partition("[")
    if not sep or not rest.endswith("]"):
        return context[name]
    inner = rest[:-1]
    base_ref = context[base]
    try:
        return base_ref[int(inner)]
    except ValueError:
        attr_ref: _BaseRef = getattr(base_ref, inner)
        return attr_ref


def reduce_resolution(graph: GraphModule, specs: list[SubgraphSpec]) -> GraphModule:
    """Reduce the multi-resolution `graph` to the minimal resolution required by `specs`.

    Specs define functions to run on subgraphs. These subgraphs in turn define the most
    granular resolution we need to traverse through the multi-resolution graph. Here we
    extract this resolution, and return a new GraphModule specifically created to run the most
    efficient path through the graph given the subgraph specs.
    """
    targets: set[NodeRef] = set()
    subgraphs: list[tuple[set[NodeRef], GraphModule]] = []
    seen_targets: set[NodeRef] = set()

    # Create "targets": the most granular collection of nodes we need to have, given the specs.
    for spec in specs:
        in_ref, out_ref = graph.node_ref(spec.input), graph.node_ref(spec.output)

        if spec.input is not spec.output:
            path_nodes = find_nodes_on_path(graph, in_ref, out_ref)
            if overlap := path_nodes & seen_targets:
                msg = f"Overlapping nodes in subgraph specs: {overlap}"
                raise ValueError(msg)
            seen_targets |= path_nodes
            targets |= path_nodes
            subgraphs.append((path_nodes, create_subgraph(graph, path_nodes)))
        else:
            if in_ref in seen_targets:
                msg = f"Overlapping nodes in subgraph specs: {{{in_ref}}}"
                raise ValueError(msg)
            seen_targets.add(in_ref)
            targets.add(in_ref)

    # Now create the coarse complement. whatever we dont see as targets, try and get the
    # coarsest level (try not to unfold into leaves if not needed).
    complement = _coarse_complement(graph, targets)

    for path_nodes, _ in subgraphs:
        targets -= path_nodes

    new_graph = GraphModule()
    context: dict[str, _BaseRef] = {name: new_graph.add_input(name) for name in graph.input_names}
    subgraph_idx = 0

    # Rewire the new graph's path in a dependency-based (topo) order.
    for node in node_order(graph, [*targets, *complement, *[gm for _, gm in subgraphs]]):
        match node:
            case GraphModule() as subgraph:
                args = [_resolve_subgraph_input(name, context) for name in subgraph.input_names]
                # Path-specific subgraphs have _no_ kwargs, that wiring is done by input name only.
                ref = new_graph.add_node(f"subgraph_{subgraph_idx}", subgraph, args, kwargs={})
                subgraph_idx += 1

                for i, out in enumerate(subgraph._outputs):
                    if isinstance(base := out.unwrap_ref(), NodeRef):
                        context[base.name] = ref if len(subgraph._outputs) == 1 else ref[i]

            case NodeRef(id=node_id, name=node_name) if node_id in graph._fold_ids:
                original_node = graph._nodes[node_id]
                args = [remap_subgraph_reference(arg, context, {}) for arg in original_node.args]
                kwargs = {
                    key: remap_subgraph_reference(value, context, {})
                    for key, value in original_node.kwargs.items()
                }
                ref = new_graph.add_node(
                    node_name, original_node.target, args, kwargs=kwargs, op=original_node.op
                )
                context[node_name] = ref

                fold_outputs = graph.fold_outputs(node)
                for i, leaf in enumerate(fold_outputs):
                    context[leaf.name] = ref if len(fold_outputs) == 1 else ref[i]

            case NodeRef(id=node_id, name=node_name):
                original_node = graph._nodes[node_id]
                args = [remap_subgraph_reference(arg, context, {}) for arg in original_node.args]
                kwargs = {
                    key: remap_subgraph_reference(value, context, {})
                    for key, value in original_node.kwargs.items()
                }
                node_ref = new_graph.add_node(
                    node_name, original_node.target, args, kwargs, op=original_node.op
                )
                context[node_name] = node_ref

    for graph_output in graph._outputs:
        rewritten = remap_subgraph_reference(graph_output, context, {})
        assert isinstance(rewritten, (NodeRef, AttributeRef))
        new_graph.add_output(rewritten)

    # Finally, add the delegate to whatever node in the new graph had it designated. For subgraphs
    # we specifically use the spec.output as ref, for other specs spec.output is spec.input.
    for spec in specs:
        target_ref = context[graph.node_ref(spec.output).name]
        base_ref = target_ref if isinstance(target_ref, NodeRef) else target_ref.unwrap_ref()
        assert isinstance(base_ref, NodeRef)
        new_graph._nodes[base_ref.id] = dataclasses.replace(
            new_graph._nodes[base_ref.id], delegate=spec.delegate
        )

    return new_graph


class _GraphExecutionContext:
    """Context manager that temporarily swaps a graph's execution program and engine.

    On enter, the graph's `_program` and `_engine` are replaced with the provided
    values. On exit, the original state is restored.

    Args:
        graph: The GraphModule whose execution state will be temporarily replaced.
        program: The program to install for the duration of the context.
        engine: The engine to install for the duration of the context.
    """

    def __init__(
        self,
        graph: GraphModule,
        program: InstructionProgram,
        engine: InstructionEngine,
    ) -> None:
        self._graph = graph
        self._original_program = graph._program
        self._original_engine = graph._engine
        graph._program = program
        graph._engine = engine

    def __enter__(self) -> GraphModule:
        return self._graph

    def __exit__(self, *args: Any) -> None:
        self._graph._program = self._original_program
        self._graph._engine = self._original_engine


def inference_mode(
    graph: GraphModule,
    offloading_strategy: OffloadingStrategy | None = None,
) -> _GraphExecutionContext:
    """Context manager that configures a graph for inference with optional offloading.

    Temporarily replaces the graph's execution program with one that includes lifetime
    management and optional device offloading. The original state is restored on exit.

    Args:
        graph: The GraphModule to configure for inference.
        offloading_strategy: Optional strategy for moving weights and activations between
            devices during execution.

    Returns:
        A context manager that yields the configured graph.
    """
    from fastforward._orchestration.instruction_engine import (
        InstructionEngine,
        InstructionPass,
        InstructionScheduler,
        lifetime_management_pass,
    )

    passes: list[InstructionPass] = [lifetime_management_pass]
    if offloading_strategy is not None:
        passes.append(offloading_strategy.create_instruction_pass(graph))

    program = InstructionScheduler(passes=passes).schedule(graph)

    class _InferenceEngine(InstructionEngine):
        @torch.inference_mode()
        def run(self, program: Any, *args: Any, **kwargs: Any) -> Any:
            return super().run(program, *args, **kwargs)

    return _GraphExecutionContext(graph, program, _InferenceEngine())


def local_optimize(
    graph: GraphModule,
    specs: list[SubgraphSpec],
    offloading_strategy: OffloadingStrategy | None = None,
) -> _GraphExecutionContext:
    """Context manager that configures a graph for local optimization.

    Builds a composite graph from the provided specs, schedules it with lifetime
    management passes, and temporarily installs the resulting program on the original
    graph. The original state is restored on exit.

    Args:
        graph: The GraphModule to optimize.
        specs: Partition boundaries with optional optimization functions. Each spec
            defines input/output nodes and an optional function to optimize that subgraph.
        offloading_strategy: Optional strategy for moving weights and activations between
            devices during execution.

    Returns:
        A context manager that yields the configured graph.
    """
    from fastforward._orchestration.instruction_engine import (
        InstructionEngine,
        InstructionPass,
        InstructionScheduler,
        lifetime_management_pass,
        optimization_only_pass,
    )

    passes: list[InstructionPass] = [optimization_only_pass, lifetime_management_pass]
    if offloading_strategy is not None:
        passes.append(offloading_strategy.create_instruction_pass(graph))

    single_resolution_graph = reduce_resolution(graph, specs)
    program = InstructionScheduler(passes=passes).schedule(single_resolution_graph)
    return _GraphExecutionContext(graph, program, InstructionEngine())
