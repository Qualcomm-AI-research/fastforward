# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""A lightweight utility for building and executing static call graphs composed of torch.nn.Module objects.

The main components are:
- Node: Wraps a PyTorch module with typed input references
- GraphModule: Container for nodes with dependency tracking
- ExecutionEngine: Executes graphs following topological order

Example:
    >>> graph = GraphModule()
    >>> input = graph.add_input("input")
    >>> linear_ref = graph.add_node("linear", nn.Linear(10, 5), [input])
    >>> graph.add_output(linear_ref)
    >>> result = graph(torch.randn(1, 10))
"""

from __future__ import annotations

import collections
import dataclasses
import enum
import functools
import itertools
import uuid

from collections.abc import Collection, Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
)

import torch

if TYPE_CHECKING:
    from fastforward._orchestration.instruction_engine import (
        InstructionEngine,
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
        value: Constant value that this node accepts
    """

    value: Any

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


def resolve_reference(reference: _BaseRef, context: dict[uuid.UUID, Any]) -> Any:
    """Resolve a `_BaseRef` object to its actual value provided in `context`.

    Args:
        reference: Reference to resolve.
        context: Execution context mapping reference IDs to actual/computed values.

    Returns:
        The actual value referenced by `reference`.
    """
    match reference:
        case AttributeRef(reference=ref, attribute=attr):
            base_value = resolve_reference(ref, context)
            match attr:
                case int():
                    return base_value[attr]
                case str():
                    return getattr(base_value, attr)
        case NodeRef(id=ref_id, name=name) | InputRef(id=ref_id, name=name):
            if ref_id not in context:
                msg = f"Missing input with id {ref_id!r}  in context {context} ({name=})"
                raise KeyError(msg)
            return context[ref_id]
        case Const(value=v):
            return v


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
        # AttributeRef wraps another reference, so remap the wrapped reference.
        case AttributeRef(reference=ref):
            remapped_ref = remap_subgraph_reference(ref, new_context, subgraph_nodes, scope_name)
            return dataclasses.replace(old_reference, reference=remapped_ref)
        case InputRef(name=input_name):
            return new_context[input_name]
        case Const():
            return old_reference
        case _BaseRef:
            assert False, f"Unexpected reference type {type(old_reference)}"


@dataclasses.dataclass(frozen=True)
class Node:
    """A node in a 'GraphModule', which wraps a PyTorch module.

    Represents a single operation in the 'GraphModule' as defined by a nn.Module.
    The node maintains references to its inputs through _BaseRef types (NodeRef for other nodes,
    InputRef for external inputs, or Const for literal values).

    Args:
        name: Unique identifier within the graph
        module: PyTorch module to execute
        args: Positional arguments (NodeRef/InputRef/Const)
        kwargs: Keyword arguments (NodeRef/InputRef/Const)
        delegate: Optional function invoked on module depending on execution context
    """

    id: uuid.UUID
    name: str
    module: torch.nn.Module
    args: Collection[_BaseRef]
    kwargs: Mapping[str, _BaseRef] = dataclasses.field(default_factory=dict)
    delegate: Callable[..., None] | None = None


class GraphModule(torch.nn.Module):
    """Static call-graph representation of a PyTorch model.

    'GraphModule' stores 'Node' objects whose output may feed to other nodes,
    giving an explicit graph. This allows for explicit dependency management,
    custom execution orders, subgraph composition and reuse, and clear separation
    between graph structure and execution.
    """

    def __init__(self) -> None:
        super().__init__()
        self._nodes: dict[uuid.UUID, Node] = {}
        self._node_refs: dict[uuid.UUID, NodeRef] = {}
        self._inputs: dict[str, InputRef] = dict()
        self._outputs: list[NodeRef | AttributeRef] = []

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
            if node.module is module:
                return NodeRef(id=id, name=node.name)

        msg = f"Module {module} not found in Graphmodule."
        raise ValueError(msg)

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
        module: torch.nn.Module,
        args: Collection[_BaseRef],
        kwargs: Mapping[str, _BaseRef] | None = None,
        *,
        node_id: uuid.UUID | None = None,
    ) -> NodeRef:
        """Add a node to this 'GraphModule'.

        Creates a new Node wrapping the given module and adds it to the graph. The node
        can reference other nodes' outputs, external inputs, or constant values through
        its arguments.

        Args:
            name: Unique identifier of the node within its scope
            module: PyTorch module to execute
            args: Positional arguments (NodeRef/InputRef/Const)
            kwargs: Keyword arguments (NodeRef/InputRef/Const)
            node_id: Optional node ID. If not provided, a new one will be generated.

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
        node = Node(node_id, name, module, args, kwargs or {})
        self._nodes[node_id] = node
        ref = NodeRef(node_id, name)
        self._node_refs[node_id] = ref

        *module_path, leaf_name = name.split(".")
        parent = self
        for attr in module_path:
            if not hasattr(parent, attr):
                parent.add_module(attr, torch.nn.Module())
            parent = getattr(parent, attr)

        parent.add_module(leaf_name, node.module)
        return ref

    def add_subgraph(
        self,
        name: str,
        subgraph: GraphModule,
        args: Collection[_BaseRef],
        kwargs: Mapping[str, _BaseRef] | None = None,
    ) -> tuple[NodeRef | AttributeRef, ...]:
        """Inline a subgraph into this 'GraphModule' as a scoped collection of nodes.

        Args:
            name: Scope name for the inlined subgraph
            subgraph: GraphModule to inline into this graph
            args: Positional arguments (NodeRef/InputRef/Const)
            kwargs: Keyword arguments (NodeRef/InputRef/Const). Kwargs are allowed but are
                    expected to be inputs to the subgraph.

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

        remap_reference = functools.partial(
            remap_subgraph_reference,
            new_context=input_binding,
            subgraph_nodes={
                node_ref.name: node_ref.id for node_ref in subgraph._node_refs.values()
            },
            scope_name=name,
        )

        for node in subgraph._nodes.values():
            node_args = [remap_reference(old_reference=arg) for arg in node.args]
            node_kwargs = {
                key: remap_reference(old_reference=arg) for key, arg in node.kwargs.items()
            }
            node_name = f"{name}.{node.name}"
            self.add_node(node_name, node.module, node_args, node_kwargs, node_id=node.id)

        # Return the in-lined subgraph outputs in stored order to preserve positional semantics.
        new_output_refs = []
        for ref in subgraph._outputs:
            remapped = remap_reference(old_reference=ref)
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
        if self._engine is None:
            from fastforward._orchestration.instruction_engine import (
                InstructionEngine,
                InstructionScheduler,
            )

            scheduler = InstructionScheduler()
            self._program = scheduler.schedule(self)
            self._engine = InstructionEngine()

        return self._engine.run(self._program, *args, **kwargs)


class Direction(enum.Enum):
    """Direction for traversal of a graph."""

    FORWARD = enum.auto()
    BACKWARD = enum.auto()
    BIDIRECTIONAL = enum.auto()


def find_reachable_nodes(
    graph: GraphModule,
    start: NodeRef,
    direction: Direction = Direction.FORWARD,
    allowlist: Collection[NodeRef] | None = None,
) -> set[NodeRef]:
    """Return all nodes reachable from start under the specified traversal direction and allow-list.

    Args:
        graph: GraphModule object to traverse.
        start: The node from which the search starts.
        direction: Direction to traverse edges.
        allowlist: Optional allowlist. If supplied, only nodes contained in this
                collection will be visited/returned.

    Returns:
        set[NodeRef]: All nodes that are reachable from `start` while respecting the
                  `allowlist` filter (if provided).  The returned set always contains
                  `start` itself.
    """
    # If no allowlist is provided, *all* nodes are allowlisted.
    if allowlist is None:
        allowlist = set(graph._node_refs.values())

    # A simple DFS algorithm.
    seen, stack = {start}, [start]
    while stack:
        node = stack.pop()

        match direction:
            case Direction.FORWARD:
                neighbors = graph.node_outputs(node)
            case Direction.BACKWARD:
                neighbors = graph.node_inputs(node)
            case Direction.BIDIRECTIONAL:
                neighbors = itertools.chain(graph.node_outputs(node), graph.node_inputs(node))

        for neighbor in neighbors:
            if neighbor in allowlist and neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)

    return seen


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

    def _external_dependencies(node: Node) -> set[str]:
        external_dependencies = set()
        for arg in (*node.args, *node.kwargs.values()):
            match arg.unwrap_ref():
                case NodeRef(name=name, id=node_id) if node_id not in node_ids:
                    external_dependencies.add(name)
                case InputRef(name=name):
                    external_dependencies.add(name)
        return external_dependencies

    external_inputs = {
        external_dependency
        for node_id in node_ids
        for external_dependency in _external_dependencies(graph._nodes[node_id])
    }

    if not external_inputs:
        msg = f"Nodes for subgraph have no (external) inputs: {path_nodes}"
        raise ValueError(msg)

    external_outputs = {
        node_ref
        for node_ref in path_nodes
        if node_ref in graph._outputs
        or any(dep not in path_nodes for dep in graph.node_outputs(node_ref))
    }
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
            module=source_node.module,
            args=[remap_reference(old_reference=arg) for arg in source_node.args],
            kwargs={
                key: remap_reference(old_reference=arg) for key, arg in source_node.kwargs.items()
            },
            node_id=source_node.id,
        )

    subgraph.add_output(*external_outputs)
    return subgraph


def topological_sort(graph: GraphModule) -> list[NodeRef]:
    """Return nodes from a GraphModule in topological order.

    Raises:
        ValueError: If graph contains circular dependencies.
    """
    order = []
    in_degree = {
        node_ref: len(list(graph.node_inputs(node_ref))) for node_ref in graph._node_refs.values()
    }

    queue = collections.deque([name for name, degree in in_degree.items() if degree == 0])

    while queue:
        current = queue.popleft()
        order.append(current)

        for dependent in graph.node_outputs(current):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(order) != len(graph._nodes):
        msg = "Circular dependency detected in graph!"
        raise ValueError(msg)

    return order


@dataclasses.dataclass
class SubgraphSpec:
    """A specification for extracting and optionally optimizing a subgraph.

    When the `input` and `output` layers of a GraphModule are selected, we can create a
    subgraph that includes all layers on the path (inclusive). We assume that if a function
    `fn` is defined, that this function will optimize the resulting subgraph later on.
    """

    input: NodeRef
    output: NodeRef
    fn: Callable[[torch.nn.Module, Collection[Any]], None] | None = None


def build_composite_graph(graph: GraphModule, specs: list[SubgraphSpec]) -> GraphModule:
    """Build a composite GraphModule where partitions become nodes with delegates attached.

    Partitions the input graph according to specs, then builds a new GraphModule where each
    partition is represented as a single node with delegates from specs attached accordingly.

    Args:
        graph: Original GraphModule to partition.
        specs: List of SubgraphSpec defining partitions with optional delegate functions.

    Returns:
        Composite GraphModule with partitions as nodes.
    """
    partitions = _partition_graph(graph, specs)

    name_to_node_ref = {
        node.name: ref for ref in graph._node_refs.values() for node in [graph._nodes[ref.id]]
    }

    partition_to_delegate = {
        partition: spec.fn for partition, spec in zip(partitions[: len(specs)], specs)
    }

    node_to_partition = {
        node_ref: partition
        for partition in partitions
        for node_ref in partition._node_refs.values()
    }
    original_outputs = {
        ref.unwrap_ref() for ref in graph._outputs if isinstance(ref.unwrap_ref(), NodeRef)
    }

    composite = GraphModule()
    composite_inputs: dict[str, InputRef] = {}
    produced_refs: dict[NodeRef, NodeRef | AttributeRef] = {}

    def get_composite_arg(input_name: str) -> _BaseRef:
        """Get or create composite graph reference for partition input."""
        if input_name in graph._inputs:
            if input_name not in composite_inputs:
                composite_inputs[input_name] = composite.add_input(input_name)
            return composite_inputs[input_name]

        return produced_refs[name_to_node_ref[input_name]]

    def register_outputs(partition: GraphModule, partition_ref: NodeRef) -> None:
        """Register partition outputs and add to composite graph outputs if needed."""
        for idx, ref in enumerate(partition._outputs):
            if not isinstance(output_ref := ref.unwrap_ref(), NodeRef):
                continue

            produced_ref = partition_ref if len(partition._outputs) == 1 else partition_ref[idx]
            produced_refs[output_ref] = produced_ref
            if output_ref in original_outputs:
                composite.add_output(produced_ref)

    # We have to add the partitions in topological order for correct dependency handling.
    processed: set[GraphModule] = set()
    for partition_idx, node_ref in enumerate(topological_sort(graph)):
        partition = node_to_partition[node_ref]
        if partition in processed:
            continue
        processed.add(partition)

        partition_args = [get_composite_arg(name) for name in partition.input_names]
        partition_ref = composite.add_node(f"partition_{partition_idx}", partition, partition_args)

        if delegate_fn := partition_to_delegate.get(partition):
            node = composite._nodes[partition_ref.id]
            composite._nodes[partition_ref.id] = dataclasses.replace(node, delegate=delegate_fn)

        register_outputs(partition, partition_ref)

    return composite


def _partition_graph(graph: GraphModule, subgraph_specs: list[SubgraphSpec]) -> list[GraphModule]:
    """Partition a GraphModule into multiple subgraphs based on specifications.

    Creates subgraphs for each SubgraphSpec, then groups remaining unpartitioned
    nodes into connected components. Each resulting partition is a standalone
    GraphModule that can be executed independently.

    Args:
        graph: GraphModule to partition.
        subgraph_specs: List of SubgraphSpec defining explicit partitions.

    Returns:
        List of GraphModule partitions, including both explicit specs and
        remaining connected components.
    """
    explicit_node_sets = _extract_node_sets_from_specs(graph, subgraph_specs)

    explicit_nodes = set().union(*explicit_node_sets)
    remaining_nodes = set(graph._node_refs.values()) - explicit_nodes
    remaining_node_sets = _group_remaining_nodes_by_layer(graph, remaining_nodes, explicit_nodes)
    partitions: list[GraphModule] = []
    for node_set in (*explicit_node_sets, *remaining_node_sets):
        partitions.append(create_subgraph(graph, node_set))

    return partitions


def _group_remaining_nodes_by_layer(
    graph: GraphModule,
    remaining_nodes: set[NodeRef],
    explicit_nodes: set[NodeRef],
) -> list[set[NodeRef]]:
    """Group remaining nodes into layers based on explicit partition boundaries.

    Nodes are grouped such that all nodes in a group:
    1. Are reachable from each other without crossing explicit partition boundaries
    2. Have the same set of explicit partitions as dependencies

    Args:
        graph: GraphModule containing all nodes.
        remaining_nodes: Nodes not in explicit partitions.
        explicit_nodes: Nodes that are in explicit partitions.

    Returns:
        List of node groups, each forming a partition.
    """
    if not remaining_nodes:
        return []

    # For each remaining node, find which explicit nodes it depends on
    node_dependencies: dict[NodeRef, set[NodeRef]] = {}

    for node in remaining_nodes:
        # Find all explicit nodes reachable backward from this node
        deps = find_reachable_nodes(
            graph, node, direction=Direction.BACKWARD, allowlist=explicit_nodes
        )
        node_dependencies[node] = deps

    # Group nodes with the same dependency signature
    # Use tuple of sorted node ids as key since sets aren't hashable
    groups: dict[tuple[uuid.UUID, ...], set[NodeRef]] = {}
    for node, deps in node_dependencies.items():
        deps_key = tuple(sorted(ref.id for ref in deps))
        if deps_key not in groups:
            groups[deps_key] = set()
        groups[deps_key].add(node)

    # Within each group, find connected components (forward only to respect order)
    result: list[set[NodeRef]] = []
    for group_nodes in groups.values():
        components = _find_connected_components(graph, group_nodes)
        result.extend(components)

    return result


def _extract_node_sets_from_specs(
    graph: GraphModule, specs: list[SubgraphSpec]
) -> list[set[NodeRef]]:
    """Extract node sets from explicit SubgraphSpecs.

    Args:
        graph: GraphModule containing nodes in `specs`.
        specs: List of SubgraphSpec to convert.

    Returns:
        List of NodeRef sets, one per SubgraphSpec.

    Raises:
        ValueError: If any specs have overlapping nodes.
    """
    used: set[NodeRef] = set()
    node_sets: list[set[NodeRef]] = []

    for spec in specs:
        nodes = find_nodes_on_path(graph, spec.input, spec.output)

        if overlap := nodes & used:
            msg = f"Overlapping nodes in subgraph specs: {overlap}"
            raise ValueError(msg)

        node_sets.append(nodes)
        used |= nodes

    return node_sets


def _find_connected_components(graph: GraphModule, nodes: set[NodeRef]) -> list[set[NodeRef]]:
    """Finds the connected components within a given set of nodes.

    Args:
        graph: GraphModule containing `nodes`.
        nodes: The set of nodes to analyze.

    Returns:
        A list of node sets, where each set represents a connected component.
    """
    nodes_to_visit = nodes.copy()

    components: list[set[NodeRef]] = []
    while nodes_to_visit:
        root = next(iter(nodes_to_visit))

        component = find_reachable_nodes(
            graph, root, direction=Direction.FORWARD, allowlist=nodes_to_visit
        )
        components.append(component)
        nodes_to_visit -= component

    return components
