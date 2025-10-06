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
import uuid

from collections.abc import Collection, Mapping
from typing import Any

import torch


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
    """

    id: uuid.UUID
    name: str
    module: torch.nn.Module
    args: Collection[_BaseRef]
    kwargs: Mapping[str, _BaseRef] = dataclasses.field(default_factory=dict)


class GraphModule(torch.nn.Module):
    """Static call-graph representation of a PyTorch model.

    'GraphModule' stores 'Node' objects whose output may feed to other nodes,
    giving an explicit graph. This allows for explicit dependency management,
    custom execution orders, subgraph composition and reuse, and clear separation
    between graph structure and execution.
    """

    def __init__(self) -> None:
        super().__init__()
        self._nodes: dict[str, Node] = {}
        self._node_refs: dict[str, NodeRef] = {}
        self._inputs: dict[str, InputRef] = dict()
        self._outputs: list[NodeRef | AttributeRef] = []
        self._engine: ExecutionEngine | None = None

    @property
    def input_names(self) -> list[str]:
        """Return the list of graph input names in definition order."""
        return list(self._inputs.keys())

    def node_ref(self, name: str) -> NodeRef:
        """Return the NodeRef for 'name', raising ValueError if it is unknown."""
        for node_ref in self._node_refs.values():
            if node_ref.name == name:
                return node_ref

        msg = f"Unknown node name '{name}'"
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
                case NodeRef(name=node_name):
                    if node_name not in self._nodes:
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
        if name in self._nodes:
            msg = f"Duplicate node name: {name}"
            raise ValueError(msg)

        node_id = node_id or uuid.uuid4()
        if any(n.id == node_id for n in self._nodes.values()):
            msg = f"Duplicate node id: {node_id}"
            raise ValueError(msg)

        # Reset execution engine if it existed since the graph will be altered.
        self._engine = None
        node = Node(node_id, name, module, args, kwargs or {})
        self._nodes[name] = node
        ref = NodeRef(node_id, name)
        self._node_refs[name] = ref

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
            subgraph_nodes={node_name: ref.id for node_name, ref in subgraph._node_refs.items()},
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

    def forward(self, *args, **kwargs):  # type: ignore[no-untyped-def]
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
            self._engine = ExecutionEngine(self)
        return self._engine.run(*args, **kwargs)


@dataclasses.dataclass
class ExecutionPlan:
    """Execution plan for a 'GraphModule'.

    This class stores an arbitrary execution order for a GraphModule, and dependencies
    between nodes that are required for a node's execution.

    Args:
        order: List of node names in execution order
        dependencies: Mapping of each node to its set of dependency nodes. An empty set
            indicates the node has no dependencies and can execute immediately (typically
            using only external inputs).
    """

    order: list[NodeRef]
    dependencies: dict[NodeRef, set[NodeRef]]
    _dependents: dict[NodeRef, set[NodeRef]] | None = dataclasses.field(
        default=None, init=False, repr=False
    )

    @property
    def dependents(self) -> dict[NodeRef, set[NodeRef]]:
        """Mapping of each node to the set of nodes that depend on it."""
        if self._dependents is None:
            dependents = collections.defaultdict(set)

            for node, predecessors in self.dependencies.items():
                for predecessor in predecessors:
                    dependents[predecessor].add(node)

            self._dependents = dependents

        return self._dependents


class TopologicalExecutionPlan(ExecutionPlan):
    """Create an execution plan with nodes ordered topologically."""

    @classmethod
    def from_graph(cls, graph: GraphModule) -> TopologicalExecutionPlan:
        """Generate an ExecutionPlan by topologically sorting the nodes of the given GraphModule."""
        dependencies: dict[NodeRef, set[NodeRef]] = collections.defaultdict(set)
        dependents: dict[NodeRef, set[NodeRef]] = collections.defaultdict(set)

        for node_name, node in graph._nodes.items():
            node_ref = graph._node_refs[node_name]

            for arg in (*node.args, *node.kwargs.values()):
                match arg.unwrap_ref():
                    case NodeRef(id=dep_id, name=dep_name):
                        if dep_name not in graph._nodes:
                            msg = f"Node '{node_name}' depends on unknown node '{dep_name}'"
                            raise ValueError(msg)

                        dep_ref = graph._node_refs[dep_name]

                        if dep_ref.id != dep_id:
                            msg = f"Node '{node_name}' depends on node '{dep_name}' more than once"
                            raise ValueError(msg)

                        dependencies[node_ref].add(dep_ref)
                    case _:
                        # Skip non-node references (e.g., constants or inputs)
                        pass

            for dep in dependencies[node_ref]:
                dependents[dep].add(node_ref)

        # Topological sort using Kahn's algorithm
        execution_order = []
        in_degree = {
            graph._node_refs[name]: len(dependencies.get(graph._node_refs[name], ()))
            for name in graph._nodes
        }

        queue = collections.deque([name for name, degree in in_degree.items() if degree == 0])

        while queue:
            current = queue.popleft()
            execution_order.append(current)

            for dependent in dependents[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(execution_order) != len(graph._nodes):
            msg = "Circular dependency detected in graph!"
            raise ValueError(msg)

        return cls(order=execution_order, dependencies=dict(dependencies))


class ExecutionEngine:
    """Executes a 'GraphModule' according to a topological execution plan."""

    def __init__(self, graph: GraphModule) -> None:
        self.graph = graph
        self._plan: ExecutionPlan | None = None

    def _bind_args(self, *args, **kwargs) -> dict[uuid.UUID, Any]:  # type: ignore[no-untyped-def]
        """Bind input arguments to graph input names.

        Maps positional args to input_names by position, or kwargs by name.
        Returns context dict for graph execution.
        """
        ctx: dict[uuid.UUID, Any] = {}
        if len(args) > len(self.graph.input_names):
            msg = f"Expected {len(self.graph.input_names)} positional arguments, got {len(args)}"
            raise TypeError(msg)

        for arg_value, ref in zip(args, self.graph._inputs.values()):
            ctx[ref.id] = arg_value

        for kwarg_name, kwarg_value in kwargs.items():
            if kwarg_name not in self.graph._inputs:
                msg = f"Got an unexpected keyword argument '{kwarg_name}'"
                raise TypeError(msg)

            ref = self.graph._inputs[kwarg_name]
            if ref.id in ctx:
                msg = f"Got multiple values for argument '{kwarg_name}'"
                raise TypeError(msg)

            ctx[ref.id] = kwarg_value

        if missing_inputs := {ref.id for ref in self.graph._inputs.values()} - ctx.keys():
            msg = f"Missing required input arguments: {missing_inputs}"
            raise TypeError(msg)

        return ctx

    def run(self, *args, **kwargs) -> Any:  # type: ignore[no-untyped-def]
        """Execute the 'GraphModule' with given inputs.

        Binds inputs to graph input names, executes nodes in topological order,
        and returns output node results.

        Args:
            *args: Positional inputs for the GraphModule, corresponding to the names defined with
                `add_input`.
            **kwargs: Key-word inputs for the GraphModule, corresponding to the names defined with
                `add_input`.

        Returns:
            Single output value or tuple of outputs depending on graph configuration
        """
        if self._plan is None:
            self._plan = TopologicalExecutionPlan.from_graph(self.graph)

        ctx = self._bind_args(*args, **kwargs)

        for node_ref in self._plan.order:
            node = self.graph._nodes[node_ref.name]

            resolved_args = [resolve_reference(arg, ctx) for arg in node.args]
            resolved_kwargs = {key: resolve_reference(arg, ctx) for key, arg in node.kwargs.items()}

            ctx[node_ref.id] = node.module(*resolved_args, **resolved_kwargs)

        results = []
        for ref in self.graph._outputs:
            match ref:
                case AttributeRef():
                    results.append(resolve_reference(ref, ctx))
                case NodeRef(id=node_id):
                    results.append(ctx[node_id])

        return tuple(results) if len(results) > 1 else results[0]


class Direction(enum.Enum):
    """Direction for traversal of a graph."""

    FORWARD = enum.auto()
    BACKWARD = enum.auto()
    BIDIRECTIONAL = enum.auto()


def find_reachable_nodes(
    plan: ExecutionPlan,
    start: NodeRef,
    direction: Direction = Direction.FORWARD,
    allowlist: Collection[NodeRef] | None = None,
) -> set[NodeRef]:
    """Return all nodes reachable from start under the specified traversal direction and allow-list.

    Args:
        plan: ExecutionPlan containing dependency information.
        start: The node from which the search starts.
        direction: Direction to traverse edges.
        allowlist: Optional allowlist. If supplied, only nodes contained in this
                collection will be visited/returned.

    Returns:
        set[NodeRef]: All nodes that are reachable from `start` while respecting the
                  `allowlist` filter (if provided).  The returned set always contains
                  `start` itself.
    """
    match direction:
        case Direction.FORWARD:
            adjacency = plan.dependents
        case Direction.BACKWARD:
            adjacency = plan.dependencies
        case Direction.BIDIRECTIONAL:
            adjacency = {
                node: plan.dependencies[node] | plan.dependents[node]
                for node in set(plan.dependencies.keys()) | set(plan.dependents.keys())
            }

    # If no allowlist is provided, *all* nodes are allowlisted.
    if allowlist is None:
        allowlist = set(plan.dependencies.keys()) | set(plan.dependents.keys())

    # A simple DFS algorithm.
    seen, stack = {start}, [start]
    while stack:
        node = stack.pop()
        for neighbor in adjacency.get(node, ()):
            if neighbor in allowlist and neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)

    return seen


def find_nodes_on_path(
    graph: GraphModule,
    plan: ExecutionPlan,
    start: NodeRef,
    end: NodeRef,
) -> set[NodeRef]:
    """Return the set of nodes on the path from `start` to `end` (inclusive).

    Args:
        graph: GraphModule to analyze.
        plan: ExecutionPlan containing dependency information.
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

        for dependent in plan.dependents.get(current, set()):
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


def create_subgraph(
    graph: GraphModule,
    plan: ExecutionPlan,
    path_nodes: set[NodeRef],
) -> GraphModule:
    """Build and return a standalone GraphModule that contains exactly `path_nodes`.

    Args:
        graph: GraphModule to extract subgraph from.
        plan: ExecutionPlan with dependency information for this GraphModule.
        path_nodes: Names of nodes that must appear in the subgraph.

    Returns:
        A new GraphModule representing the extracted sub-graph.

    Raises:
        ValueError: If proposed subgraph has no external inputs.
    """
    node_names: set[str] = {ref.name for ref in path_nodes}

    def _external_dependencies(node: Node) -> set[str]:
        external_dependencies = set()
        for arg in (*node.args, *node.kwargs.values()):
            match arg.unwrap_ref():
                case NodeRef(name=name) if name not in node_names:
                    external_dependencies.add(name)
                case InputRef(name=name):
                    external_dependencies.add(name)
        return external_dependencies

    external_inputs = {
        external_dependency
        for node_name in node_names
        for external_dependency in _external_dependencies(graph._nodes[node_name])
    }

    if not external_inputs:
        msg = f"Nodes for subgraph have no (external) inputs: {path_nodes}"
        raise ValueError(msg)

    external_outputs = {
        node_name
        for node_name in node_names
        if graph._node_refs[node_name] in graph._outputs
        or any(dep not in path_nodes for dep in plan.dependents[graph._node_refs[node_name]])
    }

    # Connect the inputs, nodes, and outputs to a new subgraph.
    subgraph = GraphModule()

    remap_reference = functools.partial(
        remap_subgraph_reference,
        new_context={name: subgraph.add_input(name) for name in external_inputs},
        subgraph_nodes={ref.name: ref.id for ref in path_nodes},
    )

    for node_name in node_names:
        source_node = graph._nodes[node_name]
        subgraph.add_node(
            name=node_name,
            module=source_node.module,
            args=[remap_reference(old_reference=arg) for arg in source_node.args],
            kwargs={
                key: remap_reference(old_reference=arg) for key, arg in source_node.kwargs.items()
            },
            node_id=source_node.id,
        )

    subgraph.add_output(*(subgraph.node_ref(node_name) for node_name in external_outputs))
    return subgraph


@dataclasses.dataclass
class SubgraphSpec:
    """A Specification of a subgraph."""

    input: NodeRef
    output: NodeRef


def partition_graph(graph: GraphModule, subgraph_specs: list[SubgraphSpec]) -> list[GraphModule]:
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
    plan = TopologicalExecutionPlan.from_graph(graph)

    explicit_node_sets = _extract_node_sets_from_specs(graph, plan, subgraph_specs)

    remaining_nodes = set(graph._node_refs.values()) - set().union(*explicit_node_sets)
    remaining_node_sets = _find_connected_components(plan, remaining_nodes)

    partitions: list[GraphModule] = []
    for node_set in (*explicit_node_sets, *remaining_node_sets):
        partitions.append(create_subgraph(graph, plan, node_set))

    return partitions


def _extract_node_sets_from_specs(
    graph: GraphModule,
    plan: ExecutionPlan,
    specs: list[SubgraphSpec],
) -> list[set[NodeRef]]:
    """Extract node sets from explicit SubgraphSpecs.

    Args:
        graph: GraphModule containing the nodes.
        specs: List of SubgraphSpec to convert.
        plan: ExecutionPlan for path finding between spec endpoints.

    Returns:
        List of NodeRef sets, one per SubgraphSpec.

    Raises:
        ValueError: If any specs have overlapping nodes.
    """
    used: set[NodeRef] = set()
    node_sets: list[set[NodeRef]] = []

    for spec in specs:
        nodes = find_nodes_on_path(graph, plan, spec.input, spec.output)

        if overlap := nodes & used:
            msg = f"Overlapping nodes in subgraph specs: {overlap}"
            raise ValueError(msg)

        node_sets.append(nodes)
        used |= nodes

    return node_sets


def _find_connected_components(
    plan: ExecutionPlan,
    nodes: set[NodeRef],
) -> list[set[NodeRef]]:
    """Finds the connected components within a given set of nodes.

    Args:
        plan: ExecutionPlan for connectivity information.
        nodes: The set of nodes to analyze.

    Returns:
        A list of node sets, where each set represents a connected component.
    """
    nodes_to_visit = nodes.copy()

    components: list[set[NodeRef]] = []
    while nodes_to_visit:
        root = next(iter(nodes_to_visit))

        component = find_reachable_nodes(
            plan, root, direction=Direction.BIDIRECTIONAL, allowlist=nodes_to_visit
        )
        components.append(component)
        nodes_to_visit -= component

    return components
