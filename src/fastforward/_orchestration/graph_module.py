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
    >>> engine = ExecutionEngine(graph)
    >>> result = engine(torch.randn(1, 10))
"""

from __future__ import annotations

import collections
import dataclasses

from collections.abc import Collection, Iterable, Mapping
from typing import Any

import torch


@dataclasses.dataclass(frozen=True)
class NodeRef:
    """Reference to a node inside a GraphModule.

    Args:
        name: Fully-qualified name inside the GraphModule
    """

    name: str

    def __repr__(self) -> str:
        return f"NodeRef({self.name})"


@dataclasses.dataclass(frozen=True)
class InputRef:
    """Reference to a GraphModule input.

    Args:
        name: Input name of the current GraphModule
    """

    name: str

    def __repr__(self) -> str:
        return f"InputRef({self.name})"


@dataclasses.dataclass(frozen=True)
class Const:
    """Literal argument for a node.

    Args:
        value: Constant value that this node accepts
    """

    value: Any

    def __repr__(self) -> str:
        return f"Const({self.value})"


NodeArg = NodeRef | InputRef | Const


def resolve_node_arg(arg: NodeArg, context: dict[str, Any]) -> Any:
    """Resolve a node argument to its actual value from execution context.

    Returns the referenced value for NodeRef/InputRef, or the literal value for Const.
    """
    match arg:
        case NodeRef(name=name) | InputRef(name=name):
            if name not in context:
                raise KeyError(f"Missing input '{name}' in context {context}")
            return context[name]

        case Const(value=v):
            return v

        case _:
            raise ValueError(f"Invalid node argument: {arg}")


@dataclasses.dataclass
class Node:
    """A node in a 'GraphModule', which wraps a PyTorch module.

    Represents a single operation in the 'GraphModule' as defined by a nn.Module.
    The node maintains references to its inputs through NodeArg types (NodeRef for other nodes,
    InputRef for external inputs, or Const for literal values).

    Args:
        name: Unique identifier within the graph
        module: PyTorch module to execute
        args: Positional arguments (NodeRef/InputRef/Const)
        kwargs: Keyword arguments (NodeRef/InputRef/Const)
    """

    name: str
    module: torch.nn.Module
    args: Collection[NodeArg]
    kwargs: Mapping[str, NodeArg] = dataclasses.field(default_factory=dict)


class GraphModule(torch.nn.Module):
    """Static call-graph representation of a PyTorch model.

    'GraphModule' stores 'Node' objects whose output may feed to other nodes,
    giving an explicit graph. This allows for explicit dependency management,
    custom execution orders, subgraph composition and reuse, and clear separation
    between graph structure and execution.

    Attributes:
        input_names: External inputs the graph expects
        _nodes: Named Node objects representing operations
        _output_names: Node names to return as outputs
        _engine: Lazily-created ExecutionEngine for running the graph
    """

    def __init__(self) -> None:
        super().__init__()
        self.input_names: list[str] = []
        self._nodes: dict[str, Node] = {}
        self._output_names: list[str] = []

        self._engine: ExecutionEngine | None = None

    def add_input(self, name: str) -> InputRef:
        """Add an input name to the graph."""
        if name in self.input_names:
            raise ValueError(f"Duplicate input name: {name}")

        if not isinstance(name, str):
            raise TypeError(f"Input name must be a str, not {type(name)}")

        self.input_names.append(name)
        return InputRef(name)

    def add_output(self, nodes: NodeRef | str | Iterable[NodeRef | str]) -> None:
        """Add output nodes to the graph."""
        if isinstance(nodes, (NodeRef, str)):
            nodes = (nodes,)

        for node in nodes:
            name = node.name if isinstance(node, NodeRef) else node
            if name not in self._nodes:
                raise KeyError(f"Unknown node {name}")
            self._output_names.append(name)

    def add_node(
        self,
        name: str,
        module: torch.nn.Module,
        args: Collection[NodeArg],
        kwargs: Mapping[str, NodeArg] | None = None,
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

        Returns:
            NodeRef: Reference to the created node

        Raises:
            ValueError: If node already exists in graph
        """
        if name in self._nodes:
            raise ValueError(f"Duplicate node name: {name}")

        # Reset execution engine if it existed since the graph will be altered.
        self._engine = None

        node = Node(name, module, args, kwargs or {})

        *module_path, leaf_name = name.split(".")
        parent = self
        for attr in module_path:
            if not hasattr(parent, attr):
                parent.add_module(attr, torch.nn.Module())
            parent = getattr(parent, attr)

        parent.add_module(leaf_name, node.module)
        self._nodes[name] = node

        return NodeRef(name)

    def add_subgraph(
        self,
        name: str,
        subgraph: GraphModule,
        args: Collection[NodeArg],
        kwargs: Mapping[str, NodeArg] | None = None,
    ) -> tuple[NodeRef, ...]:
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
        if not subgraph._output_names:
            raise ValueError("Subgraph has no output nodes defined")

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
                raise ValueError(f"Unknown subgraph input '{key}'")
            if key in input_binding:
                raise ValueError(f"Multiple values for subgraph input '{key}'")
            input_binding[key] = value

        def resolve_arg(arg: NodeArg) -> NodeArg:
            match arg:
                case InputRef(name=input_name):
                    return input_binding[input_name]
                case NodeRef(name=node_name):
                    return NodeRef(f"{name}.{node_name}")
                case Const():
                    return arg

        new_output_refs: list[NodeRef] = []
        for node in subgraph._nodes.values():
            node_args = [resolve_arg(arg) for arg in node.args]
            node_kwargs = {key: resolve_arg(arg) for key, arg in node.kwargs.items()}
            node_name = f"{name}.{node.name}"
            node_ref = self.add_node(node_name, node.module, node_args, node_kwargs)

            if node.name in subgraph._output_names:
                new_output_refs.append(node_ref)

        return tuple(new_output_refs)

    def forward(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        """Execute the graph using the default ExecutionEngine.

        Uses a topologically-sorted execution order that preserves the original
        graph's execution semantics.

        Args:
            *args: Positional arguments for graph inputs
            **kwargs: Keyword arguments for graph inputs

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
    between nodes that are required for a node's execution. See `topological_execution_plan` for
    a topological sort of the nodes of a GraphModule resulting in a ExecutionPlan.

    Args:
        order: List of node names in execution order
        dependencies: Mapping of each node to its set of dependency nodes. An empty set
            indicates the node has no dependencies and can execute immediately (typically
            using only external inputs).
    """

    order: list[str]
    dependencies: dict[str, set[str]]


def topological_execution_plan(graph: GraphModule) -> ExecutionPlan:
    """Compute a topological execution plan for a 'GraphModule'.

    Only NodeRef dependencies are considered edges. InputRef/Const are treated as sources.

    Args:
        graph: GraphModule to create execution plan for

    Returns:
        ExecutionPlan: Topologically sorted execution order with dependencies

    Raises:
        ValueError: If circular dependency detected or dangling NodeRef found
    """
    dependencies = collections.defaultdict(set)
    dependents = collections.defaultdict(set)

    for node_name, node in graph._nodes.items():
        for arg in (*node.args, *node.kwargs.values()):
            match arg:
                case NodeRef(name=dep):
                    if dep not in graph._nodes:
                        raise ValueError(f"Node '{node_name}' depends on unknown node '{dep}'")
                    dependencies[node_name].add(dep)

                case _:
                    pass

        for dep in dependencies[node_name]:
            dependents[dep].add(node_name)

    # Topological sort using Kahn's algorithm
    execution_order = []
    in_degree = {name: len(dependencies[name]) for name in dependencies.keys()}
    queue = collections.deque([name for name, degree in in_degree.items() if degree == 0])

    while queue:
        current = queue.popleft()
        execution_order.append(current)

        for dependent in dependents[current]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(execution_order) != len(graph._nodes):
        raise ValueError("Circular dependency detected in graph!")

    return ExecutionPlan(order=execution_order, dependencies=dict(dependencies))


class ExecutionEngine:
    """Executes a 'GraphModule' according to a topological execution plan."""

    def __init__(self, graph: GraphModule) -> None:
        self.graph = graph
        self._plan: ExecutionPlan | None = None

    def _bind_args(self, *args, **kwargs) -> dict[str, Any]:  # type: ignore[no-untyped-def]
        """Bind input arguments to graph input names.

        Maps positional args to input_names by position, or kwargs by name.
        Returns context dict for graph execution.
        """
        ctx: dict[str, Any] = {}
        if len(args) > len(self.graph.input_names):
            msg = f"Expected {len(self.graph.input_names)} positional arguments, got {len(args)}"
            raise TypeError(msg)

        for arg_value, input_name in zip(args, self.graph.input_names):
            ctx[input_name] = arg_value

        for kwarg_name, kwarg_value in kwargs.items():
            if kwarg_name in ctx:
                raise TypeError(f"Got multiple values for argument '{kwarg_name}'")

            if kwarg_name not in self.graph.input_names:
                raise TypeError(f"Got an unexpected keyword argument '{kwarg_name}'")

            ctx[kwarg_name] = kwarg_value

        missing_inputs = set(self.graph.input_names) - set(ctx.keys())
        if missing_inputs:
            raise TypeError(f"Missing required input arguments: {missing_inputs}")

        return ctx

    def run(self, *args, **kwargs) -> Any:  # type: ignore[no-untyped-def]
        """Execute the 'GraphModule' with given inputs.

        Binds inputs to graph input names, executes nodes in topological order,
        and returns output node results.

        Args:
            *args: Positional arguments for graph inputs
            **kwargs: Keyword arguments for graph inputs

        Returns:
            Single output value or tuple of outputs depending on graph configuration
        """
        if not self.graph._output_names:
            raise RuntimeError("Graph has no nodes to produce outputs")

        if self._plan is None:
            self._plan = topological_execution_plan(self.graph)

        ctx = self._bind_args(*args, **kwargs)

        for node_name in self._plan.order:
            node = self.graph._nodes[node_name]

            resolved_args = [resolve_node_arg(arg, ctx) for arg in node.args]
            resolved_kwargs = {key: resolve_node_arg(arg, ctx) for key, arg in node.kwargs.items()}

            out = node.module(*resolved_args, **resolved_kwargs)
            ctx[node_name] = out

        if len(self.graph._output_names) == 1:
            return ctx[self.graph._output_names[0]]

        return tuple(ctx[name] for name in self.graph._output_names)
