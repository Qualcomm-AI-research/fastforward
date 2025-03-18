import abc
import collections

from typing import Generic, TypeVar

import torch

from fastforward.export._export_helpers import QuantParametersDict

_T = TypeVar("_T")


class GraphOperation(abc.ABC, Generic[_T]):
    """Base class for interacting with a torch fx graoh."""
    @abc.abstractmethod
    def process(self) -> _T:
        """Method for performing some operation in the graph"""
        ...


class PropagateEncodingsOperation(GraphOperation[dict[str, QuantParametersDict]]):
    def __init__(self, exported_program: torch.export.exported_program.ExportedProgram, quantization_logs: dict[QuantParametersDict]) -> None:
        self.exported_program = exported_program

        self.input_spec_to_type: dict[str, torch.export.graph_signature.InputKind] = {
            input_spec.arg.name: input_spec.kind for input_spec in self.exported_program.graph_signature.input_specs
        }

        self.dynamo_graph = exported_program.graph
        self.graph_nodes = {node.name: node for node in self.dynamo_graph.nodes}
        self.quantization_logs = quantization_logs

    def get_node_and_encodings_fron_name(self, name: str) -> tuple[torch.fx.node.Node, QuantParametersDict]:
        node = self.graph_nodes.get(name)
        encodings = self.quantization_logs.get(name)
        if node is None:
            potential_node_name = f"p_{name}"
            potential_node_name = potential_node_name.replace(".", "_")
            node = self.graph_nodes.get(potential_node_name)
            encodings = self.quantization_logs.get(name)
        return (node, encodings)

    def process(self) -> dict[str, QuantParametersDict]:
        for node_name in self.quantization_logs:
            node, encodings = self.get_node_and_encodings_fron_name(node_name)
            if node is not None:
                self.propagate_to_children(node, encodings)

        for node_name in self.quantization_logs:
            node, encodings = self.get_node_and_encodings_fron_name(node_name)
            if node is not None:
                self.propagate_to_parents(node, encodings)

        propagated_encodings = {}

        for node_name, node in self.graph_nodes.items():
            quantization_parameters = node.meta.get("quantization_encodings")
            if quantization_parameters is not None and not self.is_node_parameter_node(node):
                propagated_encodings[node_name] = quantization_parameters

        return propagated_encodings

    def assign_quantization_encodings_to_node(self, node: torch.fx.node.Node, node_encodings: QuantParametersDict):
        node.meta["quantization_encodings"] = node_encodings

    def propagate_to_children(self, start_node: torch.fx.node.Node, start_node_encodings: QuantParametersDict) -> None:
        self.assign_quantization_encodings_to_node(start_node, start_node_encodings)
        queue = collections.deque([start_node])

        while queue:
            curr_node = queue.popleft()
            child_nodes = [child for child in curr_node.users]

            for child_node in child_nodes:
                is_child_visited = child_node.meta.get("quantization_encodings") is not None
                if not is_child_visited and is_node_view_op(child_node): 
                    queue.append(child_node)
                    self.assign_quantization_encodings_to_node(child_node, start_node_encodings)
                    print(f"Current node: {start_node}, propagating to child node: {child_node}")

    def propagate_to_parents(self, start_node: torch.fx.node.Node, start_node_encodings: QuantParametersDict) -> None:
        self.assign_quantization_encodings_to_node(start_node, start_node_encodings)
        if not is_node_view_op(start_node):
            return

        queue = collections.deque([start_node])

        while queue:
            curr_node = queue.popleft()
            parent_nodes = curr_node.all_input_nodes

            for parent_node in parent_nodes:
                is_parent_visited = parent_node.meta.get("quantization_encodings") is not None
                if not is_parent_visited:
                    self.assign_quantization_encodings_to_node(parent_node, start_node_encodings)
                    print(f"Current node: {start_node}, propagating to parent node: {parent_node}")
                    if is_node_view_op(parent_node):
                        queue.append(parent_node)

    def is_node_parameter_node(self, node: torch.fx.node.Node) -> bool:
        return self.input_spec_to_type.get(node.name, False)


def is_node_view_op(node: torch.fx.node.Node) -> bool:
    """
    Check if a node is a view-type operation.

    Args:
        node: The node that needs to be checked.
    Returns:
        A boolean of whether the node is a view-type operation or not.
    """
    if isinstance(node.target, torch._ops.OpOverload):
        return node.target.is_view
    return False
