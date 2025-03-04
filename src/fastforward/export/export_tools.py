import collections

from _pytest import nodes
import torch

from fastforward.export._export_helpers import QNNEncoding, QNNEncodingEntry


class DynamoGraphTraverser:

    def __init__(self, dynamo_graph: torch.fx.graph.Graph, encodings_dictionary: QNNEncoding) -> None:
        self.dynamo_graph = dynamo_graph
        self.nodes = self.dynamo_graph.nodes
        self.encodings_dictionary = encodings_dictionary

    def propagate_downward(self):
        activations_dictionary = self.encodings_dictionary["activation_encodings"]
        parameters_dictionary = self.encodings_dictionary["param_encodings"]

        nodes_with_encodings = {node_name: node_encoding for node_name, node_encoding in activations_dictionary.items()}

        for node_name in parameters_dictionary:
            dynamo_node_name = f"p_{node_name}"
            dynamo_node_name = dynamo_node_name.replace(".", "_")
            nodes_with_encodings[dynamo_node_name] = parameters_dictionary[node_name]

        nodes_of_interest = [node for node in self.nodes if node.name in nodes_with_encodings]

        extended_dictionary = {}

        for node in nodes_of_interest:
            start_node_encodings = nodes_with_encodings[node.name]
            propagated_encodings = breadth_first_search_downward(node, start_node_encodings)

            extended_dictionary.update(propagated_encodings)

        return extended_dictionary

    def propagate_upward(self):
        activations_dictionary = self.encodings_dictionary["activation_encodings"]
        nodes_of_interest = []

        for node in self.nodes:
            if (isinstance(node.target, torch._ops.OpOverload) and node.target.is_view) and node.name in activations_dictionary:
                nodes_of_interest.append(node)

        extended_dictionary = {}

        for node in nodes_of_interest:
            start_node_encodings = activations_dictionary[node.name]
            propagated_encodings = breadth_first_search_upward(node, start_node_encodings)

            extended_dictionary.update(propagated_encodings)

        return extended_dictionary

    def __call__(self):
        all_propagated_encodings = {}
        all_propagated_encodings.update(self.propagate_downward())
        all_propagated_encodings.update(self.propagate_upward())

        return all_propagated_encodings


def breadth_first_search_downward(start_node: torch.fx.node.Node, start_node_encodings: QNNEncodingEntry) -> dict[str, QNNEncodingEntry]:
    queue = collections.deque([start_node])
    explored_nodes = set()

    while queue:
        curr_node = queue.popleft()
        child_nodes = [child for child in curr_node.users]

        for child_node in child_nodes:
            if child_node not in explored_nodes and (isinstance(child_node.target, torch._ops.OpOverload) and child_node.target.is_view):
                queue.append(child_node)
                explored_nodes.add(child_node)

    propagated_encodings = {node.name: start_node_encodings for node in explored_nodes}
    return propagated_encodings


def breadth_first_search_upward(start_node: torch.fx.node.Node, start_node_encodings: QNNEncodingEntry) -> dict[str, QNNEncodingEntry]:
    queue = collections.deque([start_node])
    explored_nodes = set()

    while queue:
        curr_node = queue.popleft()
        parent_nodes = curr_node.all_input_nodes

        for parent_node in parent_nodes:
            if parent_node not in explored_nodes and isinstance(parent_node.target, torch._ops.OpOverload):
                explored_nodes.add(parent_node)
                if parent_node.target.is_view:
                    queue.append(parent_node)

    propagated_encodings = {node.name: start_node_encodings for node in explored_nodes}
    return propagated_encodings
