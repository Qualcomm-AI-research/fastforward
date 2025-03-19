# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import abc
import collections

from typing import Generic, TypeVar

import torch

from fastforward.export._export_helpers import QuantParametersDict

_T = TypeVar("_T")


class GraphOperation(abc.ABC, Generic[_T]):
    """Base class for interacting with a torch fx graph."""

    @abc.abstractmethod
    def process(self) -> _T:
        """Method for performing some operation in the graph."""
        ...


class PropagateEncodingsOperation(GraphOperation[dict[str, QuantParametersDict]]):
    """Graph operation for propagating encodings.

    Args:
        exported_program: The dynamo generated ExportedProgram where the propagation will happen.
        quantization_parameter_dict: The dictionary of quantization parameters that has been gathered through
            the NodeVisitor process.
    """

    def __init__(
        self,
        exported_program: torch.export.exported_program.ExportedProgram,
        quantization_parameter_dict: dict[str, QuantParametersDict],
    ) -> None:
        self.exported_program = exported_program

        self.input_spec_to_type: dict[str, torch.export.graph_signature.InputKind] = {
            input_spec.arg.name: input_spec.kind
            for input_spec in self.exported_program.graph_signature.input_specs
        }

        self.dynamo_graph: torch.fx.graph.Graph = exported_program.graph
        self.graph_nodes: dict[str, torch.fx.node.Node] = {
            node.name: node for node in self.dynamo_graph.nodes
        }
        self.quantization_parameter_dict = quantization_parameter_dict

    def _get_node_and_encodings_from_name(
        self, name: str
    ) -> tuple[torch.fx.node.Node | None, QuantParametersDict | None]:
        """Retrieve the node object and encodings given a potential node name.

        Operation to retrieve the node and encodings from a given node name, based on whether
        this is presented as a node on the graph, and encodings associated with it on the quantization
        logs. NB: The parameter nodes in the graph have a different name structure to signify them
        as parameters, than how they appear in the torch model. For example, say there is a weight
        parameter on the first linear layer of a model. This would usually appear as "fc1.weight". However,
        the node name for this is `p_fc1_weight`. For this reason we have some additional logic to convert
        a potential node name to the expected graph node name.

        Args:
            name: The name of the node to look up. This is meant to follow the naming paradigm of the nodes
                in the quantization logs.

        Returns:
            The node object and its corresponding parameters.
        """
        node = self.graph_nodes.get(name)
        encodings = self.quantization_parameter_dict.get(name)
        if node is None:
            potential_node_name = f"p_{name}"
            potential_node_name = potential_node_name.replace(".", "_")
            node = self.graph_nodes.get(potential_node_name)
            encodings = self.quantization_parameter_dict.get(name)
        return (node, encodings)

    def process(self) -> dict[str, QuantParametersDict]:
        """Main process for propagating quantization encodings to other nodes.

        The following logic is followed for encoding propagation:

        1) For all the nodes found in the quantization_parameter_dict dictionary we propagate the
            encodings to their children (where this is possible).
        2) For all the nodes in the quantization_parameter_dict dictionary we propagate the encodings
            to their parents (where this is possible).
        3) As we are storing the quantization parameters to the nodes we perform a final collection step
            during which we iterate through all the nodes in the graph and we collect the quantization
            parameters to a dictionary.
        """
        for node_name in self.quantization_parameter_dict:
            node, encodings = self._get_node_and_encodings_from_name(node_name)
            if node is not None and encodings is not None:
                self._propagate_to_children(node, encodings)

        for node_name in self.quantization_parameter_dict:
            node, encodings = self._get_node_and_encodings_from_name(node_name)
            if node is not None and encodings is not None:
                self._propagate_to_parents(node, encodings)

        propagated_encodings = {}

        for node_name, node in self.graph_nodes.items():
            quantization_parameters = node.meta.pop("quantization_encodings", None)
            if quantization_parameters is not None and not self._is_node_parameter_node(node):
                propagated_encodings[node_name] = quantization_parameters

        return propagated_encodings

    def _assign_quantization_encodings_to_node(
        self, node: torch.fx.node.Node, node_encodings: QuantParametersDict
    ) -> None:
        """Way of marking a node as being visited, and storing quantization parameters in its `meta` dictionary.

        Args:
            node: The node object to which encodings will be assigned.
            node_encodings: The encodings to be assigned to the `node` object.
        """
        node.meta["quantization_encodings"] = node_encodings

    def _propagate_to_children(
        self, start_node: torch.fx.node.Node, start_node_encodings: QuantParametersDict
    ) -> None:
        """Propagation of encodings parameters from a starting node to its children.

        Given a node and its corresponding encodings, propagate the encodings through its children
        where that is possible. This is desirable as the quantization logging might be unable to
        capture operations that are not explicitly quantized. For example, consider the linear
        operation, which is depicted in the following visual representation:

        input ------------> matmul -> ...
                              ^
                              |
        weight -> transpose ---

        In this case there is a weight transposition, which is not explicitly defined by the user.
        As such, this operation will not be associated with quantization encodings, which we know
        will be the same as the weights. So, propagating these encodings is useful as it will provide
        a more complete representation of the quantization parameters through more operations in the
        graph.

        To perform the propagation the following rules are considered:

            1) The child should not have been already visited, ie it has not already been assigned quantization encodings.
            2) The encodings can only be propagated from parent to child, if the child is a view-type operation,
                meaning that it should have the same quantization encodings as the parent.

        Args:
            start_node: The node object to start traversing from.
            start_node_encodings: The corresponding encodings to the `start_node`.
        """
        self._assign_quantization_encodings_to_node(start_node, start_node_encodings)
        queue = collections.deque([start_node])

        while queue:
            curr_node = queue.popleft()
            child_nodes = [child for child in curr_node.users]

            for child_node in child_nodes:
                is_child_visited = child_node.meta.get("quantization_encodings") is not None
                if not is_child_visited and _is_node_view_op(child_node):
                    queue.append(child_node)
                    self._assign_quantization_encodings_to_node(child_node, start_node_encodings)

    def _propagate_to_parents(
        self, start_node: torch.fx.node.Node, start_node_encodings: QuantParametersDict
    ) -> None:
        """Propagation of encodings parameters from a starting node to its parents.

        Given a node and its corresponding enclodings, propagate the encodings through its parents
        where that is possible. The reason for this propagation is that for certain operations the
        quantization encodings might not be applied to the desirer operation. For example, consider
        a linear layer with the following visual representation:

        input -> reshape -> matmul -> reshape_1 -> softmax
                              ^
                              |
        weight -> transpose ---

        Due to the usage of the `linear` function in torch, the `matmul` operation is
        followed by a view-type operation `reshape_1`. The logic for acquiring the quantization
        encodings is to look up the operation that came before a quantization/dequantization operation.
        In a simplified graph, this looks like this:

        input -> quant -> dequant -> linear -> quant -> dequant

        Therefore the quant/dequant operations will be attribute to the `reshape_1` node instead of
        the `matmul` node. To address this we iterate through the quantization encodings and we attempt
        to propagate any encodings to its parent following these rules:

            1) The encodings are not propagated if the starting node is not a view operation. In this
                case the traversal stops at the starting node.
            2) In the case the parent already has quantization encodings then the encodings are not
                propagated and traversal stops.
            3) In the case the parent is a view operation with no encodings attached to it, we can
                propagate the encodings. This is also the only case when we can continue propagating
                to its respective parents.

        Args:
            start_node: The node object to start traversing from.
            start_node_encodings: The corresponding encodings to the `start_node`.
        """
        self._assign_quantization_encodings_to_node(start_node, start_node_encodings)
        if not _is_node_view_op(start_node):
            return

        queue = collections.deque([start_node])

        while queue:
            curr_node = queue.popleft()
            parent_nodes = curr_node.all_input_nodes

            for parent_node in parent_nodes:
                is_parent_visited = parent_node.meta.get("quantization_encodings") is not None
                if not is_parent_visited:
                    self._assign_quantization_encodings_to_node(parent_node, start_node_encodings)
                    if _is_node_view_op(parent_node):
                        queue.append(parent_node)

    def _is_node_parameter_node(self, node: torch.fx.node.Node) -> bool:
        """Checks if node is mapped to a model input parameter.

        Args:
            node: The node object to check
        Returns:
            A boolean of whether the node is a parameter node.
        """
        return self.input_spec_to_type.get(node.name) is not None


def _is_node_view_op(node: torch.fx.node.Node) -> bool:
    """Check if a node is a view-type operation.

    Args:
        node: The node that needs to be checked.

    Returns:
        A boolean of whether the node is a view-type operation or not.
    """
    if isinstance(node.target, torch._ops.OpOverload):
        return node.target.is_view
    return False
