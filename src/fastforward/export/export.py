# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

"""Functionality to export a quantized module for running inference on device.

Supported backends:
- QNN
"""

import abc
import json
import pathlib

from operator import attrgetter
from typing import Any, Generic, Optional, Sequence, TypeVar

import onnx
import onnxscript
import torch
import torch_onnx  # type: ignore[import-untyped]

from torch.export.exported_program import ExportedProgram
from torch.export.graph_signature import InputKind, InputSpec
from torch.fx.graph import Graph
from torch.fx.node import Node
from typing_extensions import override

from fastforward.export._export_helpers import (
    generate_qnn_encodings_dictionary,
    get_activations,
    get_input_spec_new_old_mapping,
    get_inputs,
    get_parameters,
)
from fastforward.flags import export_mode

_T = TypeVar("_T")


def _node_name_as_string(node: Node) -> str | Any:
    if not hasattr(node.target, "name"):
        return node.target
    else:
        return node.target.name()


class NodeVisitor(abc.ABC, Generic[_T]):
    """Base class for interacting with torch fx node object in a dynamo graph."""

    @abc.abstractmethod
    def enter(self, node: Node) -> bool:
        """Method for checking whether a given node will be accessed.

        Args:
            node: A torch fx node object
        """
        ...

    @abc.abstractmethod
    def visit(self, graph_exported_program: ExportedProgram, node: Node) -> None:
        """Method for performing an operation with reference to the given node.

        In certain cases the `ExportedProgram` object is required in order to
        perform more in-depth graph manipulation.

        Args:
            graph_exported_program: An `ExportedProgram` object resulting from `torch.export.export`.
            node: A torch fx node object
        """
        ...

    @abc.abstractmethod
    def conclude(self) -> _T:
        """Conclude processing.

        Method for performing any additional operations after the main processing from the
        `visit` method has concluded. This can include some cleanup actions or returning some
        values.

        The value returned by `conclude` is returned by `GraphWrapper.visit()`
        """
        ...


class RequestNode(NodeVisitor[list[Node]]):
    """NodeVisitor subclass responsible for grabbing nodes from a graph.

    Simple query class which can return all the nodes that match certain
    criteria.

    Args:
        op_type: The type of operation that is of interest. The types of operations
            correspond to the `op` property of the torch.fx nodes.
        target_name: The name of the target graph operation. The underlying operation
            that a torch.fx node invokes when performing a forward pass. For example,
            for the fastforward operation `quantize_by_tile` the target name will
            be `fastforward::quantize_by_tile`.
    """

    def __init__(self, op_type: str, target_name: str) -> None:
        self._op_type = op_type
        self._target_name = target_name
        self._nodes: list[Node] = []

    @override
    def enter(self, node: Node) -> bool:
        node_name = _node_name_as_string(node)
        return node.op == self._op_type and node_name == self._target_name

    @override
    def visit(self, graph_exported_program: ExportedProgram, node: Node) -> None:
        self._nodes.append(node)

    @override
    def conclude(self) -> list[Node]:
        return self._nodes

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._op_type} -> {self._target_name})"


class RemoveFunction(NodeVisitor[None]):
    """NodeVisitor subclass responsible for removing nodes from a graph.

    This is done by reconfiguring the input/outputs from a given node. For
    example, consider a three node graph containing nodes `x`, `y`, `z` with
    the following simple structure:
    ```
        x -> y -> z
    ```

    If we want to remove the `y` node, we can access with the criteria defined to the `enter` method,
    find its input node (`x`) and its output node (`z`) and replace the input to `z` to be `x`. So,
    the resulting graph looks like this:
    ```
           y

        x -> z
    ```

    Note that at this stage the `y` node is not removed yet. This can be done by invoking the
    torch graph API. ATTENTION: a node that is being used by other nodes cannot be removed, dynamo
    will raise an error if that is attempted. For this reason, the reconfiguration of inputs/outputs
    needs to be done first.

    The class will work with nodes that have multiple output nodes, where the node's parent will
    be connected to each different output. ATTENTION: this is not the case with the node inputs, where
    we assume that only one input is of importance.

    Finally, note that this reconfiguration needs to happen for a intermediate node (the node needs to have at least
    one input, and at least one output) otherwise the reconnection will raise an error.

    Args:
        op_type: The type of operation that is of interest. The types of operations
            correspond to the `op` property of the torch.fx nodes.
        target_name: The name of the target graph operation. The underlying operation
            that a torch.fx node invokes when performing a forward pass. For example,
            for the fastforward operation `quantize_by_tile` the target name will
            be `fastforward::quantize_by_tile`.
        input_replace_idx: The index of the current nodes input that will be redirected
            to the output node's input.
    """

    def __init__(self, op_type: str, target_name: str, input_replace_idx: int) -> None:
        self._op_type = op_type
        self._target_name = target_name
        self._input_replace_idx = input_replace_idx

    @override
    def enter(self, node: Node) -> bool:
        node_name = _node_name_as_string(node)
        return node.op == self._op_type and node_name == self._target_name

    @override
    def visit(self, graph_exported_program: ExportedProgram, node: Node) -> None:
        self._remove_node(graph_exported_program, node, self._input_replace_idx)

    @override
    def conclude(self) -> None:
        return None

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._op_type} -> {self._target_name})"

    def _remove_node(
        self, graph_exported_program: ExportedProgram, node: Node, input_idx: int
    ) -> None:
        graph: Graph = graph_exported_program.graph_module.graph
        input_specs = graph_exported_program.graph_signature.input_specs
        input_nodes = node.all_input_nodes

        assert input_nodes, "The node does not have any inputs"
        assert node.users.keys(), "The node does not have any outputs"

        for output in list(node.users.keys()):
            output.replace_input_with(node, input_nodes[input_idx])

        graph.erase_node(to_erase=node)

        # Check the parameters of the deleted node, and if those are not used
        # anywhere else in the graph delete them.
        for param_node in input_nodes[1:]:
            if not param_node.users:
                graph.erase_node(to_erase=param_node)

            # Remove any dangling nodes from the input_spec list to avoid raising
            # errors in inference.
            for i, input_spec in enumerate(input_specs):
                if param_node.name == input_spec.arg.name:
                    input_specs.pop(i)
                    break


class LogQuantizationParameter(NodeVisitor[dict[str, dict[str, Any]]]):
    """Log Quantization parameters.

    NodeVisitor subclass responsible for logging the quantization parameter of
    a subset of nodes.

    In the case of quantization, the FastForward operations (`quantize_by_tile` and
    `dequantize_by_tile`) are represented in the graph. The torch.fx nodes for these
    operations contain the parameters that need to be logged, where the first input
    to the node is the parameter/activation to be quantized, and the node arguments
    provide the quantization parameters.

    The result of this subclass are stored in a dictionary where the keys are the names of the
    parameter/activation nodes, and the values are their corresponding quantization parameters.

    Args:
        op_type: The type of operation that is of interest. The types of operations
            correspond to the `op` property of the torch.fx nodes.
        target_name: The name of the target graph operation. The underlying operation
            that a torch.fx node invokes when performing a forward pass. For example,
            for the fastforward operation `quantize_by_tile` the target name will
            be `fastforward::quantize_by_tile`.
    """

    def __init__(self, op_type: str, target_name: str) -> None:
        self._op_type = op_type
        self._target_name = target_name
        self._logs: dict[str, dict[str, Any]] = {}

        self._module_named_parameters: frozenset[str] = frozenset()
        self._module_named_buffers: frozenset[str] = frozenset()
        self._module: torch.nn.Module | None = None

    @override
    def enter(self, node: Node) -> bool:
        node_name = _node_name_as_string(node)
        return node.op == self._op_type and node_name == self._target_name

    @override
    def visit(self, graph_exported_program: ExportedProgram, node: Node) -> None:
        input_specs = graph_exported_program.graph_signature.input_specs
        arg_to_parameter = self._arg_to_parameter(input_specs)
        function_name = _node_name_as_string(node)

        function_parameter_names = self._get_function_parameter_names(function_name)

        self._log_quantization_parameters(
            node, graph_exported_program, arg_to_parameter, function_parameter_names
        )

    @override
    def conclude(self) -> dict[str, dict[str, Any]]:
        return self._logs

    def _get_function_parameter_names(self, function_name: str) -> tuple[str, ...]:
        schema = attrgetter(function_name.replace("::", "."))(torch.ops).default._schema
        signature = torch.fx.operator_schemas._torchscript_schema_to_signature(schema)
        return tuple(signature.parameters.keys())[1:]

    def _arg_to_parameter(self, input_specs: Sequence[InputSpec]) -> dict[str, str]:
        arg_to_parameter = {}
        # Iterate through the input specs and only grab the parameters.
        for input_spec in input_specs:
            argument = input_spec.arg
            if input_spec.kind in (InputKind.PARAMETER, InputKind.BUFFER):
                name = getattr(argument, "name")
                arg_to_parameter[name] = input_spec.target or ""
        return arg_to_parameter

    def _maybe_load_module_and_params(self, graph_exported_program: ExportedProgram) -> None:
        if not self._module:
            self._module = graph_exported_program.module()
            self._module_named_parameters = frozenset([
                name for name, _ in self._module.named_parameters()
            ])
            self._module_named_buffers = frozenset([
                name for name, _ in self._module.named_buffers()
            ])

    def _log_quantization_parameters(
        self,
        node: Node,
        graph_exported_program: ExportedProgram,
        arg_to_parameter: dict[str, str],
        function_parameter_names: tuple[str, ...],
    ) -> None:
        input_node = node.all_input_nodes[0]
        input_node_name = arg_to_parameter.get(input_node.name, input_node.name)
        parameter_nodes = node.args[1:]

        self._maybe_load_module_and_params(graph_exported_program)

        # Check for fixing mypy union-attr issue
        if not self._module:
            return

        parameter_values: dict[str, Any] = {}
        for parameter_node, function_parameter_name in zip(
            parameter_nodes, function_parameter_names
        ):
            if (isinstance(parameter_node, Node)) and (
                parameter_name := arg_to_parameter.get(parameter_node.name, None)
            ):
                if parameter_name in self._module_named_parameters:
                    parameter_values[function_parameter_name] = self._module.get_parameter(
                        parameter_name
                    )
                elif parameter_name in self._module_named_buffers:
                    parameter_values[function_parameter_name] = self._module.get_buffer(
                        parameter_name
                    )
                else:
                    raise RuntimeError(
                        f"The node {parameter_name} contains neither a buffer or a parameter"
                    )
            else:
                parameter_values[function_parameter_name] = parameter_node

        self._logs.update({input_node_name: parameter_values})


class GraphWrapper:
    """Wrapper class around a ExportedProgram.

    It that facilitates graph operations, such as logging/manipulation.

    Args:
        graph_exported_program: The resulting exported program object resulting from
            a `torch.export.export` function call.
    """

    def __init__(self, graph_exported_program: ExportedProgram):
        self.graph_exported_program = graph_exported_program

    def visit(self, visitor: NodeVisitor[_T]) -> _T:
        """Method to iterate through the nodes of the instance's graph.

        Given a NodeVisitor object check if a given node matches the visitor's
        criteria and if so perform the visitor-defined operations.

        Args:
            visitor: A `NodeVisitor` object that defines some operation that needs to be
                performed on the graph's nodes, or a subset of its nodes.

        Returns:
            The value returned by `visitor.conclude()`
        """
        nodes: list[Node] = self.graph_exported_program.graph_module.graph.nodes
        for node in nodes:
            if visitor.enter(node):
                visitor.visit(self.graph_exported_program, node)
        return visitor.conclude()


def process_dynamo_program(
    dynamo_exported_program: ExportedProgram, graph_operators: Sequence[NodeVisitor[_T]]
) -> tuple[ExportedProgram, dict[str, str], list[Any]]:
    """Function for processing a dynamo program.

    The function accepts a dynamo program object and a list of operations to be performed on the
    graph. It then creates a `GraphWrapper` object that is responsible for invoking the user-defined
    graph operations and logging their output.

    In addition, because any graph transformations might alter the names of input parameter nodes
    of the dynamo graph a helper function is invoked for mapping the old parameter names (before
    transformation) to the new parameter names (after transformation). The function will then
    return the parameter mapping result, and any logs collected from the graph operations.

    Args:
        dynamo_exported_program: A dynamo exported program (representation of a torch module)
        graph_operators: A list of `NodeVisitor` objects that are to be invoked for graph
            modification/logging
    Returns:
        new_old_input_spec_mapping: A dictionary mapping the old input spec parameter names to
            the new input spec parameter names (after some operation has taken place in the graph)
    """
    logs: list[Any] = []

    graph_wrapper = GraphWrapper(dynamo_exported_program)
    for operation in graph_operators:
        output_logs = graph_wrapper.visit(operation)
        logs.append(output_logs)

    original_input_specs = dynamo_exported_program.graph_signature.input_specs
    dynamo_exported_program = dynamo_exported_program.run_decompositions({})

    updated_input_specs = dynamo_exported_program.graph_signature.input_specs
    new_old_input_spec_mapping = get_input_spec_new_old_mapping(
        original_input_specs, updated_input_specs
    )

    return dynamo_exported_program, new_old_input_spec_mapping, logs


def export(
    model: torch.nn.Module,
    data: tuple[torch.Tensor],
    output_directory: str,
    model_name: str,
    graph_preprocessors: None | Sequence[NodeVisitor[Any]] = None,
    model_kwargs: None | dict[str, Any] = None,
) -> pathlib.Path:
    """
    The main export function for retrieving artifacts that can be passed to QNN.

    This function takes an user-defined torch model (which can contain fastforward layers
    and quantizers). It then performs three processing steps:

    1) Extract a dynamo graph (in the form of an exported program object) from the torch model.
        During this process quantization is changed to simulated quantization, leading to the
        graph having `quantize_by_tile` operations followed by `dequantize_by_tile` operations.
    2) Performs operations on the dynamo graph. Here, the following operations are required.
        a) log the parameters of any `quantize_by_tile` functions. These are required for creating
            the QNN encodings file.
        b) remove any quantization operations as these are not required by QNN. This is done in
            sequence, first removing `dequantize_by_tile`, connecting its output to its parent
            `quantize_by_tile` node. Then `quantize_by_tile` operations are removed and their
            output is connected their respective parent node.
    3) The processed dynamo graph is exported to ONNX, using the `torch_onnx` library, and the
        names of inputs/activations/parameters are mapped between dynamo and ONNX. Based on this
        information the encodings file required for QNN is generated.

    Finally the function will store the model in the user-defined `output_directory`, using the
    user-defined `model_name`.

    Args:
        model: A torch module, can contain FF modules.
        data: A torch `Tensor` object that compatible with the model's input shape.
        output_directory: The directory where the artifacts (ONNX model, encodings file) will be
            stored.
        model_name: The name of the model.
        graph_preprocessors: Optionally pass a list of operations that can take place before the
            standard export graph operations required by QNN (parameter logging, removal of
            quantization/dequantization operations).
        model_kwargs: kwargs passed to the model during export.
    Returns:
        _: the path to the output directory where the encodings and ONNX files are stored.
    """
    output_path = pathlib.Path(output_directory) / model_name
    output_path.mkdir(exist_ok=True, parents=True)
    artifact_location = output_path / model_name

    if not graph_preprocessors:
        graph_preprocessors = []

    log_quantization_parameter_operation_location = len(graph_preprocessors)

    default_graph_operators: list[NodeVisitor[Any]] = [
        LogQuantizationParameter("call_function", "fastforward::quantize_by_tile"),
        RemoveFunction("call_function", "fastforward::dequantize_by_tile", input_replace_idx=0),
        RemoveFunction("call_function", "fastforward::quantize_by_tile", input_replace_idx=0),
    ]

    graph_operators = [*graph_preprocessors, *default_graph_operators]

    with export_mode(True):
        dynamo_exported_program = torch.export.export(model, args=data, kwargs=model_kwargs)
        dynamo_exported_program = dynamo_exported_program.run_decompositions({})

    dynamo_exported_program, new_old_input_spec_mapping, raw_logs = process_dynamo_program(
        dynamo_exported_program, graph_operators
    )

    # We only care about the logs from the first operation (LogQuantizationParameter) in the
    # hardcoded graph operations (which will always take place when exporting for QNN).
    # Given that the users could have defined additional operations that happen before these,
    # we keep track of it location in the list.
    quantization_logs = raw_logs[log_quantization_parameter_operation_location]

    torch_onnx_model: onnxscript.ir.Model = torch_onnx.exported_program_to_ir(
        dynamo_exported_program
    )
    proto = onnxscript.ir.to_proto(torch_onnx_model)
    onnx.save(proto, artifact_location.with_suffix(".onnx"))

    used_inputs, unused_inputs = get_inputs(
        torch_onnx_model, quantization_logs, new_old_input_spec_mapping
    )
    used_activations, unused_activations = get_activations(proto, quantization_logs)
    used_parameters, unused_parameters = get_parameters(torch_onnx_model, quantization_logs)

    encodings_dictionary = generate_qnn_encodings_dictionary(
        used_inputs, used_activations, used_parameters, quantization_logs
    )

    with open(artifact_location.with_suffix(".encodings"), "w") as fp:
        json.dump(encodings_dictionary, fp, indent=4)

    onnx.save(
        proto,
        artifact_location.with_suffix(".onnx"),
        save_as_external_data=True,
        all_tensors_to_one_file=False,
        location="filename",
    )

    return output_path
