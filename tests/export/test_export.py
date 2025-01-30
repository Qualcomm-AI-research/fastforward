# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import pathlib
import unittest

import numpy as np
import onnx
import onnxruntime  # type: ignore[import-untyped]
import onnxscript
import pytest
import torch
import torch_onnx  # type: ignore[import-untyped]

import fastforward as ff

from fastforward.export._export_helpers import (
    get_activations,
    get_inputs,
    get_parameters,
)
from fastforward.export.export import (
    GraphWrapper,
    LogQuantizationParameter,
    RemoveFunction,
    RequestNode,
    export,
)


@pytest.fixture
def simple_model():
    class FFNet(torch.nn.Module):
        def __init__(self):
            super(FFNet, self).__init__()
            self.fc1 = ff.nn.QuantizedLinear(10, 10)
            self.relu1 = ff.nn.QuantizedRelu()
            self.fc2 = ff.nn.QuantizedLinear(10, 10)
            self.relu2 = ff.nn.QuantizedRelu()
            self.fc3 = ff.nn.QuantizedLinear(10, 10)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.fc3(x)

            return x

    quant_model = FFNet()

    activation_quantizers = ff.find_quantizers(quant_model, "**/[quantizer:activation/output]")
    activation_quantizers |= ff.find_quantizers(quant_model, "fc1/[quantizer:activation/input]")
    parameter_quantizers = ff.find_quantizers(quant_model, "**/[quantizer:parameter]")

    return quant_model, activation_quantizers, parameter_quantizers


def initialize_quantizers(quantizers, quantizer, **quantizer_params):
    quantizers.initialize(quantizer, **quantizer_params)


def activate_quantizers(
    quant_model, data, activation_quantizers, parameter_quantizers, param_granularity=ff.PerTensor()
):
    initialize_quantizers(activation_quantizers, ff.nn.LinearQuantizer, num_bits=8)
    initialize_quantizers(
        parameter_quantizers, ff.nn.LinearQuantizer, num_bits=8, granularity=param_granularity
    )

    with ff.estimate_ranges(quant_model, ff.range_setting.smoothed_minmax):
        quant_model(data)


@pytest.mark.slow
@ff.flags.context(ff.strict_quantization, False)
def test_export_quantized_model(simple_model):
    data = torch.randn(32, 10)
    quant_model, activation_quantizers, parameter_quantizers = simple_model
    # Check that the export mode works for model with only quantizer stubs.
    with ff.export_mode(False):
        exported_graph = torch.export.export(quant_model, args=(data,))

    activate_quantizers(quant_model, data, activation_quantizers, parameter_quantizers)

    with ff.export_mode(True):
        quant_model(data)

    # The export method does not support custom tensors objects (like the QuantizedTensor)
    # and the usage of certain methods (such as the inspect.signature.bind) and will throw
    # an UnsupportedError message if these are present in the code. This test checks that
    # both those issues are resolved when exporting with the export mode set to True.
    with ff.export_mode(True):
        exported_graph = torch.export.export(quant_model, args=(data,))

    assert isinstance(exported_graph, torch.export.exported_program.ExportedProgram)


@pytest.mark.slow
@ff.flags.context(ff.strict_quantization, False)
def test_node_request(simple_model):
    data = torch.randn(32, 10)
    quant_model, activation_quantizers, parameter_quantizers = simple_model
    activate_quantizers(quant_model, data, activation_quantizers, parameter_quantizers)

    with ff.export_mode(True):
        quantized_model_graph = torch.export.export(quant_model, args=(data,))

    graph_wrapper = GraphWrapper(quantized_model_graph)

    assert not graph_wrapper.visit(
        RequestNode("call_function", "nonexisting_module::nonexisting_function")
    )
    assert not graph_wrapper.visit(
        RequestNode("nonexisting_node_type", "fastforward::dequantize_by_tile")
    )

    num_quantizers = len(
        [module for module in quant_model.modules() if isinstance(module, ff.nn.LinearQuantizer)]
    )

    dequantized_nodes = graph_wrapper.visit(
        RequestNode("call_function", "fastforward::quantize_by_tile")
    )
    quantized_nodes = graph_wrapper.visit(
        RequestNode("call_function", "fastforward::dequantize_by_tile")
    )

    assert isinstance(dequantized_nodes, list)
    assert isinstance(quantized_nodes, list)
    assert len(dequantized_nodes) == len(quantized_nodes) == num_quantizers


@pytest.mark.slow
@ff.flags.context(ff.strict_quantization, False)
def test_node_removal(simple_model):
    # GIVEN a model wuth a number of quantizers
    data = torch.randn(32, 10)
    quant_model, activation_quantizers, parameter_quantizers = simple_model
    non_quant_result = quant_model(data)
    activate_quantizers(quant_model, data, activation_quantizers, parameter_quantizers)

    num_quantizers = len(
        [module for module in quant_model.modules() if isinstance(module, ff.nn.LinearQuantizer)]
    )

    # WHEN exporting the model's dynamo graph there should be
    # a number of nodes corresponding to the quantizer operations
    # (quantize_by_tile/dequantize_by_tile)

    with ff.export_mode(True):
        quantized_model_graph = torch.export.export(quant_model, args=(data,))

    graph_wrapper = GraphWrapper(quantized_model_graph)

    dequantized_nodes = graph_wrapper.visit(
        RequestNode("call_function", "fastforward::quantize_by_tile")
    )
    quantized_nodes = graph_wrapper.visit(
        RequestNode("call_function", "fastforward::dequantize_by_tile")
    )

    assert isinstance(dequantized_nodes, list)
    assert isinstance(quantized_nodes, list)
    assert len(dequantized_nodes) == len(quantized_nodes) == num_quantizers > 0

    # THEN using the RemoveFunction to remove the dequantize_by_tile
    # nodes, we ensure that these nodes are removed, while the
    # quantize_by_tile nodes remain in the graph

    graph_wrapper.visit(RemoveFunction("call_function", "fastforward::dequantize_by_tile", 0))
    dequantized_nodes_after_removal = graph_wrapper.visit(
        RequestNode("call_function", "fastforward::dequantize_by_tile")
    )
    assert isinstance(dequantized_nodes_after_removal, list)
    assert not dequantized_nodes_after_removal

    quantized_nodes = graph_wrapper.visit(
        RequestNode("call_function", "fastforward::quantize_by_tile")
    )
    assert isinstance(quantized_nodes, list)
    assert len(quantized_nodes) == num_quantizers

    # THEN using the RemoveFunction to remove the quantize_by_tile
    # nodes, we ensure that these nodes are removed

    graph_wrapper.visit(RemoveFunction("call_function", "fastforward::quantize_by_tile", 0))
    quantized_nodes_after_removal = graph_wrapper.visit(
        RequestNode("call_function", "fastforward::quantize_by_tile")
    )
    assert isinstance(quantized_nodes_after_removal, list)
    assert not quantized_nodes_after_removal

    # Finally we check that now that all the quantize_by_tile/dequantize_by_tile
    # nodes are removed the graph output will produce the same results as the
    # non-quantized model (since it no longer contains any quantization ops)
    assert (non_quant_result == quantized_model_graph.module()(data)).all()


@pytest.mark.slow
@ff.flags.context(ff.strict_quantization, False)
@pytest.mark.parametrize("granularity", [ff.PerTensor(), ff.PerChannel(0)])
def test_node_logging(granularity, simple_model):
    data = torch.randn(32, 10)
    quant_model, activation_quantizers, parameter_quantizers = simple_model
    activate_quantizers(quant_model, data, activation_quantizers, parameter_quantizers, granularity)

    with ff.export_mode(True):
        quantized_model_graph = torch.export.export(quant_model, args=(data,))

    graph_wrapper = GraphWrapper(quantized_model_graph)
    quantization_logs = graph_wrapper.visit(
        LogQuantizationParameter("call_function", "fastforward::quantize_by_tile")
    )

    quantization_targets = [
        "x",
        "fc1.weight",
        "fc1.bias",
        "linear",
        "relu",
        "fc2.weight",
        "fc2.bias",
        "linear_1",
        "relu_1",
        "fc3.weight",
        "fc3.bias",
        "linear_2",
    ]

    assert isinstance(quantization_logs, dict)
    assert len(quantization_targets) == len(quantization_logs)
    assert set(quantization_targets) <= set(quantization_logs)

    for parameter_name, parameter_value in quantization_logs.items():
        # Only check the parameter (weight, bias) quantizers.
        # Associating the torch module input/quantizers with the
        # dynamo names is more involved, so it is omitted for now.
        if parameter_name.split(".")[-1] in ("weight", "bias"):
            quantizer_name = "_".join([parameter_name, "quantizer"])
            quantizer = quant_model.get_submodule(quantizer_name)

            assert torch.equal(quantizer.offset, parameter_value["offset"])
            assert torch.equal(quantizer.scale, parameter_value["scale"])
            assert quantizer.num_bits == parameter_value["num_bits"]


@pytest.mark.slow
@ff.flags.context(ff.strict_quantization, False)
def test_ff_model_to_onnx_export(tmp_path, simple_model):
    data = torch.randn(32, 10)
    quant_model, activation_quantizers, parameter_quantizers = simple_model
    non_quantized_result = quant_model(data)

    model_name = "test_ff_model_to_onnx_export"
    output_directory = tmp_path
    output_model_directory = pathlib.Path(output_directory) / model_name

    onnx_artifact_location = (pathlib.Path(output_model_directory) / model_name).with_suffix(
        ".onnx"
    )

    activate_quantizers(quant_model, data, activation_quantizers, parameter_quantizers)

    export(quant_model, (data,), output_directory, model_name)

    ort_session = onnxruntime.InferenceSession(
        onnx_artifact_location, providers=["CPUExecutionProvider"]
    )

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(data)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(non_quantized_result), ort_outs[0], rtol=1e-03, atol=1e-05)


@pytest.mark.slow
@pytest.mark.parametrize("quantize_input", [True, False])
@pytest.mark.parametrize("quantize_outputs", [True, False])
@pytest.mark.parametrize("quantize_weights", [True, False])
@pytest.mark.parametrize("quantize_bias", [True, False])
def test_onnx_parameter_collection(
    quantize_input, quantize_outputs, quantize_weights, quantize_bias
):
    def create_mock_sym_tensor(name):
        mock_symbolic_tensor = unittest.mock.Mock(torch_onnx._tensors.SymbolicTensor)
        mock_symbolic_tensor.name = name
        return mock_symbolic_tensor

    def create_mock_proto_node(name):
        mock_proto_node = unittest.mock.Mock(onnx.onnx_ml_pb2.NodeProto)
        mock_proto_node.output = [name]
        return mock_proto_node

    def create_mock_onnx_and_proto_objects(inputs, activations, parameters):
        mock_torch_onnx_model = unittest.mock.Mock(onnxscript.ir.Model)
        mock_proto = unittest.mock.Mock(onnx.onnx_ml_pb2.ModelProto)
        mock_proto_graph = unittest.mock.Mock(onnx.onnx_ml_pb2.GraphProto)

        mock_torch_onnx_model.graph.inputs = []
        mock_torch_onnx_model.graph.initializers = []
        mock_proto.graph = mock_proto_graph
        mock_proto.graph.node = []

        for input_ in inputs:
            mock_torch_onnx_model.graph.inputs.append(create_mock_sym_tensor(input_))

        for activation in activations:
            mock_proto.graph.node.append(create_mock_proto_node(activation))

        for parameter in parameters:
            mock_torch_onnx_model.graph.initializers.append(parameter)

        return mock_torch_onnx_model, mock_proto

    def _set_pop_key(input_set, key):
        input_set.remove(key)
        return key

    quantization_logs = {}
    new_to_old_input_spec_dictionary = {"arg6_1": "arg6_1"}

    unused_activation_names = set(["addmm", "relu", "t"])
    unused_parameter_names = set(["fc1.weight", "fc1.bias"])

    unused_input_names = set(["arg6_1"])

    used_activation_names, used_parameter_names, used_input_names = set(), set(), set()
    mock_torch_onnx_model, mock_proto = create_mock_onnx_and_proto_objects(
        unused_input_names, unused_activation_names, unused_parameter_names
    )

    if quantize_outputs:
        used_activation_names.add(_set_pop_key(unused_activation_names, "addmm"))
        used_activation_names.add(_set_pop_key(unused_activation_names, "relu"))

        quantization_logs["addmm"] = 0
        quantization_logs["relu"] = 0

    if quantize_input:
        used_input_names.add(_set_pop_key(unused_input_names, "arg6_1"))

        quantization_logs["arg18_1"] = 0
        new_to_old_input_spec_dictionary["arg6_1"] = "arg18_1"

    if quantize_weights:
        used_parameter_names.add(_set_pop_key(unused_parameter_names, "fc1.weight"))
        quantization_logs["fc1.weight"] = 0

    if quantize_bias:
        used_parameter_names.add(_set_pop_key(unused_parameter_names, "fc1.bias"))
        quantization_logs["fc1.bias"] = 0

    assert get_inputs(
        mock_torch_onnx_model, quantization_logs, new_to_old_input_spec_dictionary
    ) == (used_input_names, unused_input_names)
    assert get_activations(mock_proto, quantization_logs) == (
        used_activation_names,
        unused_activation_names,
    )
    assert get_parameters(mock_torch_onnx_model, quantization_logs) == (
        used_parameter_names,
        unused_parameter_names,
    )


@ff.flags.context(ff.strict_quantization, False)
@pytest.mark.parametrize("granularity", [ff.PerTensor(), ff.PerChannel(0)])
def test_export_function(tmp_path, granularity, simple_model):
    data = torch.randn(32, 10)
    quant_model, activation_quantizers, parameter_quantizers = simple_model
    output_directory = tmp_path
    model_name = "test_export_function"

    output_model_directory = pathlib.Path(output_directory) / model_name

    activate_quantizers(quant_model, data, activation_quantizers, parameter_quantizers, granularity)

    export(quant_model, (data,), output_directory, model_name)
    onnx_file_path = (output_model_directory / model_name).with_suffix(".onnx")
    encodings_file_path = (output_model_directory / model_name).with_suffix(".encodings")

    assert output_model_directory.is_dir()
    assert onnx_file_path.is_file()
    assert encodings_file_path.is_file()

    assert onnx_file_path.stat().st_size > 0
    assert encodings_file_path.stat().st_size > 0
