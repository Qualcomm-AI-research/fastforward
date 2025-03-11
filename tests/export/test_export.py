# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import json
import pathlib

from typing import Any, TypeAlias

import fastforward as ff
import numpy as np
import onnxruntime  # type: ignore[import-untyped]
import pytest
import torch

from fastforward.export.export import (
    GraphWrapper,
    LogQuantizationParameter,
    RemoveFunction,
    RequestNode,
    export,
)
from fastforward.nn.quantizer import Quantizer
from fastforward.quantization.granularity import Granularity
from fastforward.quantization.quant_init import QuantizerCollection

QuantizedModelFixture: TypeAlias = tuple[torch.nn.Module, QuantizerCollection, QuantizerCollection]


@pytest.fixture
def simple_model() -> QuantizedModelFixture:
    class FFNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = ff.nn.QuantizedLinear(10, 10)
            self.relu1 = ff.nn.QuantizedRelu()
            self.fc2 = ff.nn.QuantizedLinear(10, 10)
            self.relu2 = ff.nn.QuantizedRelu()
            self.fc3 = ff.nn.QuantizedLinear(10, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
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


def initialize_quantizers(
    quantizers: QuantizerCollection, quantizer: type[Quantizer], **quantizer_params: Any
) -> None:
    quantizers.initialize(quantizer, **quantizer_params)


def activate_quantizers(
    quant_model: torch.nn.Module,
    data: torch.Tensor,
    activation_quantizers: QuantizerCollection,
    parameter_quantizers: QuantizerCollection,
    param_granularity: Granularity = ff.PerTensor(),
) -> None:
    initialize_quantizers(activation_quantizers, ff.nn.LinearQuantizer, num_bits=8)
    initialize_quantizers(
        parameter_quantizers, ff.nn.LinearQuantizer, num_bits=8, granularity=param_granularity
    )

    with ff.estimate_ranges(quant_model, ff.range_setting.smoothed_minmax):
        quant_model(data)


@pytest.mark.slow
@ff.flags.context(ff.strict_quantization, False)
def test_export_quantized_model(simple_model: QuantizedModelFixture, _seed_prngs: int) -> None:
    # GIVEN a model with quantizer stubs.
    data = torch.randn(32, 10)
    quant_model, activation_quantizers, parameter_quantizers = simple_model
    # WHEN exporting the model before quantizers are activated.
    with ff.export_mode(False):
        exported_graph = torch.export.export(quant_model, args=(data,))

    # THEN the dynamo export should work as with any standard torch modules
    # without the need for switching to export mode.
    assert isinstance(exported_graph, torch.export.exported_program.ExportedProgram)

    # WHEN exporitng the model with activated quantizers.
    activate_quantizers(quant_model, data, activation_quantizers, parameter_quantizers)

    # The export method does not support custom tensors objects (like the QuantizedTensor)
    # and the usage of certain methods (such as the inspect.signature.bind) and will throw
    # an UnsupportedError message if these are present in the code. This test checks that
    # both those issues are resolved when exporting with the export mode set to True.
    with ff.export_mode(True):
        exported_graph = torch.export.export(quant_model, args=(data,))

    # THEN the dynamo export should work when export mode is activated.
    assert isinstance(exported_graph, torch.export.exported_program.ExportedProgram)


@pytest.mark.xfail_due_to_too_new_torch
@pytest.mark.slow
@ff.flags.context(ff.strict_quantization, False)
def test_node_request(simple_model: QuantizedModelFixture, _seed_prngs: int) -> None:
    # GIVEN a quantized model and its exported dynamo graph.
    data = torch.randn(32, 10)
    quant_model, activation_quantizers, parameter_quantizers = simple_model
    activate_quantizers(quant_model, data, activation_quantizers, parameter_quantizers)

    with ff.export_mode(True):
        quantized_model_graph = torch.export.export(quant_model, args=(data,))

    graph_wrapper = GraphWrapper(quantized_model_graph)

    # WHEN using the node request node for a function/module that does not exist
    nodes = graph_wrapper.visit(
        RequestNode("call_function", "nonexisting_module::nonexisting_function")
    )
    # THEN the returned node list should not include any entries
    assert len(nodes) == 0

    # WHEN using the node request node for a node type that does not exist
    nodes = graph_wrapper.visit(
        RequestNode("nonexisting_node_type", "fastforward::dequantize_by_tile")
    )
    # THEN the returned node list should include some entries
    assert len(nodes) == 0

    # GIVEN that the quantized model has a number of quantizers.
    num_quantizers = len([
        module for module in quant_model.modules() if isinstance(module, ff.nn.LinearQuantizer)
    ])

    # WHEN visiting the quantize_by_tile/dequantize_by_tile nodes.
    dequantized_nodes = graph_wrapper.visit(
        RequestNode("call_function", "fastforward::quantize_by_tile")
    )
    quantized_nodes = graph_wrapper.visit(
        RequestNode("call_function", "fastforward::dequantize_by_tile")
    )

    # THEN the resulting lists should have the same number of entries as the number
    # of quantizers in the model.
    assert isinstance(dequantized_nodes, list)
    assert isinstance(quantized_nodes, list)
    assert len(dequantized_nodes) == len(quantized_nodes) == num_quantizers


@pytest.mark.xfail_due_to_too_new_torch
@pytest.mark.slow
@ff.flags.context(ff.strict_quantization, False)
def test_node_removal(simple_model: QuantizedModelFixture, _seed_prngs: int) -> None:
    # GIVEN a model with a number of quantizers
    data = torch.randn(32, 10)
    quant_model, activation_quantizers, parameter_quantizers = simple_model
    non_quant_result = quant_model(data)
    activate_quantizers(quant_model, data, activation_quantizers, parameter_quantizers)

    num_quantizers = len([
        module for module in quant_model.modules() if isinstance(module, ff.nn.LinearQuantizer)
    ])

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


@pytest.mark.xfail_due_to_too_new_torch
@pytest.mark.slow
@ff.flags.context(ff.strict_quantization, False)
@pytest.mark.parametrize("granularity", [ff.PerTensor(), ff.PerChannel(0)])
def test_node_logging(
    granularity: Granularity, simple_model: QuantizedModelFixture, _seed_prngs: int
) -> None:
    # GIVEN a quantized model
    data = torch.randn(32, 10)
    quant_model, activation_quantizers, parameter_quantizers = simple_model
    activate_quantizers(quant_model, data, activation_quantizers, parameter_quantizers, granularity)

    with ff.export_mode(True):
        quantized_model_graph = torch.export.export(quant_model, args=(data,))
        quantized_model_graph = quantized_model_graph.run_decompositions({})

    # WHEN loggign the quantization parameters of the model (knowing beforehand
    # what the quantization targets are).
    graph_wrapper = GraphWrapper(quantized_model_graph)
    quantization_logs = graph_wrapper.visit(
        LogQuantizationParameter("call_function", "fastforward::quantize_by_tile")
    )

    quantization_targets = [
        "x",
        "fc1.weight",
        "fc1.bias",
        "addmm",
        "relu",
        "fc2.weight",
        "fc2.bias",
        "addmm_1",
        "relu_1",
        "fc3.weight",
        "fc3.bias",
        "addmm_2",
    ]

    # THEN the logs should match the targets, and the model
    # quantization parameters should match the values logged in the
    # dictionary.
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


@pytest.mark.xfail_due_to_too_new_torch
@pytest.mark.slow
@ff.flags.context(ff.strict_quantization, False)
def test_ff_model_to_onnx_export(
    tmp_path: pathlib.Path, simple_model: QuantizedModelFixture, _seed_prngs: int
) -> None:
    # GIVEN a model and its initial non-quantized result.
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

    # WHEN using the export pipeline for the quantized model
    # and running inference on the ONNX artifact
    export(quant_model, (data,), str(output_directory), model_name)

    ort_session = onnxruntime.InferenceSession(
        onnx_artifact_location, providers=["CPUExecutionProvider"]
    )

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: data.detach().cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)

    # THEN the results between the original non-quantized model inference
    # and the ONNX inference should match (since the export pipeline removes
    # all quantize_by_tile and dequantize_by_tile ops, the exported model is
    # equivalent to the original model before its quantizers are activated)
    np.testing.assert_allclose(
        non_quantized_result.detach().cpu().numpy(), ort_outs[0], rtol=1e-03, atol=1e-05
    )


@pytest.mark.xfail_due_to_too_new_torch
@pytest.mark.slow
@ff.flags.context(ff.strict_quantization, False)
def test_encodings_file_generation(
    tmp_path: pathlib.Path, simple_model: QuantizedModelFixture, _seed_prngs: int
) -> None:
    # GIVEN a quantized model and its exported encodings file path.
    data = torch.randn(32, 10)
    quant_model, activation_quantizers, parameter_quantizers = simple_model

    model_name = "test_encodings_file_generation"
    output_directory = tmp_path
    output_model_directory = pathlib.Path(output_directory) / model_name

    encodings_file_path = (pathlib.Path(output_model_directory) / model_name).with_suffix(
        ".encodings"
    )

    expected_quantized_inputs = ["x"]
    expected_quantized_activations = ["addmm", "relu", "addmm_1", "relu_1", "addmm_2"]

    expected_quantized_params = [
        "fc1.weight",
        "fc1.bias",
        "fc2.weight",
        "fc2.bias",
        "fc3.weight",
        "fc3.bias",
    ]

    expected_nested_dictionary_keys = sorted([
        "bitwidth",
        "dtype",
        "is_symmetric",
        "min",
        "max",
        "offset",
        "scale",
    ])

    activate_quantizers(quant_model, data, activation_quantizers, parameter_quantizers)

    # WHEN exporting the quantized model
    export(quant_model, (data,), str(output_directory), model_name)

    # THEN we expect a json file that can be loaded as a dictionary.
    with open(encodings_file_path) as f:
        encodings_dictionary = json.load(f)

    # THEN the structure of the top level of the dictionary should have set keys
    assert sorted(encodings_dictionary.keys()) == ["activation_encodings", "param_encodings"]

    activation_encodings_dictionary = encodings_dictionary["activation_encodings"]
    param_encodings_dictionary = encodings_dictionary["param_encodings"]

    # THEN all the nested dictionary keys should also have the expected keys
    for value in activation_encodings_dictionary.values():
        assert sorted(value[0].keys()) == expected_nested_dictionary_keys

    for value in param_encodings_dictionary.values():
        assert sorted(value[0].keys()) == expected_nested_dictionary_keys

    # THEN the number of entries for activations and parameters should match the number of
    # expected entries respectively
    assert sorted(activation_encodings_dictionary.keys()) == sorted(
        expected_quantized_inputs + expected_quantized_activations
    )
    assert sorted(param_encodings_dictionary) == sorted(expected_quantized_params)


@pytest.mark.xfail_due_to_too_new_torch
@ff.flags.context(ff.strict_quantization, False)
@pytest.mark.parametrize("granularity", [ff.PerTensor(), ff.PerChannel(0)])
def test_export_function(
    tmp_path: pathlib.Path,
    granularity: Granularity,
    simple_model: QuantizedModelFixture,
    _seed_prngs: int,
) -> None:
    # GIVEN a quantized model.
    data = torch.randn(32, 10)
    quant_model, activation_quantizers, parameter_quantizers = simple_model
    output_directory = tmp_path
    model_name = "test_export_function"

    output_model_directory = pathlib.Path(output_directory) / model_name

    activate_quantizers(quant_model, data, activation_quantizers, parameter_quantizers, granularity)

    # WHEN exporting the quantized model
    export(quant_model, (data,), str(output_directory), model_name)
    onnx_file_path = (output_model_directory / model_name).with_suffix(".onnx")
    encodings_file_path = (output_model_directory / model_name).with_suffix(".encodings")

    # THEN we expect that ONNX/encodings files are created and they are
    # not empty.
    assert output_model_directory.is_dir()
    assert onnx_file_path.is_file()
    assert encodings_file_path.is_file()

    assert onnx_file_path.stat().st_size > 0
    assert encodings_file_path.stat().st_size > 0
