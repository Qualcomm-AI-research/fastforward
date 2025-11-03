# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import pathlib
import warnings

from typing import Any, Generator, TypeAlias

import fastforward as ff
import numpy as np
import onnx
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
from fastforward.quantization.granularity import Granularity
from fastforward.quantization.quant_init import QuantizerCollection
from fastforward.testing.initialization import initialize_quantizers_to_linear_quantizer
from packaging import version

QuantizedModelFixture: TypeAlias = tuple[torch.nn.Module, QuantizerCollection, QuantizerCollection]


@pytest.fixture(autouse=True)
def disable_warnings() -> Generator[None, None, None]:
    """Disables warnings for the duration of the test module."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="onnxscript.*")
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch._dynamo.*")
        yield


@pytest.mark.slow
def test_export_quantized_model(simple_model: QuantizedModelFixture, _seed_prngs: int) -> None:
    # GIVEN a model with quantizer stubs.
    data = torch.randn(32, 10)
    quant_model, activation_quantizers, parameter_quantizers = simple_model
    # WHEN exporting the model before quantizers are activated.
    with ff.export_mode(False), ff.strict_quantization(False):
        exported_graph = torch.export.export(quant_model, args=(data,)).run_decompositions()

    # THEN the dynamo export should work as with any standard torch modules
    # without the need for switching to export mode.
    assert isinstance(exported_graph, torch.export.exported_program.ExportedProgram)

    # WHEN exporitng the model with activated quantizers.
    estimate_model_ranges = initialize_quantizers_to_linear_quantizer(
        quant_model, activation_quantizers, parameter_quantizers
    )
    estimate_model_ranges(data)

    # The export method does not support custom tensors objects (like the QuantizedTensor)
    # and the usage of certain methods (such as the inspect.signature.bind) and will throw
    # an UnsupportedError message if these are present in the code. This test checks that
    # both those issues are resolved when exporting with the export mode set to True.
    with ff.export_mode(True), ff.strict_quantization(False):
        exported_graph = torch.export.export(quant_model, args=(data,)).run_decompositions()

    # THEN the dynamo export should work when export mode is activated.
    assert isinstance(exported_graph, torch.export.exported_program.ExportedProgram)


@pytest.mark.slow
def test_node_request(simple_model: QuantizedModelFixture, _seed_prngs: int) -> None:
    # GIVEN a quantized model and its exported dynamo graph.
    data = torch.randn(32, 10)
    quant_model, activation_quantizers, parameter_quantizers = simple_model
    estimate_model_ranges = initialize_quantizers_to_linear_quantizer(
        quant_model, activation_quantizers, parameter_quantizers
    )
    estimate_model_ranges(data)

    with ff.export_mode(True), ff.strict_quantization(False):
        quantized_model_graph = torch.export.export(quant_model, args=(data,)).run_decompositions()

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


@pytest.mark.slow
def test_node_removal(simple_model: QuantizedModelFixture, _seed_prngs: int) -> None:
    # GIVEN a model with a number of quantizers
    data = torch.randn(32, 10)
    quant_model, activation_quantizers, parameter_quantizers = simple_model
    with ff.strict_quantization(False):
        non_quant_result = quant_model(data)

    estimate_model_ranges = initialize_quantizers_to_linear_quantizer(
        quant_model, activation_quantizers, parameter_quantizers
    )
    estimate_model_ranges(data)

    num_quantizers = len([
        module for module in quant_model.modules() if isinstance(module, ff.nn.LinearQuantizer)
    ])

    # WHEN exporting the model's dynamo graph there should be
    # a number of nodes corresponding to the quantizer operations
    # (quantize_by_tile/dequantize_by_tile)

    with ff.export_mode(True), ff.strict_quantization(False):
        quantized_model_graph = torch.export.export(quant_model, args=(data,)).run_decompositions()

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
@pytest.mark.parametrize("granularity", [ff.PerTensor(), ff.PerChannel(0)])
def test_node_logging(
    granularity: Granularity, simple_model: QuantizedModelFixture, _seed_prngs: int
) -> None:
    # GIVEN a quantized model
    data = torch.randn(32, 10)
    quant_model, activation_quantizers, parameter_quantizers = simple_model

    estimate_model_ranges = initialize_quantizers_to_linear_quantizer(
        quant_model, activation_quantizers, parameter_quantizers, granularity_parameters=granularity
    )
    estimate_model_ranges(data)

    with ff.export_mode(True), ff.strict_quantization(False):
        quantized_model_graph = torch.export.export(quant_model, args=(data,)).run_decompositions()

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


@pytest.mark.slow
def test_ff_model_to_onnx_export(
    tmp_path: pathlib.Path, simple_model: QuantizedModelFixture, _seed_prngs: int
) -> None:
    # GIVEN a model and its initial non-quantized result.
    data = torch.randn(32, 10)
    quant_model, activation_quantizers, parameter_quantizers = simple_model
    with ff.strict_quantization(False):
        non_quantized_result = quant_model(data)

    model_name = "test_ff_model_to_onnx_export"
    output_directory = tmp_path / model_name
    output_model_directory = output_directory / model_name

    onnx_artifact_location = output_model_directory.with_suffix(".onnx")

    estimate_model_ranges = initialize_quantizers_to_linear_quantizer(
        quant_model, activation_quantizers, parameter_quantizers
    )
    estimate_model_ranges(data)

    # WHEN using the export pipeline for the quantized model
    # and running inference on the ONNX artifact
    export(
        quant_model,
        (data,),
        output_directory,
        model_name,
        input_names=["new_x"],
        output_names=["new_output"],
    )

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


@pytest.mark.slow
@pytest.mark.slow
@pytest.mark.parametrize(
    "names",
    [
        (None, None),
        (["input_1", "input_2", "input_3", "input_4"], ["output_1", "output_2", "output_3"]),
    ],
)
def test_graph_io_renaming_valid(
    multi_input_output_model: QuantizedModelFixture,
    tmp_path: pathlib.Path,
    names: tuple[None, ...] | tuple[list[str], ...],
) -> None:
    # GIVEN a quantized model with multiple inputs/outputs
    data_x = torch.randn(32, 10)
    data_y = torch.randn(32, 10)
    subtract_from_x = torch.ones(32, 10)
    add_to_y = torch.ones(32, 10)

    input_names, output_names = names

    quant_model, activation_quantizers, parameter_quantizers = multi_input_output_model
    with ff.strict_quantization(False):
        estimate_model_ranges = initialize_quantizers_to_linear_quantizer(
            quant_model, activation_quantizers, parameter_quantizers
        )
        estimate_model_ranges(data_x, data_y, subtract_from_x=subtract_from_x, add_to_y=add_to_y)

    model_name = "test_model"
    output_directory = tmp_path / model_name
    output_model_directory = output_directory / model_name
    onnx_artifact_location = output_model_directory.with_suffix(".onnx")

    # WHEN exporting the model with new (valid) input/output names or when these are set to None.
    export(
        quant_model,
        (data_x, data_y),
        output_directory,
        model_name,
        model_kwargs={"subtract_from_x": subtract_from_x, "add_to_y": add_to_y},
        input_names=input_names,
        output_names=output_names,
    )

    ort_session = onnxruntime.InferenceSession(
        onnx_artifact_location, providers=["CPUExecutionProvider"]
    )

    graph_inputs = [input_.name for input_ in ort_session.get_inputs()]
    graph_outputs = [output_.name for output_ in ort_session.get_outputs()]

    # WHEN the input/output names are not user defined (None is passed to the `export` function)
    # we hardcode the input/output names to assert the result.
    if input_names is None:
        input_names = ["x", "y", "subtract_from_x", "add_to_y"]

    if output_names is None:
        # The `ff` prefix and `_0` suffix are added from the `_fix_onnx_names` function, to
        # deal with the problem of duplicate nodes in QNN.
        output_names = ["add_1", "addmm", "addmm_1"]

    # THEN the graph input/output names should match the user defined input/output names.
    assert graph_inputs == input_names
    assert graph_outputs == output_names


@pytest.mark.slow
@pytest.mark.parametrize(
    "names",
    [
        (["input_1", "input_2"], ["output_1", "output_2", "output_3"]),
        (["input_1", "input_2", "input_3", "input_4"], ["output_1", "output_2"]),
    ],
)
def test_graph_io_renaming_invalid(
    multi_input_output_model: QuantizedModelFixture,
    tmp_path: pathlib.Path,
    names: tuple[None, ...] | tuple[list[str], ...],
) -> None:
    # GIVEN a quantized model with multiple inputs/outputs
    data_x = torch.randn(32, 10)
    data_y = torch.randn(32, 10)
    subtract_from_x = torch.ones(32, 10)
    add_to_y = torch.ones(32, 10)

    input_names, output_names = names

    quant_model, activation_quantizers, parameter_quantizers = multi_input_output_model
    with ff.strict_quantization(False):
        estimate_model_ranges = initialize_quantizers_to_linear_quantizer(
            quant_model, activation_quantizers, parameter_quantizers
        )
        estimate_model_ranges(data_x, data_y, subtract_from_x=subtract_from_x, add_to_y=add_to_y)

    # WHEN exporting the model overriding its input/output names with invalid ones.
    model_name = "test_model"
    output_directory = tmp_path / model_name

    # THEN the export function should raise an error since the number of user defined
    # inputs/outputs do not match the number of graph inputs/outputs.
    with pytest.raises(ValueError):
        export(
            quant_model,
            (data_x, data_y),
            output_directory,
            model_name,
            model_kwargs={"subtract_from_x": subtract_from_x, "add_to_y": add_to_y},
            input_names=input_names,
            output_names=output_names,
        )


@pytest.mark.slow
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
    model_name = "test_export_function"
    output_directory = tmp_path / model_name
    output_model_directory = output_directory / model_name
    onnx_file_path = output_model_directory.with_suffix(".onnx")
    encodings_file_path = output_model_directory.with_suffix(".encodings")

    estimate_model_ranges = initialize_quantizers_to_linear_quantizer(
        quant_model, activation_quantizers, parameter_quantizers, granularity_parameters=granularity
    )
    estimate_model_ranges(data)

    # WHEN exporting the quantized model
    export(quant_model, (data,), output_directory, model_name)

    # THEN we expect that ONNX/encodings files are created and they are
    # not empty.
    assert onnx_file_path.is_file()
    assert encodings_file_path.is_file()

    assert onnx_file_path.stat().st_size > 0
    assert encodings_file_path.stat().st_size > 0


@pytest.mark.slow
def test_export_model_with_ctx_manager(
    tmp_path: pathlib.Path,
    _seed_prngs: int,
) -> None:
    class TestModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)

        def forward(self, x: torch.Tensor) -> Any:
            with ff.strict_quantization(False):
                return self.linear(x)

    # GIVEN a model with a context manager
    quant_model = TestModel()
    data = torch.randn(32, 10)
    model_name = "model_with_ctx_manager"
    output_directory = tmp_path / model_name
    # WHEN export should not fail
    export(quant_model, (data,), output_directory, model_name)


@pytest.mark.slow
@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse("2.6.0"), reason="requires PyTorch > 2.5.0"
)
def test_export_with_optimization_options(
    tmp_path: pathlib.Path, simple_model: QuantizedModelFixture, _seed_prngs: int
) -> None:
    # GIVEN a quantized model and some ONNX export options
    data = torch.randn(32, 10)
    quant_model, activation_quantizers, parameter_quantizers = simple_model
    model_name = "test_export_with_optimization_options"

    estimate_model_ranges = initialize_quantizers_to_linear_quantizer(
        quant_model, activation_quantizers, parameter_quantizers
    )

    estimate_model_ranges(data)

    # WHEN exporting without optimization AND with optimization
    unoptimized_dir = tmp_path / model_name / "unoptimized"
    export(
        quant_model,
        (data,),
        unoptimized_dir,
        model_name,
        onnx_export_options={"optimize": False, "do_constant_folding": False},
    )

    optimized_dir = tmp_path / model_name / "optimized"
    export(
        quant_model,
        (data,),
        optimized_dir,
        model_name,
        onnx_export_options={"optimize": True, "do_constant_folding": True},
    )

    # THEN both models should be created successfully and the optimized
    # model should have less nodes than the unoptimized model.
    unoptimized_model = onnx.load(unoptimized_dir / f"{model_name}.onnx")
    optimized_model = onnx.load(optimized_dir / f"{model_name}.onnx")

    unopt_nodes = len(unoptimized_model.graph.node)
    opt_nodes = len(optimized_model.graph.node)

    assert 0 < opt_nodes < unopt_nodes
