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

from fastforward.export.export import export
from fastforward.quantization.granularity import Granularity
from fastforward.quantization.quant_init import QuantizerCollection
from fastforward.testing.initialization import initialize_quantizers_to_linear_quantizer
from tests._core_package_version_utils import OPSET_VERSION, is_torch_version_at_least

QuantizedModelFixture: TypeAlias = tuple[torch.nn.Module, QuantizerCollection, QuantizerCollection]
ONNX_EXPORT_OPTIONS = {"opset_version": OPSET_VERSION}


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
    quant_model.eval()

    # WHEN using the export pipeline for the quantized model
    # and running inference on the ONNX artifact
    export(
        quant_model,
        (data,),
        output_directory,
        model_name,
        input_names=["new_x"],
        output_names=["new_output"],
        onnx_export_options=ONNX_EXPORT_OPTIONS,
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
@pytest.mark.skipif(
    not is_torch_version_at_least("2.7"),
    reason=(
        "torch <= 2.6 does not re-infer the QuantizeLinear output type after the "
        "FF quantize_by_tile lowering, leaving a stale float value_info that ONNX "
        "Runtime rejects at load time."
    ),
)
@pytest.mark.parametrize(
    ("weight_bits", "activation_bits", "rtol", "atol"),
    [
        (8, 8, 1e-2, 5e-3),
        (4, 4, 1e-2, 5e-3),
        (4, 8, 1e-2, 5e-3),
    ],
    ids=["w8a8", "w4a4", "w4a8"],
)
def test_ff_model_to_onnx_qdq_export_matches_torch(
    tmp_path: pathlib.Path,
    simple_model: QuantizedModelFixture,
    _seed_prngs: int,
    weight_bits: int,
    activation_bits: int,
    rtol: float,
    atol: float,
) -> None:
    # GIVEN a simple quantized model configured for the requested W/A bitwidths.
    data = torch.randn(32, 10)
    quant_model, activation_quantizers, parameter_quantizers = simple_model

    activation_quantizers.initialize(
        ff.nn.LinearQuantizer,
        num_bits=activation_bits,
        granularity=ff.PerTensor(),
        symmetric=False,
    )
    parameter_quantizers.initialize(
        ff.nn.LinearQuantizer,
        num_bits=weight_bits,
        granularity=ff.PerChannel(0),
        symmetric=True,
    )
    with ff.estimate_ranges(quant_model, ff.range_setting.smoothed_minmax):
        quant_model(data)

    with ff.strict_quantization(False), torch.no_grad():
        ff_output = quant_model(data)

    model_name = f"test_ff_qdq_export_matches_torch_w{weight_bits}a{activation_bits}"
    output_directory = tmp_path / model_name
    output_model_directory = output_directory / model_name
    onnx_artifact_location = output_model_directory.with_suffix(".onnx")

    # WHEN exporting using the qnn/onnx_qdq pipeline.
    # No opset_version is set: the QDQ wrapper defaults to opset 21, the minimum
    # required for INT4/INT16 storage in QuantizeLinear/DequantizeLinear.
    export(
        quant_model,
        (data,),
        output_directory,
        model_name,
        target="qnn",
        format="onnx_qdq",
    )

    # THEN the ONNX artifact exists, is structurally valid, and contains the
    # expected Q/DQ nodes — one pair per active FF quantizer.
    assert onnx_artifact_location.is_file()
    onnx_model = onnx.load(onnx_artifact_location)

    # THEN The saved model must be structurally well-formed.
    onnx.checker.check_model(onnx_model)

    # THEN (Quantize|Dequantize)Linear nodes are included in the graph
    node_types = {node.op_type for node in onnx_model.graph.node}
    assert "QuantizeLinear" in node_types
    assert "DequantizeLinear" in node_types

    # THEN Each active FF quantizer should lower to exactly one Q + one DQ node.
    # If a future change lowers only a subset of FF ops, the count mismatch
    # surfaces here rather than as silent numerical drift in the parity check.
    expected_quantizer_count = len(activation_quantizers) + len(parameter_quantizers)
    q_count = sum(1 for n in onnx_model.graph.node if n.op_type == "QuantizeLinear")
    dq_count = sum(1 for n in onnx_model.graph.node if n.op_type == "DequantizeLinear")
    assert q_count == expected_quantizer_count, (
        f"expected {expected_quantizer_count} QuantizeLinear nodes "
        f"(one per active FF quantizer), got {q_count}"
    )
    assert dq_count == expected_quantizer_count, (
        f"expected {expected_quantizer_count} DequantizeLinear nodes "
        f"(one per active FF quantizer), got {dq_count}"
    )

    ort_session = onnxruntime.InferenceSession(
        onnx_artifact_location, providers=["CPUExecutionProvider"]
    )
    ort_inputs = {ort_session.get_inputs()[0].name: data.detach().cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)

    # THEN ONNX output should match the FastForward quantized model output.
    np.testing.assert_allclose(
        ff_output.dequantize().detach().cpu().numpy(), ort_outs[0], rtol=rtol, atol=atol
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
    quant_model.eval()

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
        onnx_export_options=ONNX_EXPORT_OPTIONS,
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
    quant_model.eval()

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
            onnx_export_options=ONNX_EXPORT_OPTIONS,
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
    quant_model.eval()

    # WHEN exporting the quantized model
    export(
        quant_model,
        (data,),
        output_directory,
        model_name,
        onnx_export_options=ONNX_EXPORT_OPTIONS,
    )

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
    quant_model.eval()

    # WHEN exporting the model.
    # THEN the export completes without raising an exception.
    export(
        quant_model,
        (data,),
        output_directory,
        model_name,
        onnx_export_options=ONNX_EXPORT_OPTIONS,
    )


@pytest.mark.slow
@pytest.mark.skipif(not is_torch_version_at_least("2.6.0"), reason="requires PyTorch > 2.5.0")
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
    quant_model.eval()

    # WHEN exporting without optimization AND with optimization
    unoptimized_dir = tmp_path / model_name / "unoptimized"
    export(
        quant_model,
        (data,),
        unoptimized_dir,
        model_name,
        onnx_export_options={
            "optimize": False,
            "do_constant_folding": False,
            "opset_version": ONNX_EXPORT_OPTIONS["opset_version"],
        },
    )

    optimized_dir = tmp_path / model_name / "optimized"
    export(
        quant_model,
        (data,),
        optimized_dir,
        model_name,
        onnx_export_options={
            "optimize": True,
            "do_constant_folding": True,
            "opset_version": ONNX_EXPORT_OPTIONS["opset_version"],
        },
    )

    # THEN both models should be created successfully and the optimized
    # model should have less nodes than the unoptimized model.
    unoptimized_model = onnx.load(unoptimized_dir / f"{model_name}.onnx")
    optimized_model = onnx.load(optimized_dir / f"{model_name}.onnx")

    unopt_nodes = len(unoptimized_model.graph.node)
    opt_nodes = len(optimized_model.graph.node)

    assert 0 < opt_nodes < unopt_nodes
