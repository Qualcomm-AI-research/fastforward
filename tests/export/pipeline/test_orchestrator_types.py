# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import pathlib

import torch

from fastforward.export._export_schemas import V1SchemaHandler
from fastforward.export.pipeline.orchestrator import ExportRequest, QnnOnnxOptions


def test_export_request_converts_output_dir_to_path() -> None:
    request = ExportRequest(
        model=torch.nn.Identity(),
        sample_inputs=[((torch.randn(1, 4),), {})],
        output_dir="artifacts",
        model_name="model",
        target="qnn",
        format="onnx",
    )

    assert isinstance(request.output_dir, pathlib.Path)
    assert request.output_dir == pathlib.Path("artifacts")


def test_export_request_fields_are_pipeline_agnostic() -> None:
    request = ExportRequest(
        model=torch.nn.Identity(),
        sample_inputs=[((torch.randn(1, 4),), {})],
        output_dir=pathlib.Path("artifacts"),
        model_name="model",
        target="qnn",
        format="onnx",
    )

    assert request.target == "qnn"
    assert request.format == "onnx"
    assert request.pipeline_factory is None
    assert request.options == {}


def test_qnn_onnx_options_to_context() -> None:
    options = QnnOnnxOptions(
        input_names=["in0"],
        output_names=["out0"],
        alter_node_names=True,
        alter_node_names_prefix="ff_test",
        onnx_export_options={"opset_version": 17},
        onnx_save_kwargs={"save_as_external_data": False},
        verbose=True,
    )

    context = options.to_context()

    assert context["input_names"] == ["in0"]
    assert context["output_names"] == ["out0"]
    assert isinstance(context["encoding_schema_handler"], V1SchemaHandler)
    assert context["alter_node_names"] is True
    assert context["alter_node_names_prefix"] == "ff_test"
    assert context["onnx_export_options"] == {"opset_version": 17}
    assert context["onnx_save_kwargs"] == {"save_as_external_data": False}
    assert context["verbose"] is True


def test_qnn_onnx_options_to_context_copies_mutable_dictionaries() -> None:
    options = QnnOnnxOptions(
        onnx_export_options={"optimize": False},
        onnx_save_kwargs={"all_tensors_to_one_file": True},
    )

    context = options.to_context()
    context["onnx_export_options"]["optimize"] = True
    context["onnx_save_kwargs"]["all_tensors_to_one_file"] = False

    assert options.onnx_export_options["optimize"] is False
    assert options.onnx_save_kwargs["all_tensors_to_one_file"] is True
