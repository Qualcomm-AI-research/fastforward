# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import pathlib

from typing import Any

import pytest
import torch

from fastforward.export.pipeline import Pipeline
from fastforward.export.pipeline.orchestrator import (
    ExportOrchestrator,
    ExportRequest,
    QnnOnnxOptions,
)
from fastforward.export.pipeline.registry import PipelineRegistry


def _artifact_writer_factory(context: dict[str, Any]) -> Pipeline:
    pipeline = Pipeline(context)

    def write_artifacts(
        modules: tuple[torch.nn.Module, ...],
        sample_inputs: list[tuple[tuple[Any, ...], dict[str, Any]]],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        del modules, sample_inputs
        output_dir = pathlib.Path(context["output_dir"])
        model_name = context["model_name"]
        onnx_path = output_dir / f"{model_name}.onnx"
        encodings_path = output_dir / f"{model_name}.encodings"
        onnx_path.write_text("onnx")
        encodings_path.write_text("{}")
        return {"saved": True}

    pipeline.register_stage(write_artifacts, "write_artifacts", capture_stage_output=True)
    return pipeline


def _override_writer_factory(context: dict[str, Any]) -> Pipeline:
    pipeline = Pipeline(context)

    def write_override(
        modules: tuple[torch.nn.Module, ...],
        sample_inputs: list[tuple[tuple[Any, ...], dict[str, Any]]],
        context: dict[str, Any],
    ) -> pathlib.Path:
        del modules, sample_inputs
        output_dir = pathlib.Path(context["output_dir"])
        model_name = context["model_name"]
        encodings_path = output_dir / f"{model_name}.encodings"
        encodings_path.write_text("override")
        return encodings_path

    pipeline.register_stage(write_override, "write_override", capture_stage_output=True)
    return pipeline


def _request(tmp_path: pathlib.Path, *, pipeline_factory: Any = None) -> ExportRequest:
    qnn_options = QnnOnnxOptions()
    return ExportRequest(
        model=torch.nn.Identity(),
        sample_inputs=[((torch.randn(1, 4),), {})],
        output_dir=tmp_path,
        model_name="model",
        target="qnn",
        format="onnx",
        pipeline_factory=pipeline_factory,
        options=qnn_options.to_context(),
    )


def test_orchestrator_uses_registry_pipeline_and_returns_stage_outputs(
    tmp_path: pathlib.Path,
) -> None:
    registry = PipelineRegistry()
    registry.register("qnn", "onnx", _artifact_writer_factory)
    orchestrator = ExportOrchestrator(registry=registry)

    result = orchestrator.export(_request(tmp_path))

    assert (tmp_path / "model.onnx").is_file()
    assert (tmp_path / "model.encodings").is_file()
    assert result.stage_outputs["write_artifacts"]["saved"] is True
    assert result.eval_results == {}


def test_orchestrator_prefers_request_pipeline_factory_override(tmp_path: pathlib.Path) -> None:
    registry = PipelineRegistry()
    registry.register("qnn", "onnx", _artifact_writer_factory)
    orchestrator = ExportOrchestrator(registry=registry)

    result = orchestrator.export(_request(tmp_path, pipeline_factory=_override_writer_factory))

    assert result.stage_outputs["write_override"] == tmp_path / "model.encodings"
    assert (tmp_path / "model.encodings").read_text() == "override"


def test_orchestrator_raises_on_missing_registry_entry(tmp_path: pathlib.Path) -> None:
    orchestrator = ExportOrchestrator(registry=PipelineRegistry())

    with pytest.raises(KeyError, match="No pipeline registered"):
        orchestrator.export(_request(tmp_path))


def test_orchestrator_merges_request_options_into_context() -> None:
    orchestrator = ExportOrchestrator(registry=PipelineRegistry())
    qnn_options = QnnOnnxOptions(
        input_names=["input_0"],
        output_names=["output_0"],
        alter_node_names=True,
        onnx_export_options={"opset_version": 17},
    )
    request = ExportRequest(
        model=torch.nn.Identity(),
        sample_inputs=[((torch.randn(1, 4),), {})],
        output_dir=pathlib.Path("artifacts"),
        model_name="model",
        target="qnn",
        format="onnx",
        options=qnn_options.to_context(),
    )
    output_dir = orchestrator._build_output_dir(request)
    context = orchestrator._build_pipeline_context(request, output_dir)

    assert context["input_names"] == ["input_0"]
    assert context["output_names"] == ["output_0"]
    assert context["alter_node_names"] is True
    assert context["onnx_export_options"] == {"opset_version": 17}
    assert context["output_dir"] == pathlib.Path("artifacts")
    assert context["model_name"] == "model"


def test_orchestrator_pipeline_factory_override_is_used() -> None:
    orchestrator = ExportOrchestrator(registry=PipelineRegistry())
    request = _request(pathlib.Path("."), pipeline_factory=_override_writer_factory)
    assert orchestrator._resolve_pipeline_factory(request) is _override_writer_factory
