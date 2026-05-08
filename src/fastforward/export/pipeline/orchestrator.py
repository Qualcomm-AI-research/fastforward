# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import pathlib

from dataclasses import dataclass, field
from typing import Any, Callable, TypeAlias

import torch

from fastforward.export._export_schemas import EncodingSchemaHandler, V1SchemaHandler
from fastforward.export.pipeline.core import Pipeline, StageReference
from fastforward.export.pipeline.registry import PipelineRegistry, build_default_registry

_SampleInputsT: TypeAlias = list[tuple[tuple[Any, ...], dict[str, Any]]]
_PipelineFactoryT: TypeAlias = Callable[[dict[str, Any]], Pipeline]
_EvalResultsT: TypeAlias = dict[tuple[StageReference, StageReference], torch.Tensor]


@dataclass(slots=True)
class ExportRequest:
    """Generic request for pipeline-based export.

    This request is pipeline-agnostic by design:
    target-specific options should be carried in `options`.
    """

    model: torch.nn.Module
    sample_inputs: _SampleInputsT
    output_dir: pathlib.Path | str
    model_name: str
    target: str
    format: str
    pipeline_factory: _PipelineFactoryT | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.output_dir = pathlib.Path(self.output_dir)


@dataclass(slots=True)
class QnnOnnxOptions:
    """Options for the QNN->ONNX pipeline.

    This typed container is optional and can be converted to stage context
    with `to_context()`.
    """

    input_names: list[str] | None = None
    output_names: list[str] | None = None
    encoding_schema_handler: EncodingSchemaHandler = field(default_factory=V1SchemaHandler)
    alter_node_names: bool = False
    alter_node_names_prefix: str = "ff"
    onnx_export_options: dict[str, Any] = field(default_factory=dict)
    onnx_save_kwargs: dict[str, Any] = field(default_factory=dict)
    verbose: bool | None = None

    def to_context(self) -> dict[str, Any]:
        """Convert options to pipeline context values."""
        return {
            "input_names": self.input_names,
            "output_names": self.output_names,
            "encoding_schema_handler": self.encoding_schema_handler,
            "alter_node_names": self.alter_node_names,
            "alter_node_names_prefix": self.alter_node_names_prefix,
            "onnx_export_options": dict(self.onnx_export_options),
            "onnx_save_kwargs": dict(self.onnx_save_kwargs),
            "verbose": self.verbose,
        }


@dataclass(slots=True)
class ExportArtifacts:
    """Artifacts and metadata produced by a pipeline export run."""

    pipeline_name: str
    stage_outputs: dict[str, Any] = field(default_factory=dict)
    eval_results: _EvalResultsT = field(default_factory=dict)


class ExportOrchestrator:
    """Resolve and execute an export pipeline from an `ExportRequest`."""

    def __init__(
        self,
        registry: PipelineRegistry | None = None,
    ) -> None:
        self._registry = registry or build_default_registry()

    def export(self, request: ExportRequest) -> ExportArtifacts:
        """Run export for the given request and return produced artifact metadata."""
        output_dir = self._build_output_dir(request)
        pipeline_factory = self._resolve_pipeline_factory(request)
        pipeline_context = self._build_pipeline_context(request, output_dir)

        pipeline = pipeline_factory(pipeline_context)
        stage_outputs, eval_results = pipeline(request.model, request.sample_inputs)

        return ExportArtifacts(
            pipeline_name=getattr(pipeline_factory, "__name__", type(pipeline_factory).__name__),
            stage_outputs=stage_outputs,
            eval_results=eval_results,
        )

    def _resolve_pipeline_factory(self, request: ExportRequest) -> _PipelineFactoryT:
        if request.pipeline_factory is not None:
            return request.pipeline_factory

        return self._registry.get(request.target, request.format)

    def _build_output_dir(self, request: ExportRequest) -> pathlib.Path:
        raw_output_dir = request.output_dir
        if isinstance(raw_output_dir, pathlib.Path):
            output_dir = raw_output_dir
        else:
            output_dir = pathlib.Path(raw_output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        return output_dir

    def _build_pipeline_context(
        self,
        request: ExportRequest,
        output_dir: pathlib.Path,
    ) -> dict[str, Any]:
        context: dict[str, Any] = {
            "output_dir": output_dir,
            "model_name": request.model_name,
        }

        context.update(dict(request.options))
        return context
