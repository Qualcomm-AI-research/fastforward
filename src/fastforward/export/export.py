# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""
!!! experimental
    Export is an experimental feature and is currently under active development.
    Please expect API changes. We encourage you to file bug reports if you run into any problems.

Functionality to export a quantized module for running inference on device.

Supported backends:
 - QNN

"""  # noqa: D205, D212

import pathlib
import warnings

from typing import Any

import torch

import fastforward as ff

from fastforward.export._export_schemas import EncodingSchemaHandler, V1SchemaHandler
from fastforward.export.pipeline import (
    ExportArtifacts,
    ExportOrchestrator,
    ExportRequest,
    PipelineRegistry,
    QnnOnnxOptions,
    build_default_registry,
)
from fastforward.flags import export_mode


@ff.flags.context(ff.strict_quantization, False)
def export(
    model: torch.nn.Module,
    data: tuple[torch.Tensor, ...],
    output_directory: str | pathlib.Path,
    model_name: str,
    target: str = "qnn",
    format: str = "onnx",
    model_kwargs: None | dict[str, Any] = None,
    input_names: None | list[str] = None,
    output_names: None | list[str] = None,
    verbose: bool | None = None,
    encoding_schema_handler: EncodingSchemaHandler | None = None,
    alter_node_names: bool = False,
    onnx_export_options: dict[str, Any] | None = None,
    onnx_save_kwargs: dict[str, Any] | None = None,
    pipeline_factory: Any | None = None,
    orchestrator: ExportOrchestrator | None = None,
    registry: PipelineRegistry | None = None,
) -> ExportArtifacts:
    """Export a model by constructing and executing a pipeline export request."""
    if orchestrator is not None and registry is not None:
        raise ValueError("Pass either `orchestrator` or `registry`, not both.")

    qnn_options = QnnOnnxOptions(
        input_names=input_names,
        output_names=output_names,
        encoding_schema_handler=encoding_schema_handler or V1SchemaHandler(),
        alter_node_names=alter_node_names,
        onnx_export_options=onnx_export_options or {},
        onnx_save_kwargs=onnx_save_kwargs or {},
        verbose=verbose,
    )
    request = ExportRequest(
        model=model,
        sample_inputs=[(data, model_kwargs or {})],
        output_dir=output_directory,
        model_name=model_name,
        target=target,
        format=format,
        pipeline_factory=pipeline_factory,
        options=qnn_options.to_context(),
    )

    with export_mode(True), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="onnxscript.*")
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch._dynamo.*")
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=r".*torch\.autograd\.function\.Function.*should not be instantiated.*",
        )
        active_registry = registry or build_default_registry()
        active_orchestrator = orchestrator or ExportOrchestrator(registry=active_registry)
        return active_orchestrator.export(request)
