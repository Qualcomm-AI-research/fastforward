# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import pathlib

from dataclasses import dataclass, field
from typing import Any, Callable, TypeAlias

import torch

from fastforward.export._export_schemas import EncodingSchemaHandler, V1SchemaHandler

_SampleInputsT: TypeAlias = list[tuple[tuple[Any, ...], dict[str, Any]]]
_PipelineFactoryT: TypeAlias = Callable[[dict[str, Any]], Any]


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

