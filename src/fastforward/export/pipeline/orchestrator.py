# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import pathlib

from dataclasses import dataclass, field
from typing import Any, Callable, TypeAlias

import torch

from fastforward.export._export_schemas import EncodingSchemaHandler, V1SchemaHandler
from fastforward.export.pipeline.core import Pipeline, StageReference
from fastforward.export.pipeline.registry import PipelineRegistry, build_default_registry
from fastforward.export.stages.gguf.adapter import ArchAdapter, GgufQuantFormat

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
    store_weights_as_qdq: bool = True

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
            "store_weights_as_qdq": self.store_weights_as_qdq,
        }


@dataclass(slots=True)
class GgufLlamaCppOptions:
    """Options for the GGUF -> llama.cpp export pipeline.

    This typed container is optional and can be converted to stage context with
    `to_context()`. The pipeline expects an already-quantized model.

    Attributes:
        arch_adapter: :class:`ArchAdapter` carrying all HF-to-GGUF assumptions
            for the target architecture: parameter-name map, RoPE permute,
            metadata writer, and tokenizer discriminators. Built-in adapters
            ``LLAMA_ADAPTER``, ``QWEN2_ADAPTER``, and ``QWEN3_ADAPTER`` are
            importable from :mod:`fastforward.export.stages.gguf`. For a model
            whose module tree differs from the standard HuggingFace layout,
            construct your own ``ArchAdapter``.
        quant_format: :class:`GgufQuantFormat` selecting the target block type,
            block size, and packing function. Built-in formats ``GGUF_Q4_0``
            and ``GGUF_Q8_0`` are importable from
            :mod:`fastforward.export.stages.gguf`.
        model_config: Source model config satisfying
            :class:`~fastforward.export.stages.gguf.GgufSourceConfig`. Required
            attributes: ``hidden_size``, ``num_attention_heads``,
            ``num_hidden_layers``, ``intermediate_size``,
            ``max_position_embeddings``, ``rms_norm_eps``, ``vocab_size``.
            Optional (accessed via ``getattr`` with defaults):
            ``num_key_value_heads``, ``rope_theta``, ``head_dim``,
            ``rope_scaling``, ``tie_word_embeddings``. A HuggingFace
            ``PretrainedConfig`` or a ``SimpleNamespace`` carrying these fields
            both work.
        tokenizer: Optional HuggingFace tokenizer; when provided, its vocabulary
            is written into the GGUF.
    """

    arch_adapter: ArchAdapter
    quant_format: GgufQuantFormat | None = None
    model_config: Any = None
    tokenizer: Any = None

    def to_context(self) -> dict[str, Any]:
        """Convert options to pipeline context values."""
        context: dict[str, Any] = {
            "arch_adapter": self.arch_adapter,
            "model_config": self.model_config,
            "tokenizer": self.tokenizer,
        }
        if self.quant_format is not None:
            context["quant_format"] = self.quant_format
        return context


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


def export_with_pipeline(
    request: ExportRequest,
    orchestrator: ExportOrchestrator | None = None,
) -> ExportArtifacts:
    """Export using the pipeline-based orchestrator API."""
    active_orchestrator = orchestrator or ExportOrchestrator()
    return active_orchestrator.export(request)
