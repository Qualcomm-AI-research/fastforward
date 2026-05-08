# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Any, Callable, TypeAlias

from fastforward.export.pipeline.core import Pipeline
from fastforward.export.pipeline.qnn_onnx_pipeline import qnn_onnx_pipeline

_PipelineFactoryT: TypeAlias = Callable[[dict[str, Any]], Pipeline]


class PipelineRegistry:
    """Registry for export pipeline factories keyed by `(target, format)`."""

    def __init__(self) -> None:
        self._factories: dict[tuple[str, str], _PipelineFactoryT] = {}

    def register(
        self,
        target: str,
        format: str,
        factory: _PipelineFactoryT,
        *,
        replace: bool = False,
    ) -> None:
        """Register a pipeline factory for a target/format pair.

        Args:
            target: Export target identifier (for example, ``"qnn"``).
            format: Export format identifier (for example, ``"onnx"``).
            factory: Pipeline factory to register.
            replace: Whether to replace an existing registration for the same key.
        """
        key = self._key(target, format)
        if not replace and key in self._factories:
            msg = f"Pipeline for target='{key[0]}' and format='{key[1]}' is already registered"
            raise ValueError(msg)
        self._factories[key] = factory

    def get(self, target: str, format: str) -> _PipelineFactoryT:
        """Get the registered pipeline factory for a target/format pair."""
        key = self._key(target, format)
        try:
            return self._factories[key]
        except KeyError as exc:
            msg = f"No pipeline registered for target='{key[0]}' and format='{key[1]}'"
            raise KeyError(msg) from exc

    def has(self, target: str, format: str) -> bool:
        """Return whether a pipeline factory exists for a target/format pair."""
        return self._key(target, format) in self._factories

    @staticmethod
    def _key(target: str, format: str) -> tuple[str, str]:
        return target.strip().lower(), format.strip().lower()


def build_default_registry() -> PipelineRegistry:
    """Build the default registry with built-in export pipelines."""
    registry = PipelineRegistry()
    registry.register("qnn", "onnx", qnn_onnx_pipeline)
    return registry
