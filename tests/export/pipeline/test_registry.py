# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Any

import pytest

from fastforward.export.pipeline import Pipeline
from fastforward.export.pipeline.registry import PipelineRegistry


def _dummy_factory(_context: dict[str, Any]) -> Pipeline:
    return Pipeline()


def test_pipeline_registry_register_and_get() -> None:
    registry = PipelineRegistry()
    registry.register("qnn", "onnx", _dummy_factory)

    assert registry.has("qnn", "onnx")
    assert registry.get("qnn", "onnx") is _dummy_factory


def test_pipeline_registry_normalizes_key_case_and_whitespace() -> None:
    registry = PipelineRegistry()
    registry.register(" QNN ", " ONNX ", _dummy_factory)

    assert registry.has("qnn", "onnx")
    assert registry.has("QNN", "ONNX")


def test_pipeline_registry_register_raises_on_duplicate_without_replace() -> None:
    registry = PipelineRegistry()
    registry.register("qnn", "onnx", _dummy_factory)

    with pytest.raises(ValueError, match="already registered"):
        registry.register("qnn", "onnx", _dummy_factory)


def test_pipeline_registry_register_replace_overwrites_existing_factory() -> None:
    registry = PipelineRegistry()

    def replacement_factory(_context: dict[str, Any]) -> Pipeline:
        return Pipeline()

    registry.register("qnn", "onnx", _dummy_factory)
    registry.register("qnn", "onnx", replacement_factory, replace=True)

    assert registry.get("qnn", "onnx") is replacement_factory


def test_pipeline_registry_get_raises_when_missing() -> None:
    registry = PipelineRegistry()
    with pytest.raises(KeyError, match="No pipeline registered"):
        registry.get("qnn", "onnx")
