# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Tests for the multimodal projector (``mmproj``) adapters and their source config."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from fastforward.exceptions import ExportError
from fastforward.export.stages.gguf import (
    ArchAdapter,
    ExtractedTensor,
    MmprojSourceConfig,
)
from fastforward.export.stages.gguf._mmproj import (
    _pin_clip_float_type,
)
from gguf import GGUFWriter

_PREPROCESSOR = {"image_mean": [0.5, 0.25, 0.125], "image_std": [1, 2, 3]}


def _qwen25vl_config(**vision_overrides: Any) -> SimpleNamespace:
    """A top-level Qwen2.5-VL-shaped config with a tiny vision tower."""
    vision_config = SimpleNamespace(
        depth=2,
        hidden_size=32,
        intermediate_size=64,
        num_heads=2,
        patch_size=14,
        temporal_patch_size=2,
        fullatt_block_indexes=[1],
    )
    for key, value in vision_overrides.items():
        setattr(vision_config, key, value)
    return SimpleNamespace(
        model_type="qwen2_5_vl",
        text_config=SimpleNamespace(hidden_size=64),
        vision_config=vision_config,
    )


def _write_metadata(config: Any, adapter: ArchAdapter) -> dict[str, Any]:
    writer = GGUFWriter(None, arch=adapter.gguf_arch)
    adapter.write_metadata(writer, adapter.resolve_config(config))
    return {key: value.value for key, value in writer.kv_data[0].items()}


@pytest.mark.parametrize(
    ("gguf_name", "ndim", "float_type", "expected"),
    [
        ("v.blk.0.attn_q.bias", 1, "F16", "F32"),
        ("v.blk.0.ln1.weight", 1, "BF16", "F32"),
        ("v.blk.0.attn_q_norm.weight", 2, "F16", "F32"),
        ("v.position_embd.weight", 2, "F16", "F32"),
        ("v.patch_embd.weight", 4, "F16", "F16"),
        ("v.patch_embd.weight.1", 4, "F16", "F16"),
        ("v.patch_embd.weight", 4, "BF16", "F32"),
        ("v.patch_embd.weight", 4, "F32", "F32"),
        ("v.blk.0.attn_q.weight", 2, "F16", None),
        ("mm.0.weight", 2, "BF16", None),
    ],
)
def test_pin_clip_float_type_matches_llama_cpp_converter(
    gguf_name: str, ndim: int, float_type: str, expected: str | None
) -> None:
    # GIVEN: a tensor of an mmproj file and the export's default float type.
    # WHEN: asking which type clip.cpp requires for it.
    pinned = _pin_clip_float_type(gguf_name, ndim, float_type)

    # THEN: 1-D tensors, norms, and position embeddings are F32, patch
    # embeddings are F16 only in F16 files, and other tensors are not pinned.
    assert pinned == expected


class _ImageProcessor:
    def to_dict(self) -> dict[str, Any]:
        return dict(_PREPROCESSOR)


@pytest.mark.parametrize(
    "processor",
    [
        _PREPROCESSOR,
        _ImageProcessor(),
        SimpleNamespace(image_processor=_ImageProcessor()),
        {"processor_class": "Qwen2_5_VLProcessor", "image_processor": _PREPROCESSOR},
    ],
    ids=["mapping", "image-processor", "processor", "nested-mapping"],
)
def test_mmproj_source_config_reads_preprocessing_parameters(processor: Any) -> None:
    # GIVEN: a multimodal config and a processor in any of the accepted forms.
    model_config = _qwen25vl_config()

    # WHEN: building the source config.
    config = MmprojSourceConfig.from_hf(model_config, processor)

    # THEN: the preprocessing parameters are flat, read-only, and the encoder
    # and text configs are reachable.
    assert config.preprocessor_config["image_mean"] == _PREPROCESSOR["image_mean"]
    assert config.preprocessor_config["image_std"] == _PREPROCESSOR["image_std"]
    with pytest.raises(TypeError):
        config.preprocessor_config["image_mean"] = []  # type: ignore[index]
    assert config.vision_config is model_config.vision_config
    assert config.audio_config is None
    assert config.text_config.hidden_size == 64


def test_mmproj_source_config_rejects_config_without_encoders() -> None:
    # GIVEN: a text-only config, as when passing model.config.text_config.
    text_config = SimpleNamespace(hidden_size=64)

    # WHEN/THEN: building the source config raises.
    with pytest.raises(ExportError, match="top-level model config"):
        MmprojSourceConfig.from_hf(text_config, _PREPROCESSOR)


def test_mmproj_source_config_rejects_unknown_processor() -> None:
    # GIVEN/WHEN/THEN: a processor that is neither a mapping nor a processor raises.
    with pytest.raises(ExportError, match="got int"):
        MmprojSourceConfig.from_hf(_qwen25vl_config(), 0)


def _float(hf_name: str, *shape: int) -> ExtractedTensor:
    data = torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape)
    return ExtractedTensor(hf_name=hf_name, kind="float", rows=shape[0], float_data=data)
