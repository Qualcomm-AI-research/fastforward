# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Built-in :class:`ArchAdapter` instances for multimodal projector (``mmproj``) files.

llama.cpp runs the encoders of a vision-language model from a second GGUF file
next to the text model: architecture ``clip``, ``general.type=mmproj``, loaded
by ``tools/mtmd/clip.cpp``. It carries the vision tower and the projector that
maps its features into the text decoder's embedding space, and no vocabulary.

The mmproj adapters take an
:class:`~fastforward.export.stages.gguf.MmprojSourceConfig` as ``model_config``
and mirror llama.cpp's ``MmprojModel`` converters. As for a text decoder,
weights with an initialized FastForward quantizer are written quantized and
all other tensors in float, with the float types ``clip.cpp`` requires pinned.
"""

from __future__ import annotations

from typing import Any, cast

from gguf import GGUFType, GGUFWriter

from fastforward.exceptions import ExportError
from fastforward.export.stages.gguf._config import GgufSourceConfig, MmprojSourceConfig
from fastforward.export.stages.gguf.rules import Drop, Split

_CLIP_ARCH = "clip"

# Everything outside the vision tower (``model.visual.`` in transformers 5,
# ``visual.`` before): the text decoder and the LM head.
_NON_VISION_PATTERN = r"(?!(?:model\.)?visual\.).*"


def _as_mmproj_config(config: Any) -> MmprojSourceConfig:
    """Accept only an :class:`MmprojSourceConfig` as an mmproj adapter's ``model_config``."""
    if not isinstance(config, MmprojSourceConfig):
        msg = (
            "mmproj GGUF export needs an MmprojSourceConfig as model_config, e.g. "
            + "MmprojSourceConfig.from_hf(model.config, processor), "
            + f"got {type(config).__name__}"
        )
        raise ExportError(msg)
    return config


def _require_mmproj_config(config: Any) -> GgufSourceConfig:
    # The hooks of the mmproj adapters read it back via `_as_mmproj_config`.
    return cast(GgufSourceConfig, _as_mmproj_config(config))


def _find_param(params: Any, *names: str) -> Any:
    """Return the first of ``names`` set on ``params``, like llama.cpp's ``find_vparam``."""
    for name in names:
        value = getattr(params, name, None)
        if value is not None:
            return value
    msg = f"mmproj export requires one of {list(names)} on the vision config"
    raise ExportError(msg)


def _preprocessor_param(config: MmprojSourceConfig, name: str) -> Any:
    value = config.preprocessor_config.get(name)
    if value is None:
        msg = f"mmproj export requires '{name}' in the processor's preprocessing parameters"
        raise ExportError(msg)
    return value


def _write_clip_vision_metadata(
    writer: GGUFWriter, config: MmprojSourceConfig, *, default_image_size: int
) -> None:
    """Write the vision-encoder metadata every ``clip`` mmproj file carries.

    Mirrors ``MmprojModel.set_gguf_parameters`` in llama.cpp's converter. The
    image size is the vision config's ``image_size`` if set, and
    ``default_image_size`` otherwise.
    """
    vision_config = config.vision_config
    if vision_config is None:
        msg = "mmproj vision export requires a 'vision_config' on the model config"
        raise ExportError(msg)
    writer.add_type(GGUFType.MMPROJ)
    writer.add_clip_has_vision_encoder(True)
    writer.add_vision_projection_dim(config.text_config.hidden_size)
    writer.add_vision_image_size(getattr(vision_config, "image_size", None) or default_image_size)
    writer.add_vision_patch_size(_find_param(vision_config, "patch_size"))
    writer.add_vision_embedding_length(_find_param(vision_config, "hidden_size"))
    writer.add_vision_feed_forward_length(_find_param(vision_config, "intermediate_size"))
    writer.add_vision_block_count(_find_param(vision_config, "num_hidden_layers", "depth"))
    writer.add_vision_head_count(_find_param(vision_config, "num_attention_heads", "num_heads"))
    writer.add_vision_image_mean([float(v) for v in _preprocessor_param(config, "image_mean")])
    writer.add_vision_image_std([float(v) for v in _preprocessor_param(config, "image_std")])


# ``clip.cpp`` loads the Conv3D patch embedding as two Conv2D kernels, one per frame.
_QWEN_VL_TEMPORAL_PATCH_SIZE = 2

# transformers hardcodes the eps of every Qwen2.5-VL and Qwen3-VL vision norm.
# llama.cpp's converter writes the text decoder's ``rms_norm_eps`` instead,
# which is the same value in the released checkpoints.
_QWEN_VL_VISION_NORM_EPS = 1e-6


def _qwen_vl_vision_config(config: GgufSourceConfig, model_type: str, model: str) -> Any:
    """Return the vision config of a Qwen-VL ``model_type`` checkpoint llama.cpp supports.

    Raises:
        ExportError: If the config is not of ``model_type``, or its temporal
            patch size is not 2.
    """
    mmproj_config = _as_mmproj_config(config)
    actual_model_type = getattr(mmproj_config.model_config, "model_type", None)
    if actual_model_type != model_type:
        msg = f"{model} mmproj export requires model_type {model_type!r}, got {actual_model_type!r}"
        raise ExportError(msg)
    vision_config = mmproj_config.vision_config
    temporal_patch_size = getattr(vision_config, "temporal_patch_size", None)
    if temporal_patch_size != _QWEN_VL_TEMPORAL_PATCH_SIZE:
        msg = (
            f"llama.cpp only supports {model} with temporal_patch_size "
            + f"{_QWEN_VL_TEMPORAL_PATCH_SIZE}, got {temporal_patch_size!r}"
        )
        raise ExportError(msg)
    return vision_config


def _pin_clip_float_type(gguf_name: str, ndim: int, float_type: str) -> str | None:
    """Pin the float types ``clip.cpp`` requires, as llama.cpp's converter does.

    1-D tensors (norms, biases) and position embeddings are F32; patch
    embeddings are F16 in F16 files and F32 otherwise.
    """
    if ndim <= 1 or gguf_name.endswith("_norm.weight") or ".position_embd." in gguf_name:
        return "F32"
    if ".patch_embd.weight" in gguf_name:
        return "F16" if float_type == "F16" else "F32"
    return None


_DROP_NON_VISION = Drop(_NON_VISION_PATTERN)

_SPLIT_QWEN_VL_PATCH_EMBED = Split(
    pattern=r"((?:model\.)?visual)\.patch_embed\.proj\.weight",
    targets=(r"\1.patch_embed.proj.weight.0", r"\1.patch_embed.proj.weight.1"),
    axis=2,
    squeeze=True,
)
