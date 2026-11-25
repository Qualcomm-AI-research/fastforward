# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Structural type of the "source config" the GGUF stages read from.

Any HuggingFace ``PretrainedConfig`` (or a ``SimpleNamespace``) that carries
these attributes satisfies this Protocol structurally — no inheritance
required. Fields that are optional across source architectures
(``num_key_value_heads``, ``head_dim``, ``rope_scaling``,
``tie_word_embeddings``) are deliberately excluded: every call site reads them
via ``getattr(..., default)`` with a documented fallback, so a config that
omits them still exports correctly.

The RoPE base frequency is required but also excluded, because its location
depends on the transformers version: ``rope_parameters["rope_theta"]`` from
transformers 5 on, ``rope_theta`` before. The built-in metadata writers accept
either and raise :class:`~fastforward.exceptions.ExportError` if both are
missing.

Multimodal checkpoints nest the decoder's fields under ``text_config`` and keep
others (``tie_word_embeddings``, ``vision_config``) on the top-level config;
:func:`text_config_view` merges the two into one source config. A multimodal
projector (``mmproj``) export also needs the image preprocessing parameters,
which :class:`MmprojSourceConfig` bundles with the model config.
"""

import types

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

from typing_extensions import override

from fastforward.exceptions import ExportError


@runtime_checkable
class GgufSourceConfig(Protocol):
    """Minimal structural surface a source model config must expose."""

    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    intermediate_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    vocab_size: int


class _TextConfigView:
    """Read attributes from ``text_config`` first, then from the top-level config."""

    __slots__ = ("_config", "_text_config")

    def __init__(self, config: Any, text_config: Any) -> None:
        self._config = config
        self._text_config = text_config

    def __getattr__(self, name: str) -> Any:
        if name in self.__slots__:
            # Unset slot, e.g. while copying: don't recurse into ourselves.
            raise AttributeError(name)
        value = getattr(self._text_config, name, None)
        if value is not None:
            return value
        return getattr(self._config, name)

    @override
    def __repr__(self) -> str:
        return f"text_config_view({self._config!r})"


def text_config_view(config: Any) -> GgufSourceConfig:
    """Merge a multimodal config's ``text_config`` with its top-level fields.

    Attributes are read from ``config.text_config`` and fall back to ``config``
    when missing or ``None`` there, as llama.cpp's converter merges
    ``{**config, **config["text_config"]}``. For example, transformers 5 keeps
    ``tie_word_embeddings`` only on the top-level ``Qwen2_5_VLConfig``. A
    config without a ``text_config`` is returned unchanged.

    Use it as ``ArchAdapter.resolve_config`` for adapters that take the
    top-level config of a multimodal checkpoint.
    """
    text_config = getattr(config, "text_config", None)
    if text_config is None:
        return cast(GgufSourceConfig, config)
    return cast(GgufSourceConfig, _TextConfigView(config, text_config))


def _preprocessor_dict(processor: Any) -> dict[str, Any]:
    """Return a processor's preprocessing parameters as a flat dict.

    Accepts a mapping (e.g. a parsed ``preprocessor_config.json``), an image
    processor, or a multimodal processor carrying an ``image_processor``. A
    nested ``image_processor`` mapping is lifted to the top level, as
    llama.cpp's converter does for ``processor_config.json``.
    """
    if isinstance(processor, Mapping):
        params = dict(processor)
    elif (image_processor := getattr(processor, "image_processor", None)) is not None:
        params = _preprocessor_dict(image_processor)
    elif callable(to_dict := getattr(processor, "to_dict", None)):
        params = dict(to_dict())
    else:
        msg = (
            "Expected a processor, an image processor, or a mapping of preprocessing "
            + f"parameters, got {type(processor).__name__}"
        )
        raise ExportError(msg)
    nested = params.get("image_processor")
    if isinstance(nested, Mapping):
        params = {**params, **nested}
    return params


@dataclass(frozen=True, slots=True)
class MmprojSourceConfig:
    """Source config of a multimodal projector (``mmproj``) export.

    llama.cpp loads the vision (and audio) encoders of a multimodal model from
    a separate ``mmproj`` GGUF file, whose header needs fields from both the
    model config (encoder dims, the text hidden size the projector maps to)
    and the processor (image normalization). Build it with :meth:`from_hf` and
    pass it as ``model_config`` together with an mmproj
    :class:`~fastforward.export.stages.gguf.ArchAdapter`.

    Attributes:
        model_config: The top-level config of the multimodal checkpoint (e.g.
            ``model.config``), with ``text_config`` and ``vision_config``
            and/or ``audio_config``.
        preprocessor_config: The processor's preprocessing parameters (e.g.
            ``image_mean`` and ``image_std``), read-only.
    """

    model_config: Any
    preprocessor_config: Mapping[str, Any]

    @classmethod
    def from_hf(cls, model_config: Any, processor: Any) -> "MmprojSourceConfig":
        """Build the source config from a HuggingFace config and processor.

        Args:
            model_config: The top-level config of the multimodal checkpoint,
                e.g. ``model.config``.
            processor: The checkpoint's processor (``AutoProcessor``), its
                image processor, or a mapping of its preprocessing parameters,
                e.g. the parsed ``preprocessor_config.json``.

        Raises:
            ExportError: If the config has neither a ``vision_config`` nor an
                ``audio_config``, or ``processor`` is none of the above.
        """
        if (
            getattr(model_config, "vision_config", None) is None
            and getattr(model_config, "audio_config", None) is None
        ):
            msg = (
                "mmproj export needs the top-level model config (with 'vision_config' "
                + "or 'audio_config'), e.g. model.config"
            )
            raise ExportError(msg)
        return cls(model_config, types.MappingProxyType(_preprocessor_dict(processor)))

    @property
    def text_config(self) -> GgufSourceConfig:
        """The text decoder's config, as :func:`text_config_view` merges it."""
        return text_config_view(self.model_config)

    @property
    def vision_config(self) -> Any:
        """The vision encoder's config, or ``None`` if the model has none."""
        return getattr(self.model_config, "vision_config", None)

    @property
    def audio_config(self) -> Any:
        """The audio encoder's config, or ``None`` if the model has none."""
        return getattr(self.model_config, "audio_config", None)
