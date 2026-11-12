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
:func:`text_config_view` merges the two into one source config.
"""

from typing import Any, Protocol, cast, runtime_checkable

from typing_extensions import override


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
