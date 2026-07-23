# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Structural type of the "source config" the GGUF stages read from.

Any HuggingFace ``PretrainedConfig`` (or a ``SimpleNamespace``) that carries
these attributes satisfies this Protocol structurally — no inheritance
required. Fields that are optional across source architectures
(``num_key_value_heads``, ``rope_theta``, ``head_dim``, ``rope_scaling``,
``tie_word_embeddings``) are deliberately excluded: every call site reads them
via ``getattr(..., default)`` with a documented fallback, so a config that
omits them still exports correctly.
"""

from typing import Protocol, runtime_checkable


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
