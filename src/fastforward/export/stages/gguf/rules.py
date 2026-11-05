# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

r"""Tensor rules: the ordered rewrite from extracted model tensors to GGUF tensors.

An :class:`~fastforward.export.stages.gguf.adapter.ArchAdapter` describes every
model-specific tensor rewrite as an ordered sequence of :class:`TensorRule`
objects. The pipeline applies them one after another to the full list of
extracted tensors, so a rule can act on one tensor or several at once:

- :class:`Rename` assigns GGUF names (1 -> 1).
- :class:`Transform` rewrites tensors matching a pattern (1 -> 1).
- :class:`Fuse` concatenates groups of tensors (N -> 1).
- :class:`Drop` removes tensors matching a pattern (1 -> 0).

Every pattern is matched against :attr:`ExtractedTensor.hf_name`; :class:`Rename`
only sets ``gguf_name``, so rules placed after it still see HuggingFace names.
Tensors produced by :class:`Fuse` take the fusion's ``target_name`` as their
``hf_name`` and need a :class:`Rename` placed after the fusion. After the last
rule, tensors without a GGUF name are dropped with a warning.

For rewrites the built-in rules cannot express, subclass :class:`TensorRule`
and implement :meth:`TensorRule.apply` (and :meth:`TensorRule.claims` if the
rule consumes model parameters no other rule claims)::

    tensor_rules = (
        DROP_TIED_LM_HEAD,
        Fuse(
            sources=(
                r"model\.layers\.(\d+)\.self_attn\.q_proj\.weight",
                r"model\.layers\.(\d+)\.self_attn\.k_proj\.weight",
                r"model\.layers\.(\d+)\.self_attn\.v_proj\.weight",
            ),
            target_name=r"model.layers.\1.self_attn.qkv_proj.weight",
        ),
        Rename(my_name_map),
    )
"""

from __future__ import annotations

import abc
import logging
import re

from collections import Counter
from dataclasses import dataclass, replace
from typing import Callable, Sequence, TypeAlias

from typing_extensions import override

from fastforward.exceptions import ExportError
from fastforward.export.stages.gguf._config import GgufSourceConfig
from fastforward.export.stages.gguf._extract import ExtractedTensor
from fastforward.export.stages.gguf._fusion import fuse_matching

logger = logging.getLogger(__name__)

_NameMapT: TypeAlias = Callable[[str], str | None]
_ConditionT: TypeAlias = Callable[[GgufSourceConfig], bool]

TensorTransformT: TypeAlias = Callable[[ExtractedTensor, GgufSourceConfig], ExtractedTensor]
"""A 1 -> 1 tensor rewrite used by :class:`Transform`.

Receives one tensor and the source config and returns the rewritten tensor.
Quantized tensors carry their resolved ``quant_format``.
"""


class TensorRule(abc.ABC):
    """One ordered step of an adapter's tensor rewrite."""

    __slots__ = ()

    @abc.abstractmethod
    def apply(
        self, tensors: list[ExtractedTensor], config: GgufSourceConfig
    ) -> list[ExtractedTensor]:
        """Return the rewritten tensor list; must not mutate ``tensors`` in place."""

    def claims(self, hf_name: str) -> bool:
        """Whether this rule needs the model parameter ``hf_name`` to be extracted.

        Extraction only reads (and quantizes) parameters claimed by at least one
        rule; the others are skipped with a warning. Rules that only rewrite
        tensors claimed by other rules keep the default ``False``.
        """
        del hf_name
        return False

    def removes(self, hf_name: str, config: GgufSourceConfig) -> bool:
        """Whether this rule removes the model parameter ``hf_name`` unchanged.

        Extraction skips (and neither validates nor quantizes) a parameter that a
        rule removes before any rule claims it. Rules that keep, rewrite, or
        consume tensors keep the default ``False``.
        """
        del hf_name, config
        return False


@dataclass(frozen=True, slots=True)
class Rename(TensorRule):
    """Assign GGUF names via ``name_map``.

    Tensors for which ``name_map`` returns ``None`` keep their current
    ``gguf_name`` (empty unless set by an earlier rule), so several
    :class:`Rename` rules can be combined.

    Attributes:
        name_map: Maps a HuggingFace parameter name (e.g.
            ``"model.layers.0.self_attn.q_proj.weight"``) to its GGUF tensor
            name (``"blk.0.attn_q.weight"``), or ``None`` if it has none.
            Parameters it maps are claimed for extraction.
    """

    name_map: _NameMapT

    @override
    def apply(
        self, tensors: list[ExtractedTensor], config: GgufSourceConfig
    ) -> list[ExtractedTensor]:
        del config
        renamed: list[ExtractedTensor] = []
        for tensor in tensors:
            gguf_name = self.name_map(tensor.hf_name)
            renamed.append(tensor if gguf_name is None else replace(tensor, gguf_name=gguf_name))
        return renamed

    @override
    def claims(self, hf_name: str) -> bool:
        return self.name_map(hf_name) is not None


@dataclass(frozen=True, slots=True)
class Transform(TensorRule):
    """Apply ``fn`` to every tensor whose ``hf_name`` fully matches ``pattern``.

    Attributes:
        pattern: Regex matched (``re.fullmatch``) against ``hf_name``.
        fn: The :data:`TensorTransformT` to apply, e.g.
            :func:`~fastforward.export.stages.gguf.llama_rope_permute`.
    """

    pattern: str
    fn: TensorTransformT

    @override
    def apply(
        self, tensors: list[ExtractedTensor], config: GgufSourceConfig
    ) -> list[ExtractedTensor]:
        pattern = re.compile(self.pattern)
        return [
            self.fn(tensor, config) if pattern.fullmatch(tensor.hf_name) else tensor
            for tensor in tensors
        ]


@dataclass(frozen=True, slots=True)
class Fuse(TensorRule):
    r"""Concatenate groups of tensors into one tensor per group.

    Source names are regex patterns with capture groups; tensors whose captured
    groups agree (e.g. the same layer index) form one group and are
    concatenated in the order of ``sources``. ``target_name`` may use
    back-references (``\1``, ``\2``, ...) to the captured groups. The fused
    tensors replace their sources and are appended after the untouched tensors.

    Example — fuse Q/K/V per layer::

        Fuse(
            sources=(
                r"model\.layers\.(\d+)\.self_attn\.q_proj\.weight",
                r"model\.layers\.(\d+)\.self_attn\.k_proj\.weight",
                r"model\.layers\.(\d+)\.self_attn\.v_proj\.weight",
            ),
            target_name=r"model.layers.\1.self_attn.qkv_proj.weight",
        )

    Attributes:
        sources: Regex patterns identifying the tensors to fuse, in concat
            order. Each pattern must have the same capture groups. Literal
            names (no metacharacters) match exactly. Matching parameters are
            claimed for extraction.
        target_name: ``hf_name`` of the fused tensor, with back-references
            substituted from the captured groups.
        axis: Concatenation axis. 0 = row-concat (the common case for QKV
            fusion where each tensor contributes its rows).
    """

    sources: tuple[str, ...]
    target_name: str
    axis: int = 0

    @override
    def apply(
        self, tensors: list[ExtractedTensor], config: GgufSourceConfig
    ) -> list[ExtractedTensor]:
        del config
        return fuse_matching(tensors, self.sources, self.target_name, self.axis)

    @override
    def claims(self, hf_name: str) -> bool:
        return any(re.fullmatch(pattern, hf_name) for pattern in self.sources)


@dataclass(frozen=True, slots=True)
class Drop(TensorRule):
    """Remove tensors whose ``hf_name`` fully matches ``pattern``.

    Attributes:
        pattern: Regex matched (``re.fullmatch``) against ``hf_name``.
        when: Optional predicate on the source config; when given, the rule
            only drops tensors if it returns ``True``.
    """

    pattern: str
    when: _ConditionT | None = None

    @override
    def apply(
        self, tensors: list[ExtractedTensor], config: GgufSourceConfig
    ) -> list[ExtractedTensor]:
        return [tensor for tensor in tensors if not self.removes(tensor.hf_name, config)]

    @override
    def removes(self, hf_name: str, config: GgufSourceConfig) -> bool:
        if self.when is not None and not self.when(config):
            return False
        return re.fullmatch(self.pattern, hf_name) is not None


def _ties_word_embeddings(config: GgufSourceConfig) -> bool:
    return bool(getattr(config, "tie_word_embeddings", False))


DROP_TIED_LM_HEAD = Drop(r"lm_head\.weight", when=_ties_word_embeddings)
"""Drop ``lm_head.weight`` when ``config.tie_word_embeddings`` is true.

llama.cpp reuses ``token_embd.weight`` as the output projection when no
``output.weight`` tensor is present.
"""


def apply_tensor_rules(
    tensors: list[ExtractedTensor],
    rules: Sequence[TensorRule],
    config: GgufSourceConfig,
) -> list[ExtractedTensor]:
    """Apply ``rules`` in order, then keep only tensors that received a GGUF name.

    Raises:
        ExportError: If two tensors end up with the same GGUF name.
    """
    for rule in rules:
        tensors = rule.apply(tensors, config)

    named = [tensor for tensor in tensors if tensor.gguf_name]
    unnamed = [tensor.hf_name for tensor in tensors if not tensor.gguf_name]
    if unnamed:
        logger.warning(
            "GGUF export: %d tensor(s) left unnamed by the tensor rules were dropped: %s",
            len(unnamed),
            ", ".join(unnamed),
        )

    duplicates = [name for name, count in Counter(t.gguf_name for t in named).items() if count > 1]
    if duplicates:
        msg = f"Tensor rules produced duplicate GGUF tensor names: {', '.join(duplicates)}"
        raise ExportError(msg)
    return named
