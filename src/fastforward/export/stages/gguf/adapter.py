# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Public ``ArchAdapter`` and ``GgufQuantFormat`` — the seams by which users configure the GGUF pipeline.

FastForward's GGUF pipeline is deliberately model-agnostic: every HuggingFace
parameter-name assumption, every architecture-specific tensor transform, and
every tokenizer-metadata knob lives on the adapter passed by the user. Built-in
adapters (``LLAMA_ADAPTER``, ``QWEN2_ADAPTER``, ``QWEN3_ADAPTER``) are shipped
alongside as reference implementations; for a model whose module tree differs
(custom wrapper class, LoRA-merged copy, non-HF codebase), construct your own
``ArchAdapter`` and pass it via ``GgufLlamaCppOptions.arch_adapter``.

Similarly, the quantization format (block size, byte layout, GGML type) is
encapsulated in :class:`GgufQuantFormat`. Built-in formats ``GGUF_Q4_0`` and
``GGUF_Q8_0`` are provided. Users targeting a custom GGUF type construct their
own ``GgufQuantFormat`` with the appropriate pack function.

Example::

    from fastforward.export.stages.gguf import ArchAdapter, GgufQuantFormat

    def my_name_map(hf_name: str) -> str | None:
        # Map user's own parameter naming onto GGUF's blk.<i>.<slot>.weight scheme.
        ...

    adapter = ArchAdapter(
        gguf_arch="my-arch",
        name_map=my_name_map,
        transforms=[],
        write_metadata=my_write_metadata,
        tokenizer_model="gpt2",
        tokenizer_pre="default",
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, TypeAlias

import torch

from gguf import GGUFWriter

from fastforward.export.stages.gguf._config import GgufSourceConfig

if TYPE_CHECKING:
    from fastforward.export.stages.gguf._extract import ExtractedTensor

_NameMapT: TypeAlias = Callable[[str], str | None]
_WriteMetadataT: TypeAlias = Callable[[GGUFWriter, GgufSourceConfig], None]
_IsTiedT: TypeAlias = Callable[[str, GgufSourceConfig], bool]
_PackFnT: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
TensorTransformT: TypeAlias = Callable[
    ["ExtractedTensor", GgufSourceConfig, "GgufQuantFormat"], "ExtractedTensor"
]


def _default_is_tied(hf_name: str, config: GgufSourceConfig) -> bool:
    """Default tied-weight predicate: skips ``lm_head.weight`` when config indicates tying."""
    if not getattr(config, "tie_word_embeddings", False):
        return False
    return hf_name == "lm_head.weight"


@dataclass(frozen=True, slots=True)
class GgufQuantFormat:
    """Describes a GGUF block-quantized format for the packing and writing stages.

    Each GGUF quant type has a fixed block size (number of elements per block),
    a pack function that produces the raw byte layout, and a GGML type name that
    selects the correct dequantizer in llama.cpp.

    Attributes:
        name: GGML type name (e.g. ``"Q4_0"``, ``"Q8_0"``). Must match a key in
            ``gguf.GGMLQuantizationType`` so the writer can look up the enum.
        num_bits: Number of quantization bits (e.g. 4 for Q4_0, 8 for Q8_0).
            Used to validate that the user's quantizers match the target format.
        block_size: Number of elements per quantized block (e.g. 32 for Q4_0/Q8_0,
            256 for K-quants).
        symmetric: Whether the format requires symmetric quantization. Q4_0 and
            Q8_0 are symmetric (only ``d`` per block); Q4_1 is asymmetric
            (stores both ``d`` and ``m``).
        pack_fn: Callable ``(int_codes: Tensor[n_blocks, block_size],
            scales: Tensor[n_blocks]) -> Tensor[n_blocks, block_bytes]`` that
            produces the raw byte layout for this quant type.
        file_type: GGUF file-type integer written to the header (used by llama.cpp
            to select a default compute type).
    """

    name: str
    num_bits: int
    block_size: int
    symmetric: bool
    pack_fn: _PackFnT
    file_type: int


@dataclass(frozen=True, slots=True)
class ArchAdapter:
    """Bundle of architecture-specific choices consumed by the GGUF stages.

    Every field is consumed by exactly one stage — this is the whole surface
    between generic FastForward code and model-specific knowledge.

    Attributes:
        gguf_arch: Architecture string passed to ``GGUFWriter(arch=)``. Written
            into the GGUF header and used by llama.cpp to pick its loader.
        name_map: Maps a HuggingFace parameter name (e.g.
            ``"model.layers.0.self_attn.q_proj.weight"``) to its GGUF tensor
            name (``"blk.0.attn_q.weight"``), or ``None`` to skip the parameter.
        transforms: Ordered list of per-tensor transform functions applied by
            ``stage_apply_target_transforms``. Each receives an
            :class:`ExtractedTensor`, the source config, and the quant format,
            and returns a (possibly modified) ``ExtractedTensor``. Built-in
            helpers like :func:`llama_rope_permute` slot in here. An empty list
            means no transforms are applied.
        write_metadata: Writes architecture metadata (dims, RoPE, vocab size,
            head counts) onto an open ``GGUFWriter`` from the source config.
        tokenizer_model: Value passed to ``writer.add_tokenizer_model(...)``
            (llama.cpp's tokenizer-model discriminator, e.g. ``"gpt2"``).
        tokenizer_pre: Value passed to ``writer.add_tokenizer_pre(...)``
            (llama.cpp's pre-tokenizer discriminator, e.g. ``"llama-bpe"``,
            ``"qwen2"``, or ``"default"``).
        is_tied: Predicate that decides whether a parameter should be skipped
            because it shares storage with another exported parameter. Receives
            the HF parameter name and the source config. The default
            implementation skips ``"lm_head.weight"`` when
            ``config.tie_word_embeddings`` is true. Override for architectures
            with different or additional tied parameters.
        float_type: GGML type name for float tensors (e.g. ``"F32"``,
            ``"F16"``, ``"BF16"``). All non-quantized tensors are cast to this
            type at write time. Defaults to ``"F32"``.
        float_type_overrides: Mapping of regex patterns (matched against the
            GGUF tensor name) to GGML type names. First matching pattern wins;
            unmatched tensors fall back to ``float_type``. Use this to keep
            specific layers at higher precision (e.g. norms at F32 while the
            rest is F16).
    """

    gguf_arch: str
    name_map: _NameMapT
    transforms: list[TensorTransformT]
    write_metadata: _WriteMetadataT
    tokenizer_model: str
    tokenizer_pre: str
    is_tied: _IsTiedT = field(default=_default_is_tied)
    float_type: str = "F32"
    float_type_overrides: dict[str, str] = field(default_factory=dict)
