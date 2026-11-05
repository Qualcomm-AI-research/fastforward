# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Public ``ArchAdapter`` and ``GgufQuantFormat`` — the seams by which users configure the GGUF pipeline.

FastForward's GGUF pipeline is deliberately model-agnostic: every HuggingFace
parameter-name assumption, every architecture-specific tensor rewrite, and
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

    from fastforward.export.stages.gguf import DROP_TIED_LM_HEAD, ArchAdapter, Rename

    def my_name_map(hf_name: str) -> str | None:
        # Map user's own parameter naming onto GGUF's blk.<i>.<slot>.weight scheme.
        ...

    adapter = ArchAdapter(
        gguf_arch="my-arch",
        tensor_rules=(DROP_TIED_LM_HEAD, Rename(my_name_map)),
        write_metadata=my_write_metadata,
        tokenizer_model="gpt2",
        tokenizer_pre="default",
    )

See :mod:`fastforward.export.stages.gguf.rules` for the available tensor rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Protocol, Sequence, TypeAlias, runtime_checkable

import torch

from gguf import GGUFWriter

from fastforward.export.stages.gguf._config import GgufSourceConfig
from fastforward.export.stages.gguf._vocab import write_bpe_vocab

if TYPE_CHECKING:
    from fastforward.export.stages.gguf.rules import TensorRule

_WriteMetadataT: TypeAlias = Callable[[GGUFWriter, GgufSourceConfig], None]
WriteVocabT: TypeAlias = Callable[[GGUFWriter, Any, GgufSourceConfig, "ArchAdapter"], None]
"""A vocabulary writer used as ``ArchAdapter.write_vocab``.

Called as ``write_vocab(writer, tokenizer, config, adapter)`` with an open
``GGUFWriter``, the tokenizer passed to the export (any object the writer
understands), the source config, and the adapter (for ``tokenizer_model`` and
``tokenizer_pre``). Writers that read the tokenizer's files from disk can use
:func:`~fastforward.export.stages.gguf.tokenizer_dir`.
"""


@runtime_checkable
class PackFn(Protocol):
    """Protocol for GGUF block-packing functions.

    A pack function converts per-block quantization data from FastForward's
    internal representation into the raw byte layout expected by llama.cpp.

    Implementations receive integer codes, per-block scales, and (for asymmetric
    formats) per-block offsets, and must return a ``uint8`` tensor containing
    the packed block bytes.

    Example::

        def pack_q5_0_blocks(
            int_codes: torch.Tensor,
            scales: torch.Tensor,
            offsets: torch.Tensor | None = None,
        ) -> torch.Tensor:
            ...  # produce (n_blocks, 22) uint8 tensor
            return packed
    """

    def __call__(
        self,
        int_codes: torch.Tensor,
        scales: torch.Tensor,
        offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pack quantized blocks into raw GGUF bytes.

        Args:
            int_codes: ``(n_blocks, block_size)`` int8 — FastForward signed codes.
            scales: ``(n_blocks,)`` float32 — per-block scale factor.
            offsets: ``(n_blocks,)`` float32 for asymmetric formats, ``None`` for
                symmetric. Integer-valued: FastForward quantizes with the rounded
                offset, and extraction passes it rounded. The caller always
                passes all three positional args.

        Returns:
            ``(n_blocks, block_bytes)`` uint8 — the packed byte blocks.
        """
        ...


@dataclass(frozen=True, slots=True)
class GgufQuantFormat:
    """Describes a GGUF block-quantized format for the packing and writing stages.

    Each GGUF quant type has a fixed block size (number of elements per block),
    a pack function that produces the raw byte layout, and a GGML type name that
    selects the correct dequantizer in llama.cpp.

    Custom packing formats can be defined by implementing a function matching
    the :class:`PackFn` protocol and wrapping it in a ``GgufQuantFormat``::

        GGUF_Q5_0 = GgufQuantFormat(
            name="Q5_0", num_bits=5, block_size=32, block_bytes=22,
            symmetric=True, pack_fn=pack_q5_0_blocks, file_type=8,
        )

    Attributes:
        name: GGML type name (e.g. ``"Q4_0"``, ``"Q8_0"``). Must match a key in
            ``gguf.GGMLQuantizationType`` so the writer can look up the enum.
        num_bits: Number of quantization bits (e.g. 4 for Q4_0, 8 for Q8_0).
            Used to validate that the user's quantizers match the target format.
        block_size: Number of elements per quantized block (e.g. 32 for Q4_0/Q8_0,
            256 for K-quants).
        block_bytes: Number of raw bytes per packed block (e.g. 18 for Q4_0, 20
            for Q4_1, 34 for Q8_0). Used for output validation.
        symmetric: Whether the format requires symmetric quantization. Q4_0 and
            Q8_0 are symmetric (only ``d`` per block); Q4_1 is asymmetric
            (stores both ``d`` and ``m``).
        pack_fn: Callable matching the :class:`PackFn` protocol that produces
            the raw byte layout for this quant type.
        file_type: GGUF file-type integer written to the header (used by llama.cpp
            to select a default compute type).
    """

    name: str
    num_bits: int
    block_size: int
    block_bytes: int
    symmetric: bool
    pack_fn: PackFn
    file_type: int


@dataclass(frozen=True, slots=True)
class ArchAdapter:
    """Bundle of architecture-specific choices consumed by the GGUF stages.

    This is the whole surface between generic FastForward code and
    model-specific knowledge: what the tensors become (``tensor_rules``) and
    what the header says (``gguf_arch``, ``write_metadata``, tokenizer
    discriminators and vocabulary writer). Per-export policy (float output
    types, allowed quant formats) is not architecture knowledge and lives on
    ``GgufLlamaCppOptions`` instead.

    Attributes:
        gguf_arch: Architecture string passed to ``GGUFWriter(arch=)``. Written
            into the GGUF header and used by llama.cpp to pick its loader.
        tensor_rules: Ordered :class:`~fastforward.export.stages.gguf.rules.TensorRule`
            sequence rewriting the extracted model tensors into GGUF tensors
            (renaming, per-tensor transforms, fusion, dropping, synthesizing).
            Only model parameters claimed by a rule are extracted, and only
            tensors that end up with a GGUF name are written. See
            :mod:`fastforward.export.stages.gguf.rules`.
        write_metadata: Writes architecture metadata (dims, RoPE, vocab size,
            head counts) onto an open ``GGUFWriter`` from the source config.
        tokenizer_model: Value passed to ``writer.add_tokenizer_model(...)``
            (llama.cpp's tokenizer-model discriminator, e.g. ``"gpt2"``).
        tokenizer_pre: Value passed to ``writer.add_tokenizer_pre(...)``
            (llama.cpp's pre-tokenizer discriminator, e.g. ``"llama-bpe"``,
            ``"qwen2"``, or ``"default"``). ``None`` writes no pre-tokenizer,
            for tokenizer models that don't use one (e.g. ``"gemma4"``).
        write_vocab: :data:`WriteVocabT` writing the tokenizer's vocabulary,
            called only when the export is given a tokenizer. Defaults to
            :func:`~fastforward.export.stages.gguf.write_bpe_vocab`; use
            :func:`~fastforward.export.stages.gguf.make_llama_hf_vocab_writer`
            for byte-fallback BPE tokenizers converted from SentencePiece, or
            supply your own for other tokenizer types.
    """

    gguf_arch: str
    tensor_rules: Sequence[TensorRule]
    write_metadata: _WriteMetadataT
    tokenizer_model: str
    tokenizer_pre: str | None
    write_vocab: WriteVocabT = write_bpe_vocab


class GgufFormatRegistry:
    """Registry of GGUF quantization formats, keyed by quantizer properties.

    At extraction time, each quantizer's ``(num_bits, symmetric, block_size)``
    tuple is looked up in the registry to find the matching
    :class:`GgufQuantFormat`. The default registry ships pre-loaded with all
    built-in formats (Q4_0, Q4_1, Q8_0). Users can register additional formats
    for custom quantizer/packer pairs (e.g. K-quants)::

        from fastforward.export.stages.gguf import GgufFormatRegistry, default_format_registry

        registry = default_format_registry()
        registry.register(my_k_quant_format)
    """

    def __init__(self) -> None:
        self._formats: dict[tuple[int, bool, int], GgufQuantFormat] = {}

    def register(self, fmt: GgufQuantFormat) -> None:
        """Register a format, keyed by ``(num_bits, symmetric, block_size)``."""
        key = (fmt.num_bits, fmt.symmetric, fmt.block_size)
        self._formats[key] = fmt

    def resolve(self, num_bits: int, symmetric: bool, block_size: int) -> GgufQuantFormat | None:
        """Look up the format matching the given quantizer properties."""
        return self._formats.get((num_bits, symmetric, block_size))

    def first(self) -> GgufQuantFormat:
        """Return the first registered format (for fallback/default contexts)."""
        return next(iter(self._formats.values()))
