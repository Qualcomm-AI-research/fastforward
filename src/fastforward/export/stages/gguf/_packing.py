# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Pack FastForward integer codes + scales into GGUF block-32 quantized bytes.

These helpers convert from FastForward's quantization convention to the exact
byte layout llama.cpp expects for the block-32 quantized types (``Q4_0``,
``Q4_1``, ``Q8_0``). The output is fed to
``GGUFWriter.add_tensor(..., raw_dtype=...)``, which writes the bytes verbatim
without re-quantizing — this is what preserves FastForward's learned scales.

All packers are pure ``torch`` and carry no dependency on the ``gguf`` package,
so they can be imported and unit-tested without it installed.
"""

from __future__ import annotations

import torch

from fastforward.export.stages.gguf.adapter import GgufFormatRegistry, GgufQuantFormat


def pack_q4_0_blocks(
    int_codes: torch.Tensor, scales: torch.Tensor, offsets: torch.Tensor | None = None
) -> torch.Tensor:
    """Pack FastForward quantized data into Q4_0 GGUF blocks.

    Args:
        int_codes: ``(n_blocks, 32)`` int8, values in ``[-8, +7]`` (FastForward
            signed convention).
        scales: ``(n_blocks,)`` float32, FastForward per-block scale (positive).
        offsets: Unused (Q4_0 is symmetric). Accepted for :class:`PackFn`
            protocol conformance.

    Returns:
        ``(n_blocks, 18)`` uint8 — raw Q4_0 block bytes: ``[fp16 d | 16 nibble bytes]``.

    Convention conversion:
        FastForward stores ``deq = scale * code`` with a positive ``scale`` and
        signed ``code`` in ``[-8, +7]``. llama.cpp's Q4_0 dequant is
        ``y = d * (qs - 8)``. Reproducing FastForward's values byte-for-byte
        therefore needs ``d = +scale`` and ``qs = code + 8`` (unsigned ``[0, 15]``,
        nibble-packed low/high). This differs from packing the block maximum's sign
        into ``d`` the way llama.cpp's own quantizer does; either representation
        dequantizes to the same values, but ``d = +scale`` is the one that matches
        FastForward's learned codes.
    """
    del offsets
    n_blocks = int_codes.shape[0]

    d_bytes = scales.to(torch.float16).view(torch.uint8).reshape(n_blocks, 2)

    gguf_qs = (int_codes.to(torch.int16) + 8).clamp(0, 15).to(torch.uint8)
    block_size = int_codes.shape[1]
    gguf_qs = gguf_qs.reshape(n_blocks, 2, block_size // 2)
    packed = gguf_qs[:, 0, :] | (gguf_qs[:, 1, :] << 4)

    return torch.cat([d_bytes, packed], dim=-1)


def pack_q8_0_blocks(
    int_codes: torch.Tensor, scales: torch.Tensor, offsets: torch.Tensor | None = None
) -> torch.Tensor:
    """Pack FastForward quantized data into Q8_0 GGUF blocks.

    Args:
        int_codes: ``(n_blocks, 32)`` int8, values in ``[-128, +127]`` (FastForward
            signed convention).
        scales: ``(n_blocks,)`` float32, FastForward per-block scale (positive).
        offsets: Unused (Q8_0 is symmetric). Accepted for :class:`PackFn`
            protocol conformance.

    Returns:
        ``(n_blocks, 34)`` uint8 — raw Q8_0 block bytes: ``[fp16 d | 32 int8 qs]``.

    GGUF Q8_0 dequant is ``y = qs * d`` with ``d`` positive. FastForward's scale is
    already positive (symmetric), so no sign flip is applied. Codes are clipped to
    ``[-127, +127]`` because llama.cpp's reference quantizer uses ``max(|x|) / 127``
    (leaving ``-128`` unused to stay symmetric); clipping guarantees a valid
    round-trip.
    """
    del offsets
    n_blocks = int_codes.shape[0]

    d_bytes = scales.to(torch.float16).view(torch.uint8).reshape(n_blocks, 2)

    gguf_qs = int_codes.clamp(-127, 127).to(torch.int8).view(torch.uint8)

    return torch.cat([d_bytes, gguf_qs], dim=-1)


def pack_q4_1_blocks(
    int_codes: torch.Tensor, scales: torch.Tensor, offsets: torch.Tensor | None = None
) -> torch.Tensor:
    """Pack FastForward asymmetric quantized data into Q4_1 GGUF blocks.

    Args:
        int_codes: ``(n_blocks, 32)`` int8, values in ``[-8, +7]`` (FastForward
            signed convention).
        scales: ``(n_blocks,)`` float32, FastForward per-block scale.
        offsets: ``(n_blocks,)`` float32, FastForward per-block offset.

    Returns:
        ``(n_blocks, 20)`` uint8 — raw Q4_1 block bytes: ``[fp16 d | fp16 m | 16 nibble bytes]``.

    Convention conversion:
        FastForward dequant: ``x = (code + offset) * scale`` with signed
        ``code`` in ``[-8, +7]``. GGUF Q4_1 dequant: ``y = d * qs + m`` with
        unsigned ``qs`` in ``[0, 15]``. Substituting ``code = qs - 8`` gives
        ``d = scale`` and ``m = (offset - 8) * scale``.
    """
    if offsets is None:
        msg = "Q4_1 requires per-block offsets (asymmetric quantization)"
        raise ValueError(msg)
    n_blocks = int_codes.shape[0]

    d_bytes = scales.to(torch.float16).view(torch.uint8).reshape(n_blocks, 2)
    m = (offsets - 8) * scales
    m_bytes = m.to(torch.float16).view(torch.uint8).reshape(n_blocks, 2)

    gguf_qs = (int_codes.to(torch.int16) + 8).clamp(0, 15).to(torch.uint8)
    block_size = int_codes.shape[1]
    gguf_qs = gguf_qs.reshape(n_blocks, 2, block_size // 2)
    packed = gguf_qs[:, 0, :] | (gguf_qs[:, 1, :] << 4)

    return torch.cat([d_bytes, m_bytes, packed], dim=-1)


GGUF_Q4_0 = GgufQuantFormat(
    name="Q4_0",
    num_bits=4,
    block_size=32,
    block_bytes=18,
    symmetric=True,
    pack_fn=pack_q4_0_blocks,
    file_type=2,
)
"""4-bit symmetric per-block quantization.

Requires a FastForward quantizer configured as:
- ``num_bits=4``
- ``symmetric=True``
- ``granularity=PerBlock(block_sizes=(32,), block_dims=(1,), per_channel_dims=(0,))``

Block layout (18 bytes): ``[fp16 scale | 16 nibble-packed code bytes]``.
Dequantization: ``x = scale * (code - 8)``.
"""

GGUF_Q8_0 = GgufQuantFormat(
    name="Q8_0",
    num_bits=8,
    block_size=32,
    block_bytes=34,
    symmetric=True,
    pack_fn=pack_q8_0_blocks,
    file_type=7,
)
"""8-bit symmetric per-block quantization.

Requires a FastForward quantizer configured as:
- ``num_bits=8``
- ``symmetric=True``
- ``granularity=PerBlock(block_sizes=(32,), block_dims=(1,), per_channel_dims=(0,))``

Block layout (34 bytes): ``[fp16 scale | 32 int8 codes]``.
Dequantization: ``x = scale * code``.
"""

GGUF_Q4_1 = GgufQuantFormat(
    name="Q4_1",
    num_bits=4,
    block_size=32,
    block_bytes=20,
    symmetric=False,
    pack_fn=pack_q4_1_blocks,
    file_type=3,
)
"""4-bit asymmetric per-block quantization.

Requires a FastForward quantizer configured as:
- ``num_bits=4``
- ``symmetric=False``
- ``granularity=PerBlock(block_sizes=(32,), block_dims=(1,), per_channel_dims=(0,))``

Block layout (20 bytes): ``[fp16 scale | fp16 min | 16 nibble-packed code bytes]``.
Dequantization: ``x = scale * code + min``.
"""


def default_format_registry() -> GgufFormatRegistry:
    """Return a :class:`GgufFormatRegistry` pre-loaded with all built-in formats."""
    registry = GgufFormatRegistry()
    registry.register(GGUF_Q4_0)
    registry.register(GGUF_Q4_1)
    registry.register(GGUF_Q8_0)
    return registry
