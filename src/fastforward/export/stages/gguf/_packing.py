# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Pack FastForward integer codes + scales into GGUF block-32 quantized bytes.

These helpers convert from FastForward's signed-symmetric quantization
convention to the exact byte layout llama.cpp expects for the block-32
quantized types (``Q4_0``, ``Q8_0``). The output is fed to
``GGUFWriter.add_tensor(..., raw_dtype=...)``, which writes the bytes verbatim
without re-quantizing — this is what preserves FastForward's learned scales.

Both packers are pure ``torch`` and carry no dependency on the ``gguf`` package,
so they can be imported and unit-tested without it installed.
"""

from __future__ import annotations

import torch

from fastforward.export.stages.gguf.adapter import GgufQuantFormat


def pack_q4_0_blocks(int_codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Pack FastForward quantized data into Q4_0 GGUF blocks.

    Args:
        int_codes: ``(n_blocks, 32)`` int8, values in ``[-8, +7]`` (FastForward
            signed convention).
        scales: ``(n_blocks,)`` float32, FastForward per-block scale (positive).

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
    n_blocks = int_codes.shape[0]

    d_bytes = scales.to(torch.float16).view(torch.uint8).reshape(n_blocks, 2)

    gguf_qs = (int_codes.to(torch.int16) + 8).clamp(0, 15).to(torch.uint8)
    block_size = int_codes.shape[1]
    gguf_qs = gguf_qs.reshape(n_blocks, 2, block_size // 2)
    packed = gguf_qs[:, 0, :] | (gguf_qs[:, 1, :] << 4)

    return torch.cat([d_bytes, packed], dim=-1)


def pack_q8_0_blocks(int_codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Pack FastForward quantized data into Q8_0 GGUF blocks.

    Args:
        int_codes: ``(n_blocks, 32)`` int8, values in ``[-128, +127]`` (FastForward
            signed convention).
        scales: ``(n_blocks,)`` float32, FastForward per-block scale (positive).

    Returns:
        ``(n_blocks, 34)`` uint8 — raw Q8_0 block bytes: ``[fp16 d | 32 int8 qs]``.

    GGUF Q8_0 dequant is ``y = qs * d`` with ``d`` positive. FastForward's scale is
    already positive (symmetric), so no sign flip is applied. Codes are clipped to
    ``[-127, +127]`` because llama.cpp's reference quantizer uses ``max(|x|) / 127``
    (leaving ``-128`` unused to stay symmetric); clipping guarantees a valid
    round-trip.
    """
    n_blocks = int_codes.shape[0]

    d_bytes = scales.to(torch.float16).view(torch.uint8).reshape(n_blocks, 2)

    gguf_qs = int_codes.clamp(-127, 127).to(torch.int8).view(torch.uint8)

    return torch.cat([d_bytes, gguf_qs], dim=-1)


GGUF_Q4_0 = GgufQuantFormat(
    name="Q4_0",
    num_bits=4,
    block_size=32,
    symmetric=True,
    pack_fn=pack_q4_0_blocks,
    file_type=2,
)

GGUF_Q8_0 = GgufQuantFormat(
    name="Q8_0",
    num_bits=8,
    block_size=32,
    symmetric=True,
    pack_fn=pack_q8_0_blocks,
    file_type=7,
)
