# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Tests for GGUF block packing (Q4_0 / Q8_0).

The packers convert FastForward's positive-scale signed codes into the exact
byte layout llama.cpp reads back. The risk is entirely in the convention
(scale sign, nibble offset, low/high packing), so these tests unpack the bytes
the same way llama.cpp does and assert the round-trip reproduces
``scale * code`` within fp16 tolerance.
"""

from typing import Callable

import numpy as np
import numpy.typing as npt
import pytest
import torch

from fastforward.export.stages.gguf._packing import (
    GGUF_Q4_0,
    pack_q4_0_blocks,
    pack_q8_0_blocks,
)

BLOCK_SIZE = GGUF_Q4_0.block_size

_PackFnT = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _dequantize_q4_0(block_bytes: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    """Dequantize Q4_0 blocks the way llama.cpp does: ``y = d * (qs - 8)``."""
    n_blocks = block_bytes.shape[0]
    d = block_bytes[:, :2].copy().view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    qs = block_bytes[:, 2:]
    low = (qs & 0x0F).astype(np.int16) - 8
    high = (qs >> 4).astype(np.int16) - 8
    nibbles = np.concatenate([low, high], axis=1).astype(np.float32)
    return d * nibbles


def _dequantize_q8_0(block_bytes: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    """Dequantize Q8_0 blocks the way llama.cpp does: ``y = d * qs``."""
    n_blocks = block_bytes.shape[0]
    d = block_bytes[:, :2].copy().view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    qs = block_bytes[:, 2:].view(np.int8).astype(np.float32)
    return d * qs


def test_pack_q4_0_block_layout() -> None:
    # GIVEN: two blocks of signed 4-bit codes and positive scales.
    int_codes = torch.zeros(2, BLOCK_SIZE, dtype=torch.int8)
    int_codes[0, 0] = -8
    int_codes[0, 1] = 7
    int_codes[1, :] = 3
    scales = torch.tensor([0.5, 2.0], dtype=torch.float32)

    # WHEN: packing to Q4_0 bytes.
    packed = pack_q4_0_blocks(int_codes, scales)

    # THEN: each block is 18 bytes (fp16 scale + 16 nibble bytes) and the stored
    # scale is positive (matches FastForward's convention, not a negated one).
    assert packed.shape == (2, 18)
    assert packed.dtype == torch.uint8
    packed_np = packed.numpy()
    stored_d = packed_np[:, :2].copy().view(np.float16).astype(np.float32)
    np.testing.assert_allclose(stored_d.reshape(-1), scales.numpy(), rtol=1e-3)

    # THEN: the signed codes map to unsigned nibbles via a +8 offset.
    first_low_nibble = int(packed[0, 2]) & 0x0F
    assert first_low_nibble == 0  # code -8 -> 0
    second_low_nibble = int(packed[0, 2]) >> 4  # element 16 of block 0 is code 0 -> 8
    assert second_low_nibble == 8


def test_pack_q4_0_round_trip_matches_scale_times_code() -> None:
    # GIVEN: random signed 4-bit codes in [-8, 7] with positive per-block scales.
    rng = np.random.default_rng(0)
    n_blocks = 16
    int_codes_np = rng.integers(-8, 8, size=(n_blocks, BLOCK_SIZE)).astype(np.int8)
    scales_np = rng.uniform(0.05, 2.0, size=n_blocks).astype(np.float32)

    int_codes = torch.from_numpy(int_codes_np)
    scales = torch.from_numpy(scales_np)

    # WHEN: packing then dequantizing with llama.cpp's Q4_0 formula.
    reconstructed = _dequantize_q4_0(pack_q4_0_blocks(int_codes, scales).numpy())

    # THEN: it reproduces scale * code within fp16 tolerance.
    expected = scales_np[:, None] * int_codes_np.astype(np.float32)
    np.testing.assert_allclose(reconstructed, expected, atol=1e-2)


def test_pack_q8_0_round_trip_matches_scale_times_code() -> None:
    # GIVEN: random signed 8-bit codes in [-127, 127] with positive scales.
    rng = np.random.default_rng(1)
    n_blocks = 16
    int_codes_np = rng.integers(-127, 128, size=(n_blocks, BLOCK_SIZE)).astype(np.int8)
    scales_np = rng.uniform(0.001, 0.5, size=n_blocks).astype(np.float32)

    int_codes = torch.from_numpy(int_codes_np)
    scales = torch.from_numpy(scales_np)

    # WHEN: packing then dequantizing with llama.cpp's Q8_0 formula.
    reconstructed = _dequantize_q8_0(pack_q8_0_blocks(int_codes, scales).numpy())

    # THEN: it reproduces scale * code within fp16 tolerance.
    expected = scales_np[:, None] * int_codes_np.astype(np.float32)
    np.testing.assert_allclose(reconstructed, expected, atol=1e-2)


def test_pack_q8_0_clips_negative_128() -> None:
    # GIVEN: a block containing the -128 code (unused by llama.cpp's symmetric Q8_0).
    int_codes = torch.full((1, BLOCK_SIZE), -128, dtype=torch.int8)
    scales = torch.tensor([0.1], dtype=torch.float32)

    # WHEN: packing to Q8_0.
    packed = pack_q8_0_blocks(int_codes, scales)

    # THEN: the stored codes are clipped to -127 to stay symmetric.
    stored = packed[:, 2:].numpy().view(np.int8)
    assert stored.min() == -127


@pytest.mark.parametrize("pack_fn, block_bytes", [(pack_q4_0_blocks, 18), (pack_q8_0_blocks, 34)])
def test_pack_block_byte_width(pack_fn: _PackFnT, block_bytes: int) -> None:
    # GIVEN: a single zero block.
    int_codes = torch.zeros(1, BLOCK_SIZE, dtype=torch.int8)
    scales = torch.ones(1, dtype=torch.float32)

    # WHEN / THEN: the packed block has the GGUF-defined byte width.
    assert pack_fn(int_codes, scales).shape == (1, block_bytes)
