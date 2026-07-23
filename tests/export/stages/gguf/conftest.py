# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Shared test fixtures and helpers for the GGUF export stages."""

import torch

from fastforward.export.stages.gguf._extract import ExtractedTensor
from fastforward.export.stages.gguf._packing import GGUF_Q4_0


def make_quantized_tensor(hf_name: str, rows: int, cols: int) -> ExtractedTensor:
    """Build a minimal quantized ExtractedTensor for transform/stage tests."""
    block_size = GGUF_Q4_0.block_size
    n_blocks = (rows * cols) // block_size
    gen = torch.manual_seed(0)
    return ExtractedTensor(
        hf_name=hf_name,
        kind="quantized",
        rows=rows,
        cols=cols,
        int_codes=torch.randint(-8, 8, (n_blocks, block_size), generator=gen, dtype=torch.int8),
        scales=torch.rand(n_blocks, generator=gen) * 0.95 + 0.05,
    )
