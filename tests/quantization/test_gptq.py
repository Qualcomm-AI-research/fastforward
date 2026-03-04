# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import fastforward as ff
import pytest
import torch

from fastforward.quantization.gptq import column_quantizer


def _calibrated_quantizer(
    granularity: ff.granularity.Granularity, weights: torch.Tensor
) -> ff.nn.LinearQuantizer:
    """Create and calibrate a LinearQuantizer on the given weights."""
    quantizer = ff.nn.LinearQuantizer(num_bits=4, granularity=granularity, symmetric=False)
    with (
        ff.strict_quantization(False),
        ff.estimate_ranges(quantizer, ff.range_setting.smoothed_minmax),
    ):
        quantizer(weights)
    return quantizer


@pytest.mark.parametrize(
    "granularity",
    [
        ff.granularity.PerTensor(),
        ff.granularity.PerChannel(channel_dim=0),
        ff.granularity.PerChannel(channel_dim=1),
        ff.granularity.PerChannel(channel_dim=(0, 1)),
        ff.granularity.PerBlock(block_dims=1, block_sizes=16, per_channel_dims=0),
        ff.granularity.PerBlock(block_dims=(0, 1), block_sizes=(16, 16)),
        ff.granularity.PerTile((16, 16)),
    ],
    ids=[
        "per_tensor",
        "per_channel_dim0",
        "per_channel_dim1",
        "per_channel_dim01",
        "per_block_col16",
        "per_block_16x16",
        "per_tile_16x16",
    ],
)
def test_column_quantizer_matches_full_quantizer(granularity: ff.granularity.Granularity) -> None:
    # Given: a calibrated quantizer and a weight matrix
    weights = torch.randn(64, 128)
    quantizer = _calibrated_quantizer(granularity, weights)

    # When: we quantize the full matrix, then quantize each column individually
    with ff.strict_quantization(False):
        expected = quantizer(weights).dequantize()

    result = torch.zeros_like(weights)
    for col in range(weights.shape[1]):
        quant_deq = column_quantizer(quantizer, weights.shape, col)
        result[:, col] = quant_deq(weights[:, col])

    # Then: the per-column results match the full quantization exactly
    torch.testing.assert_close(result, expected)
