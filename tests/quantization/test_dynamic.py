# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import fastforward as ff
import fastforward.quantization.affine.dynamic as dynamic
import pytest
import torch

DEVICES = ["cpu", "cuda"]
NUM_BITS = [3, 4, 8]


@pytest.mark.parametrize("num_bits", NUM_BITS)
@pytest.mark.parametrize("device", DEVICES)
def test_dynamic_quantizer(num_bits: int, device: torch.device | str, _seed_prngs: int) -> None:
    channel_idx = -1
    x = torch.randn(32, 8)
    out = dynamic.quantize_per_channel(x, channel_idx, num_bits).dequantize()

    quantizer = ff.nn.LinearQuantizer(
        num_bits, symmetric=False, granularity=ff.PerChannel(channel_idx), allow_one_sided=True
    )
    quantizer.quantization_range = (x.min(0).values, x.max(0).values)
    expected_out = quantizer(x).dequantize()

    assert (out - expected_out).max() == 0
