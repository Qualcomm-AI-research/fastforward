# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

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
    x = torch.randn(32, 8, device=device)
    out = dynamic.quantize_per_channel(x, channel_idx, num_bits).dequantize()

    quantizer = ff.nn.LinearQuantizer(
        num_bits,
        symmetric=False,
        granularity=ff.PerChannel(channel_idx),
        allow_one_sided=True,
        device=device,
    )
    quantizer.quantization_range = (x.min(0).values, x.max(0).values)
    expected_out = quantizer(x).dequantize()

    assert (out - expected_out).max() == 0


def test_quantized_value_precision_loss() -> None:
    """Test that exception is raised when output_dtype is too small to keep the quantized value perprecisely."""
    # GIVEN a tensor
    data = torch.tensor([1.0], dtype=torch.float32)

    # WHEN the tensor is dynamically quantized to a bitwidth that cannot be
    # represented in the provided output_dtype
    # THEN a RuntimeError is raised
    with pytest.raises(RuntimeError):
        dynamic.quantize_per_tensor(data, 16, output_dtype=torch.bfloat16)


def test_dynamic_quantizer_symmetric_onesided_control(_seed_prngs: int) -> None:
    x = torch.rand(32, 8)

    out_allow = dynamic.quantize_per_channel(
        x, -1, 4, symmetric=True, allow_one_sided=True
    ).dequantize()
    out_no_allow = dynamic.quantize_per_channel(
        x, -1, 4, symmetric=True, allow_one_sided=False
    ).dequantize()

    assert not torch.equal(out_allow, out_no_allow)


@pytest.mark.parametrize("allow_one_sided", [True, False])
def test_dynamic_symmetric_matches_static_per_channel(
    allow_one_sided: bool, _seed_prngs: int
) -> None:
    x = torch.rand(8, 4)

    actual = dynamic.quantize_per_channel(
        x,
        -1,
        4,
        symmetric=True,
        allow_one_sided=allow_one_sided,
    ).dequantize()

    quantizer = ff.nn.LinearQuantizer(
        4,
        symmetric=True,
        allow_one_sided=allow_one_sided,
        granularity=ff.PerChannel(-1),
    )
    quantizer.quantization_range = (x.min(0).values, x.max(0).values)
    expected = quantizer(x).dequantize()

    torch.testing.assert_close(actual, expected)
