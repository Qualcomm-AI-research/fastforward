# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Any

import fastforward as ff
import pytest
import torch

from fastforward.quantization.gptq import column_quantizer, gptq, update_partial_range


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


def test_update_partial_range_writes_expected_scale_and_offset() -> None:
    # GIVEN a calibrated grouped quantizer and one column group's weights
    granularity = ff.granularity.PerBlock(block_dims=1, block_sizes=8, per_channel_dims=0)
    torch.manual_seed(0)
    weights = torch.randn(16, 32)
    quantizer = _calibrated_quantizer(granularity, weights)
    group_idx = 2
    group_weights = weights[:, group_idx * 8 : (group_idx + 1) * 8]

    # WHEN we update the scale/offset for that one column group
    update_partial_range(
        quantizer,
        group_weights.min(dim=-1).values,
        group_weights.max(dim=-1).values,
        param_view_shape=(16, 4),
        param_view_index=(slice(None), group_idx),
    )

    # THEN the written scale/offset match what parameters_for_range gives for that slice
    expected_scale, expected_offset = ff.quantization.affine.parameters_for_range(
        group_weights.min(dim=-1).values,
        group_weights.max(dim=-1).values,
        num_bits=quantizer.num_bits,
        symmetric=quantizer.symmetric,
        allow_one_sided=quantizer.allow_one_sided,
    )
    actual_scale = quantizer.scale.data.view(16, 4)[:, group_idx]
    torch.testing.assert_close(actual_scale, expected_scale.to(actual_scale.dtype))
    if quantizer.offset is not None and expected_offset is not None:
        actual_offset = quantizer.offset.data.view(16, 4)[:, group_idx]
        torch.testing.assert_close(actual_offset, expected_offset.to(actual_offset.dtype))


def test_update_partial_range_clears_stale_offset_on_two_sided_range() -> None:
    # GIVEN a symmetric quantizer calibrated on all-positive weights, so the one-sided path
    # fills the offset buffer with a non-zero value
    granularity = ff.granularity.PerBlock(block_dims=1, block_sizes=8, per_channel_dims=0)
    torch.manual_seed(0)
    weights = torch.rand(16, 32) + 0.1
    quantizer = ff.nn.LinearQuantizer(num_bits=4, granularity=granularity, symmetric=True)
    with (
        ff.strict_quantization(False),
        ff.estimate_ranges(quantizer, ff.range_setting.smoothed_minmax),
    ):
        quantizer(weights)
    assert quantizer.offset is not None
    assert (quantizer.offset.data != 0).any()

    # WHEN one group is updated with a range that spans zero, for which
    # `parameters_for_range` returns no offset
    group_idx = 1
    group_weights = weights[:, group_idx * 8 : (group_idx + 1) * 8].clone()
    group_weights[:, 0] = -0.9
    min_range = group_weights.min(dim=-1).values
    max_range = group_weights.max(dim=-1).values
    update_partial_range(
        quantizer,
        min_range,
        max_range,
        param_view_shape=(16, 4),
        param_view_index=(slice(None), group_idx),
    )

    # THEN the stale offset is cleared for that group and the resulting range covers the
    # negative half instead of clipping it
    scale = quantizer.scale.data.view(16, 4)[:, group_idx]
    offset = quantizer.offset.data.view(16, 4)[:, group_idx]
    torch.testing.assert_close(offset, torch.zeros_like(offset))
    range_min, range_max = ff.quantization.affine.quantization_range(
        scale, offset, quantizer.num_bits
    )
    assert (range_min <= min_range).all()
    # A small tolerance is needed because scale/offset are computed in float32.
    assert (range_max >= max_range - 1e-6).all()

    # AND the other groups keep their calibrated offsets
    other_offsets = quantizer.offset.data.view(16, 4)[:, [0, 2, 3]]
    assert (other_offsets != 0).all()


def test_gptq_recomputes_grouped_scales_from_error_corrected_weights() -> None:
    # GIVEN a grouped quantizer whose second column group would receive a distinctly
    # different scale after GPTQ propagates group 0's rounding error into it
    granularity = ff.granularity.PerBlock(block_dims=1, block_sizes=8, per_channel_dims=0)
    torch.manual_seed(0)
    weights = torch.randn(16, 32)
    module = ff.nn.QuantizedLinear(32, 16, bias=False)
    with torch.no_grad():
        module.weight.copy_(weights)
    module.weight_quantizer = ff.nn.LinearQuantizer(
        num_bits=4, granularity=granularity, symmetric=True
    )
    dataset: list[tuple[tuple[Any, ...], dict[str, Any]]] = [
        ((torch.randn(1, 8, 32),), {}) for _ in range(2)
    ]

    # WHEN we run GPTQ (grouped, no actorder → recompute path fires)
    with torch.inference_mode(), ff.strict_quantization(False):
        gptq(module, dataset, block_size=8, perc_damp=0.01, actorder=False)

    # THEN group 1's scale reflects error-corrected weights, not the original slice.
    # Compare against what parameters_for_range would give for the *original* weight slice
    # — that's the scale the pre-fix (static-groups) code would have kept.
    original_slice = weights[:, 8:16]
    static_scale, _ = ff.quantization.affine.parameters_for_range(
        original_slice.min(dim=-1).values,
        original_slice.max(dim=-1).values,
        num_bits=module.weight_quantizer.num_bits,
        symmetric=module.weight_quantizer.symmetric,
        allow_one_sided=module.weight_quantizer.allow_one_sided,
    )
    actual_group1_scale = module.weight_quantizer.scale.data.view(16, 4)[:, 1]
    assert not torch.allclose(actual_group1_scale, static_scale.to(actual_group1_scale.dtype))
