# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import logging
import re

from typing import Any, cast

import fastforward as ff
import pytest
import torch

from fastforward.algorithms.adaround import (
    _beta,
    _SoftRounding,
    adaround,
)
from fastforward.quantization import tiled_tensor

Batches = list[tuple[tuple[Any, ...], dict[str, Any]]]


def _quantized_linear(
    granularity: ff.granularity.Granularity, num_bits: int = 4
) -> ff.nn.QuantizedLinear:
    """Create a QuantizedLinear with an uncalibrated LinearQuantizer on its weight."""
    torch.manual_seed(0)
    module = ff.nn.QuantizedLinear(64, 32, bias=False)
    module.weight_quantizer = ff.nn.LinearQuantizer(
        num_bits=num_bits, granularity=granularity, symmetric=False
    )
    return module


def _weight_quantizer(module: ff.nn.QuantizedLinear) -> ff.nn.LinearQuantizer:
    """Return the weight quantizer of `module`, narrowed to `LinearQuantizer`."""
    quantizer = module.weight_quantizer
    assert isinstance(quantizer, ff.nn.LinearQuantizer)
    return quantizer


def _calibrate(quantizer: ff.nn.LinearQuantizer, weights: torch.Tensor) -> None:
    """Calibrate `quantizer` on `weights` the way `adaround` does internally."""
    with (
        ff.strict_quantization(False),
        ff.estimate_ranges(quantizer, ff.range_setting.smoothed_minmax),
    ):
        quantizer(weights)


def _low_rank_batches(num_batches: int = 2) -> Batches:
    """Build activations concentrated on one input direction.

    AdaRound exploits correlation between input features: for isotropic activations the
    reconstruction error reduces to the weight error, which round-to-nearest already
    minimizes. A dominant direction makes the rounding choice matter.
    """
    torch.manual_seed(1)
    direction = torch.randn(64)
    coefficients = torch.randn(4, 8, 1)
    return [
        ((coefficients * direction + 0.05 * torch.randn(4, 8, 64),), {}) for _ in range(num_batches)
    ]


def _symmetric_batches(num_batches: int = 2) -> tuple[Batches, Batches]:
    """Return the same activations for both input flows of `adaround`.

    Equal activations in both flows give the symmetric layer-wise objective of Eq. 21,
    which is what most tests below need.
    """
    batches = _low_rank_batches(num_batches)
    return batches, batches


GRANULARITIES = [
    ff.granularity.PerTensor(),
    ff.granularity.PerChannel(channel_dim=0),
    ff.granularity.PerChannel(channel_dim=1),
    ff.granularity.PerChannel(channel_dim=(0, 1)),
    ff.granularity.PerBlock(block_dims=1, block_sizes=16, per_channel_dims=0),
    ff.granularity.PerBlock(block_dims=(0, 1), block_sizes=(16, 16)),
    ff.granularity.PerTile((16, 16)),
]
GRANULARITY_IDS = [
    "per_tensor",
    "per_channel_dim0",
    "per_channel_dim1",
    "per_channel_dim01",
    "per_block_col16",
    "per_block_16x16",
    "per_tile_16x16",
]


@pytest.mark.parametrize("granularity", GRANULARITIES, ids=GRANULARITY_IDS)
def test_adaround_without_optimization_matches_nearest_rounding(
    granularity: ff.granularity.Granularity,
) -> None:
    # GIVEN a layer and the round-to-nearest weight of its calibrated quantizer
    reference = _quantized_linear(granularity)
    _calibrate(_weight_quantizer(reference), reference.weight)
    with ff.strict_quantization(False), torch.no_grad():
        nearest = reference.weight_quantizer(reference.weight).dequantize()

    # WHEN we run AdaRound with no optimization steps, so the rounding variable keeps its
    # initial value
    module = _quantized_linear(granularity)
    with torch.no_grad():
        adaround(module, *_symmetric_batches(), num_iterations=0)

    # THEN the baked weight is exactly the round-to-nearest weight, for every granularity
    torch.testing.assert_close(module.weight.data, nearest)


@pytest.mark.parametrize("granularity", GRANULARITIES, ids=GRANULARITY_IDS)
def test_soft_rounding_reproduces_the_unquantized_weight_at_initialization(
    granularity: ff.granularity.Granularity,
) -> None:
    # GIVEN a calibrated quantizer and its soft rounding override
    module = _quantized_linear(granularity)
    quantizer = _weight_quantizer(module)
    _calibrate(quantizer, module.weight)
    rounding = _SoftRounding(quantizer, module.weight)

    # WHEN we evaluate the soft weight at the initial rounding variable
    soft_weight = rounding(None, None, (module.weight,), {})

    # THEN it equals the unquantized weight, clamped to the quantization range, because
    # the rounding variable is initialized so that h(V) equals the rounding residual
    assert quantizer.offset is not None
    grid_min, grid_max = ff.quantization.affine.quantization_range(
        quantizer.scale.detach(), quantizer.offset.detach().round(), quantizer.num_bits
    )
    tile_size = granularity.tile_size(module.weight.shape)
    rows = tiled_tensor.tiles_to_rows(module.weight.data, tile_size)
    clamped = rows.clamp(
        torch.as_tensor(grid_min).reshape(-1, 1), torch.as_tensor(grid_max).reshape(-1, 1)
    )
    expected = tiled_tensor.rows_to_tiles(clamped, module.weight.shape, tile_size)
    torch.testing.assert_close(soft_weight, expected)


def test_soft_rounding_hard_weight_lands_on_the_quantization_grid() -> None:
    # GIVEN a layer whose weight was replaced by an AdaRound result
    module = _quantized_linear(ff.granularity.PerChannel(channel_dim=0))
    with torch.no_grad():
        adaround(module, *_symmetric_batches(), num_iterations=20)

    # WHEN we quantize the baked weight again
    with ff.strict_quantization(False), torch.no_grad():
        requantized = module.weight_quantizer(module.weight).dequantize()

    # THEN nothing changes, so the weight is representable by the quantizer
    torch.testing.assert_close(requantized, module.weight.data)


@pytest.mark.slow
def test_adaround_reduces_the_output_error_over_nearest_rounding() -> None:
    # GIVEN a 3 bit layer, activations with a dominant direction, and the output of the
    # unquantized layer as reference
    batches = _low_rank_batches()
    activation = batches[0][0][0]
    original_weight = _quantized_linear(ff.granularity.PerChannel(channel_dim=0)).weight.data
    reference = torch.nn.functional.linear(activation, original_weight)

    nearest_module = _quantized_linear(ff.granularity.PerChannel(channel_dim=0), num_bits=3)
    _calibrate(_weight_quantizer(nearest_module), nearest_module.weight)
    with ff.strict_quantization(False), torch.no_grad():
        nearest_error = (nearest_module(activation) - reference).pow(2).mean()

    # WHEN we quantize the same layer with AdaRound. The iteration count must let the
    # regularizer decide the rounding: at a few hundred iterations most elements are still
    # undecided, and then the hard rounding of the result throws the reconstruction away.
    module = _quantized_linear(ff.granularity.PerChannel(channel_dim=0), num_bits=3)
    with torch.no_grad():
        adaround(module, batches, batches, num_iterations=2_000)
    with ff.strict_quantization(False), torch.no_grad():
        adaround_error = (module(activation) - reference).pow(2).mean()

    # THEN the output error is far below the round-to-nearest error
    assert adaround_error < 0.6 * nearest_error


@pytest.mark.slow
def test_adaround_reconstructs_the_original_output_from_the_quantized_input() -> None:
    # GIVEN original activations, activations of the same batches that a model with its
    # previous layers quantized would give, and the output of the unquantized layer on the
    # original activations, which is the target `Wx` of Eq. 25
    granularity = ff.granularity.PerChannel(channel_dim=0)
    original = _low_rank_batches()
    quantized = [((1.2 * args[0],), kwargs) for args, kwargs in original]
    original_weight = _quantized_linear(granularity, num_bits=3).weight.data.clone()
    target = torch.nn.functional.linear(original[0][0][0], original_weight)

    # WHEN we run AdaRound with the two flows, and again with the original flow twice
    asymmetric = _quantized_linear(granularity, num_bits=3)
    symmetric = _quantized_linear(granularity, num_bits=3)
    with torch.no_grad():
        adaround(asymmetric, original, quantized, num_iterations=500)
        adaround(symmetric, original, original, num_iterations=500)

    # THEN the asymmetric run reconstructs the target from the quantized activations better
    # than the symmetric run, thus the quantized activations reach the reconstruction and
    # the original activations reach the target
    quantized_activation = quantized[0][0][0]
    asymmetric_error = _output_error(asymmetric, quantized_activation, target)
    symmetric_error = _output_error(symmetric, quantized_activation, target)
    assert asymmetric_error < 0.6 * symmetric_error


def _output_error(
    module: ff.nn.QuantizedLinear, activation: torch.Tensor, target: torch.Tensor
) -> float:
    """Return the mean squared error of the layer output against `target`."""
    with ff.strict_quantization(False), torch.no_grad():
        output = cast(torch.Tensor, module(activation))
    return (output - target).pow(2).mean().item()


def test_adaround_leaves_the_module_unchanged_apart_from_the_weight() -> None:
    # GIVEN a layer with a bias and the gradient flags of its parameters
    torch.manual_seed(0)
    module = ff.nn.QuantizedLinear(64, 32, bias=True)
    module.weight_quantizer = ff.nn.LinearQuantizer(
        num_bits=4, granularity=ff.granularity.PerChannel(channel_dim=0), symmetric=False
    )
    bias = module.bias.data.clone()

    # WHEN we run AdaRound
    with torch.no_grad():
        adaround(module, *_symmetric_batches(), num_iterations=20)

    # THEN the bias is untouched, the gradient flags are restored, no gradient is left on
    # any parameter and the soft rounding override is removed again
    torch.testing.assert_close(module.bias.data, bias)
    assert all(param.requires_grad for param in module.parameters())
    assert all(param.grad is None for param in module.parameters())
    with ff.strict_quantization(False), torch.no_grad():
        assert isinstance(module.weight_quantizer(module.weight), ff.QuantizedTensor)


def _linear_with_output_quantizer() -> ff.nn.QuantizedLinear:
    """Create a QuantizedLinear with an uncalibrated quantizer on its weight and its output."""
    module = _quantized_linear(ff.granularity.PerChannel(channel_dim=0))
    module.output_quantizer = ff.nn.LinearQuantizer(
        num_bits=8, granularity=ff.granularity.PerTensor(), symmetric=False
    )
    return module


def test_adaround_optimizes_through_a_calibrated_output_quantizer() -> None:
    # GIVEN a layer whose output quantizer is calibrated before AdaRound runs
    module = _linear_with_output_quantizer()
    output_quantizer = module.output_quantizer
    assert isinstance(output_quantizer, ff.nn.LinearQuantizer)
    original_inputs, quantized_inputs = _symmetric_batches()
    with (
        ff.disable_quantization(module.weight_quantizer),
        ff.estimate_ranges(output_quantizer, ff.range_setting.smoothed_minmax),
    ):
        for args, kwargs in original_inputs:
            module(*args, **kwargs)
    scale = output_quantizer.scale.data.clone()
    original_weight = module.weight.data.clone()

    # WHEN we run AdaRound, thus the reconstruction runs through the output quantizer while the
    # target does not
    with torch.no_grad():
        adaround(module, original_inputs, quantized_inputs, num_iterations=20)

    # THEN AdaRound learns a rounding and leaves the grid of the output quantizer alone
    assert not torch.allclose(module.weight.data, original_weight)
    torch.testing.assert_close(output_quantizer.scale.data, scale)


def test_adaround_reports_an_uninitialized_activation_quantizer() -> None:
    # GIVEN a layer with an activation quantizer that has no grid yet
    module = _linear_with_output_quantizer()

    # WHEN we run AdaRound
    # THEN the quantizer reports that it is uninitialized. The lazy parameters of that quantizer
    # must not fail earlier, for example in the parameter freeze.
    with pytest.raises(ValueError, match="uninitialized quantizer"), torch.no_grad():
        adaround(module, *_symmetric_batches(), num_iterations=20)


def test_adaround_logs_the_soft_and_the_hard_loss(caplog: pytest.LogCaptureFixture) -> None:
    # GIVEN a layer, the batches of the symmetric objective and the unquantized weight, which
    # gives the target of the objective
    module = _quantized_linear(ff.granularity.PerChannel(channel_dim=0), num_bits=3)
    original_inputs, quantized_inputs = _symmetric_batches()
    original_weight = module.weight.data.clone()

    # WHEN we run AdaRound and capture the log
    with caplog.at_level(logging.INFO), torch.no_grad():
        adaround(module, original_inputs, quantized_inputs, num_iterations=20)

    # THEN the log holds one line with both losses
    messages = [message for message in caplog.messages if "soft_loss=" in message]
    assert len(messages) == 1
    match = re.search(r"soft_loss=([0-9.]+) hard_loss=([0-9.]+)", messages[0])
    assert match is not None

    # AND the hard loss is the loss of the weight that the layer keeps after the call
    losses = []
    for (args, kwargs), target in zip(
        quantized_inputs,
        [torch.nn.functional.linear(args[0], original_weight) for args, _ in original_inputs],
    ):
        with ff.strict_quantization(False), torch.no_grad():
            output = cast(torch.Tensor, module(*args, **kwargs))
        error = (output - target).reshape(target.shape[0], -1)
        losses.append(error.pow(2).sum(dim=1).mean().item())
    assert float(match.group(2)) == pytest.approx(sum(losses) / len(losses), rel=1e-3)


def test_adaround_runs_with_the_gradient_mode_enabled() -> None:
    # GIVEN a layer whose weight requires a gradient, which is the default state
    module = _quantized_linear(ff.granularity.PerChannel(channel_dim=0))
    original_weight = module.weight.data.clone()
    assert module.weight.requires_grad

    # WHEN we run AdaRound without `torch.no_grad`, thus the write-back of the weight is an
    # in-place operation on a leaf that requires a gradient
    assert torch.is_grad_enabled()
    adaround(module, *_symmetric_batches(), num_iterations=20)

    # THEN the weight holds the AdaRound result and its gradient flag is unchanged
    assert not torch.allclose(module.weight.data, original_weight)
    assert module.weight.requires_grad
    assert torch.is_grad_enabled()


def test_adaround_uses_the_given_range_estimator() -> None:
    # GIVEN two layers, one for the default min-max grid and one for the MSE grid
    default_module = _quantized_linear(ff.granularity.PerChannel(channel_dim=0), num_bits=3)
    mse_module = _quantized_linear(ff.granularity.PerChannel(channel_dim=0), num_bits=3)

    # WHEN we run AdaRound with the default estimator and with the MSE grid estimator
    with torch.no_grad():
        adaround(default_module, *_symmetric_batches(), num_iterations=20)
        adaround(
            mse_module,
            *_symmetric_batches(),
            num_iterations=20,
            range_estimator=ff.range_setting.mse_grid,
        )

    # THEN the two runs work on a different grid, thus range_estimator takes effect
    default_scale = _weight_quantizer(default_module).scale
    mse_scale = _weight_quantizer(mse_module).scale
    assert not torch.allclose(default_scale, mse_scale)


def test_adaround_keeps_the_existing_grid_for_a_none_range_estimator() -> None:
    # GIVEN a layer whose quantizer is calibrated on data that is not its own weight, thus its
    # grid differs from the grid that AdaRound would estimate
    module = _quantized_linear(ff.granularity.PerChannel(channel_dim=0), num_bits=3)
    _calibrate(_weight_quantizer(module), 2.0 * module.weight.data)
    scale = _weight_quantizer(module).scale.clone()

    # WHEN we run AdaRound without a range estimator
    with torch.no_grad():
        adaround(module, *_symmetric_batches(), num_iterations=20, range_estimator=None)

    # THEN the grid is untouched and the weight lands on that grid
    torch.testing.assert_close(_weight_quantizer(module).scale, scale)
    with ff.strict_quantization(False), torch.no_grad():
        requantized = _weight_quantizer(module)(module.weight).dequantize()
    torch.testing.assert_close(requantized, module.weight.data)


def test_adaround_rejects_a_none_range_estimator_without_a_grid() -> None:
    # GIVEN a layer whose quantizer has no grid yet
    module = _quantized_linear(ff.granularity.PerChannel(channel_dim=0))
    assert _weight_quantizer(module).has_uninitialized_params

    # WHEN we run AdaRound without a range estimator, THEN it reports the missing grid
    with pytest.raises(ValueError, match="uninitialized"):
        adaround(module, *_symmetric_batches(), num_iterations=1, range_estimator=None)


def test_adaround_raises_under_inference_mode() -> None:
    # GIVEN a layer and a caller that uses inference mode
    module = _quantized_linear(ff.granularity.PerChannel(channel_dim=0))

    # WHEN we run AdaRound, THEN it reports that inference mode cannot be used
    with torch.inference_mode(), pytest.raises(RuntimeError, match="inference_mode"):
        adaround(module, *_symmetric_batches(), num_iterations=1)


def test_adaround_rejects_a_non_linear_weight_quantizer() -> None:
    # GIVEN a layer whose weight quantizer is still a stub
    torch.manual_seed(0)
    module = ff.nn.QuantizedLinear(64, 32, bias=False)

    # WHEN we run AdaRound, THEN it reports the unsupported quantizer
    with pytest.raises(ValueError, match="LinearQuantizer"):
        adaround(module, *_symmetric_batches(), num_iterations=1)


def test_adaround_rejects_an_out_of_range_warm_start() -> None:
    # GIVEN a layer and a warm start that covers the whole schedule
    module = _quantized_linear(ff.granularity.PerChannel(channel_dim=0))

    # WHEN we run AdaRound, THEN it reports the invalid warm start
    with pytest.raises(ValueError, match="warm_start"):
        adaround(module, *_symmetric_batches(), num_iterations=1, warm_start=1.0)


def test_beta_schedule_runs_from_the_start_to_the_end_of_its_range() -> None:
    # GIVEN a cosine schedule from 20 to 2 with a warm-up over the first fifth
    num_iterations, warm_iterations = 1000, 200
    beta_range = (20.0, 2.0)

    # WHEN we evaluate it at the end of the warm-up, halfway and at the last iteration
    first = _beta(warm_iterations, num_iterations, beta_range, warm_iterations)
    middle = _beta(600, num_iterations, beta_range, warm_iterations)
    last = _beta(num_iterations, num_iterations, beta_range, warm_iterations)

    # THEN it starts at 20, passes through the midpoint and ends at 2
    assert first == pytest.approx(20.0)
    assert middle == pytest.approx(11.0)
    assert last == pytest.approx(2.0)
