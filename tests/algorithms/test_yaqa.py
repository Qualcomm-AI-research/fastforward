# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import importlib

from typing import Any

import fastforward as ff
import pytest
import torch

from fastforward.algorithms.gptq import gptq
from fastforward.algorithms.hessian_calculation import calculate_hessian
from fastforward.algorithms.yaqa import (
    _feedback,
    _round,
    calculate_kronecker_factors,
    combine_kronecker_factors,
    finalize_kronecker_factors,
    sampled_nll,
    unit_upper_ldl,
    yaqa,
)

Batches = list[tuple[tuple[Any, ...], dict[str, Any]]]


def _calibrated_quantizer(
    granularity: ff.granularity.Granularity, weights: torch.Tensor
) -> ff.nn.LinearQuantizer:
    """Create a 4-bit quantizer with ranges calibrated on `weights`."""
    quantizer = ff.nn.LinearQuantizer(num_bits=4, granularity=granularity, symmetric=False)
    with (
        ff.strict_quantization(False),
        ff.estimate_ranges(quantizer, ff.range_setting.smoothed_minmax),
    ):
        quantizer(weights)
    return quantizer


def _bundle(*tensors: torch.Tensor | None) -> Batches:
    return [((tensor,), {}) for tensor in tensors]


def _factors(inputs: Batches, gradients: Batches) -> tuple[torch.Tensor, torch.Tensor]:
    """The factors that YAQA gets from the `FlowOp` it registers, for one run on all the data."""
    return finalize_kronecker_factors(calculate_kronecker_factors(inputs, gradients))


def _correlated_hessian(size: int) -> torch.Tensor:
    """Symmetric positive definite matrix with a wide, geometrically decaying spectrum."""
    basis: torch.Tensor = torch.linalg.qr(torch.randn(size, size, dtype=torch.float64)).Q
    eigenvalues = torch.logspace(0, -3, size, dtype=torch.float64)
    return (basis * eigenvalues) @ basis.transpose(0, 1)


@pytest.mark.parametrize("perc_damp", [1e-6, 0.01])
def test_unit_upper_ldl_is_unit_upper_and_rebuilds_the_damped_hessian(perc_damp: float) -> None:
    # GIVEN a symmetric positive definite matrix
    torch.manual_seed(0)
    hessian = _correlated_hessian(12)
    damped = hessian + perc_damp * hessian.diagonal().mean() * torch.eye(12, dtype=torch.float64)

    # WHEN we factor it
    factor, damp = unit_upper_ldl(hessian, perc_damp, max_perc_damp=1.0)
    assert damp == perc_damp

    # THEN the factor is unit upper triangular
    torch.testing.assert_close(factor, torch.triu(factor))
    torch.testing.assert_close(factor.diagonal(), torch.ones(12, dtype=torch.float64))

    # THEN `L^{-1} H L^{-T}` is diagonal with positive entries, so `H = L D L^T`
    left = torch.linalg.solve_triangular(factor, damped, upper=True, unitriangular=True)
    diagonal = torch.linalg.solve_triangular(
        factor, left.transpose(0, 1), upper=True, unitriangular=True
    )
    diagonal = diagonal.transpose(0, 1)
    torch.testing.assert_close(diagonal, torch.diag(diagonal.diagonal()), atol=1e-10, rtol=0)
    assert (diagonal.diagonal() > 0).all()


def test_unit_upper_ldl_retries_with_more_damping_until_the_matrix_is_positive_definite() -> None:
    # GIVEN a rank-deficient matrix that needs more damping than the first value
    torch.manual_seed(0)
    vectors = torch.randn(12, 3, dtype=torch.float64)
    hessian = vectors @ vectors.transpose(0, 1)

    # WHEN we factor it with a first damping that is too small
    factor, damp = unit_upper_ldl(hessian, perc_damp=1e-20, max_perc_damp=1.0)

    # THEN the damping grew, and the factor is that of the matrix damped with the final value
    assert 1e-20 < damp <= 1.0
    damped = hessian + damp * hessian.diagonal().mean() * torch.eye(12, dtype=torch.float64)
    rebuilt = factor @ torch.diag(
        torch.linalg.cholesky(damped.flip((0, 1))).diagonal().flip(0) ** 2
    )
    torch.testing.assert_close(rebuilt @ factor.transpose(0, 1), damped)


def test_unit_upper_ldl_does_not_change_its_input() -> None:
    # GIVEN a matrix that needs a retry
    hessian = torch.ones(4, 4, dtype=torch.float64)
    original = hessian.clone()

    # WHEN we factor it
    unit_upper_ldl(hessian, perc_damp=1e-20, max_perc_damp=1.0)

    # THEN the matrix is the same
    torch.testing.assert_close(hessian, original, atol=0, rtol=0)


def test_unit_upper_ldl_rejects_a_matrix_that_is_not_positive_definite_at_max_perc_damp() -> None:
    # GIVEN a negative definite matrix
    hessian = -torch.eye(4)

    # WHEN / THEN factoring it fails after the largest damping
    with pytest.raises(ValueError, match="not positive definite after dampening with perc_damp=1"):
        unit_upper_ldl(hessian, perc_damp=0.01, max_perc_damp=1.0)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_unit_upper_ldl_rejects_non_finite_entries_without_retrying(
    value: float, caplog: pytest.LogCaptureFixture
) -> None:
    # GIVEN a matrix with a non-finite entry
    hessian = torch.eye(4)
    hessian[1, 2] = value

    # WHEN / THEN factoring it fails at once
    with pytest.raises(ValueError, match="non-finite"), caplog.at_level("WARNING"):
        unit_upper_ldl(hessian, perc_damp=0.01, max_perc_damp=1.0)
    assert not caplog.records


@pytest.mark.parametrize("perc_damp, max_perc_damp", [(0.0, 1.0), (-1.0, 1.0), (2.0, 1.0)])
def test_unit_upper_ldl_rejects_an_invalid_damping_range(
    perc_damp: float, max_perc_damp: float
) -> None:
    with pytest.raises(ValueError, match="perc_damp <= max_perc_damp"):
        unit_upper_ldl(torch.eye(4), perc_damp, max_perc_damp)


def test_calculate_kronecker_factors_match_per_sequence_autograd() -> None:
    # GIVEN a linear layer, a loss that is a sum over sequences, and batches with extra
    # token dimensions
    torch.manual_seed(0)
    out_features, in_features = 5, 7
    weight = torch.randn(out_features, in_features, requires_grad=True)
    batches = [torch.randn(2, 3, 4, in_features) for _ in range(3)]

    inputs: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    h_out_ref = torch.zeros(out_features, out_features)
    h_in_ref = torch.zeros(in_features, in_features)
    num_sequences = 0
    for x in batches:
        y = x @ weight.transpose(0, 1)
        (g,) = torch.autograd.grad(y.tanh().sum(), y)
        inputs.append(x)
        gradients.append(g)
        for sequence in x:
            (weight_grad,) = torch.autograd.grad(
                (sequence @ weight.transpose(0, 1)).tanh().sum(), weight
            )
            h_out_ref += weight_grad @ weight_grad.transpose(0, 1) / in_features
            h_in_ref += weight_grad.transpose(0, 1) @ weight_grad / out_features
            num_sequences += 1

    # WHEN we estimate the factors from the activations and the output gradients
    h_out, h_in = _factors(_bundle(*inputs), _bundle(*gradients))

    # THEN they equal the factors built from the per-sequence weight gradients
    torch.testing.assert_close(h_out, h_out_ref / num_sequences)
    torch.testing.assert_close(h_in, h_in_ref / num_sequences)


@pytest.mark.parametrize(
    ("compute_dtype", "accumulate_dtype"),
    [
        (torch.float32, torch.float32),
        (torch.float32, torch.float64),
        (torch.float64, torch.float64),
    ],
)
def test_calculate_kronecker_factors_in_blocks_match_the_full_products(
    monkeypatch: pytest.MonkeyPatch, compute_dtype: torch.dtype, accumulate_dtype: torch.dtype
) -> None:
    # GIVEN activations and gradients with more features than one block
    torch.manual_seed(0)
    monkeypatch.setattr(
        importlib.import_module(calculate_kronecker_factors.__module__), "_GRAM_BLOCK_SIZE", 3
    )
    inputs = [torch.randn(2, 4, 7), torch.randn(1, 4, 7)]
    gradients = [torch.randn(2, 4, 8), torch.randn(1, 4, 8)]

    # WHEN we compute the sums in blocks, in the given dtypes
    h_out, h_in, num_sequences = calculate_kronecker_factors(
        _bundle(*inputs),
        _bundle(*gradients),
        compute_dtype=compute_dtype,
        accumulate_dtype=accumulate_dtype,
    )

    # THEN they have the accumulate dtype, are symmetric, and equal the float64 sums of the
    # full products up to the precision of float32
    weight_grads = [
        g.double().transpose(0, 1) @ x.double()
        for x, g in zip(torch.cat(inputs), torch.cat(gradients))
    ]
    assert h_out.dtype == h_in.dtype == accumulate_dtype
    assert num_sequences == 3
    torch.testing.assert_close(h_out, h_out.transpose(0, 1), atol=0, rtol=0)
    torch.testing.assert_close(h_in, h_in.transpose(0, 1), atol=0, rtol=0)
    h_out_ref = sum(w @ w.transpose(0, 1) for w in weight_grads)
    h_in_ref = sum(w.transpose(0, 1) @ w for w in weight_grads)
    torch.testing.assert_close(h_out.double(), h_out_ref, atol=1e-5, rtol=1.3e-6)
    torch.testing.assert_close(h_in.double(), h_in_ref, atol=1e-5, rtol=1.3e-6)


def test_combine_kronecker_factors_joins_the_sums_of_two_chunks() -> None:
    # GIVEN activations and gradients in two chunks, with different numbers of sequences
    torch.manual_seed(0)
    inputs = [torch.randn(2, 3, 7), torch.randn(3, 4, 7), torch.randn(1, 3, 7)]
    gradients = [torch.randn(2, 3, 5), torch.randn(3, 4, 5), torch.randn(1, 3, 5)]

    # WHEN we compute the sums on each chunk and join them, and on all data in one call
    first = calculate_kronecker_factors(_bundle(*inputs[:2]), _bundle(*gradients[:2]))
    second = calculate_kronecker_factors(_bundle(*inputs[2:]), _bundle(*gradients[2:]))
    chunked = combine_kronecker_factors(first, second)
    whole = calculate_kronecker_factors(_bundle(*inputs), _bundle(*gradients))

    # THEN both ways give the same sums over the same six sequences
    assert chunked[2] == whole[2] == 6
    torch.testing.assert_close(chunked[0], whole[0])
    torch.testing.assert_close(chunked[1], whole[1])


def test_finalize_kronecker_factors_leaves_the_state_as_it_was() -> None:
    # GIVEN the state for some sequences
    torch.manual_seed(0)
    state = calculate_kronecker_factors(
        _bundle(torch.randn(3, 4, 7)), _bundle(torch.randn(3, 4, 5))
    )
    before = (state[0].clone(), state[1].clone())

    # WHEN we finalize it twice, as two layers that read the state do
    first = finalize_kronecker_factors(state)
    second = finalize_kronecker_factors(state)

    # THEN the state is as it was, and both give the same factors
    torch.testing.assert_close(state[0], before[0], rtol=0, atol=0)
    torch.testing.assert_close(state[1], before[1], rtol=0, atol=0)
    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
    torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)


@pytest.mark.parametrize(
    ("inputs", "gradients", "match"),
    [
        (_bundle(torch.ones(2, 3, 4)), _bundle(None), "None"),
        (_bundle(torch.ones(2, 3, 4)), _bundle(torch.ones(2, 5, 6)), "shape"),
        (_bundle(torch.ones(4)), _bundle(torch.ones(6)), "shape"),
        (_bundle(torch.ones(2, 3, 4)), [], "shorter"),
        ([], [], "without any sequences"),
    ],
    ids=["none_gradient", "token_mismatch", "one_dimension", "batch_count", "empty"],
)
def test_calculate_kronecker_factors_rejects_invalid_data(
    inputs: Batches, gradients: Batches, match: str
) -> None:
    # GIVEN activations and gradients that cannot be paired
    # WHEN / THEN estimating the factors fails with an informative error
    with pytest.raises(ValueError, match=match):
        calculate_kronecker_factors(inputs, gradients)


def _dequantize(quantizer: ff.nn.Quantizer, weights: torch.Tensor) -> torch.Tensor:
    """Round `weights` onto the grid of `quantizer`."""
    with ff.strict_quantization(False):
        return quantizer(weights).dequantize()


@pytest.mark.parametrize(
    "granularity",
    [
        ff.granularity.PerTensor(),
        ff.granularity.PerChannel(channel_dim=0),
        ff.granularity.PerChannel(channel_dim=1),
        ff.granularity.PerBlock(block_dims=1, block_sizes=16, per_channel_dims=0),
    ],
    ids=["per_tensor", "per_channel_dim0", "per_channel_dim1", "per_block_col16"],
)
def test_round_returns_the_corrected_weight_of_a_fixed_point(
    granularity: ff.granularity.Granularity,
) -> None:
    # GIVEN a calibrated quantizer and two unit upper factors
    torch.manual_seed(0)
    weights = torch.randn(16, 32)
    quantizer = _calibrated_quantizer(granularity, weights)
    l_out, _ = unit_upper_ldl(_correlated_hessian(16).float(), perc_damp=1e-4, max_perc_damp=1.0)
    l_in, _ = unit_upper_ldl(_correlated_hessian(32).float(), perc_damp=1e-4, max_perc_damp=1.0)

    # WHEN we correct the weights
    corrected, iterations = _round(weights, quantizer, l_out, l_in, torch.float32)

    # THEN the iteration converged, and the corrected weight is the target of its own rounding
    assert iterations is not None
    rounded = _dequantize(quantizer, corrected)
    torch.testing.assert_close(
        corrected, weights + _feedback(weights - rounded, l_out, l_in), rtol=0, atol=0
    )


def test_round_without_output_factor_rounds_to_the_gptq_weight() -> None:
    # GIVEN a layer, calibration data, and a per-channel quantizer
    torch.manual_seed(0)
    out_features, in_features = 16, 32
    weights = torch.randn(out_features, in_features)
    activations = [torch.randn(2, 8, in_features) for _ in range(4)]
    granularity = ff.granularity.PerChannel(channel_dim=0)

    module = ff.nn.QuantizedLinear(in_features, out_features, bias=False)
    with torch.no_grad():
        module.weight.copy_(weights)
    module.weight_quantizer = ff.nn.LinearQuantizer(
        num_bits=4, granularity=granularity, symmetric=False
    )

    # WHEN we run GPTQ, and round with the factor of the same Hessian and no output factor
    with torch.inference_mode(), ff.strict_quantization(False):
        hessian = calculate_hessian(_bundle(*activations))
        gptq(module, hessian, perc_damp=0.01, actorder=False)
    flat = torch.cat([a.reshape(-1, in_features) for a in activations])
    l_in, _ = unit_upper_ldl(flat.transpose(0, 1) @ flat, perc_damp=0.01, max_perc_damp=1.0)
    quantizer = _calibrated_quantizer(granularity, weights)
    corrected, _ = _round(weights, quantizer, torch.eye(out_features), l_in, torch.float32)

    # THEN the rounding of the corrected weights is the weight of GPTQ
    torch.testing.assert_close(_dequantize(quantizer, corrected), module.weight.detach())


def test_round_lowers_the_proxy_loss_below_nearest_rounding() -> None:
    # GIVEN correlated factors on both sides
    torch.manual_seed(0)
    out_features, in_features = 16, 32
    weights = torch.randn(out_features, in_features, dtype=torch.float64)
    h_out = _correlated_hessian(out_features)
    h_in = _correlated_hessian(in_features)
    quantizer = _calibrated_quantizer(ff.granularity.PerChannel(channel_dim=0), weights.float())

    def proxy_loss(rounded: torch.Tensor) -> float:
        delta = weights - rounded.double()
        return torch.trace(delta @ h_in @ delta.transpose(0, 1) @ h_out).item()

    # WHEN we round with error feedback, and with nearest rounding
    l_out, _ = unit_upper_ldl(h_out.float(), perc_damp=1e-4, max_perc_damp=1.0)
    l_in, _ = unit_upper_ldl(h_in.float(), perc_damp=1e-4, max_perc_damp=1.0)
    corrected, _ = _round(weights.float(), quantizer, l_out, l_in, torch.float32)
    rounded = _dequantize(quantizer, corrected)
    nearest = _dequantize(quantizer, weights.float())

    # THEN the proxy loss `tr(Delta H_in Delta^T H_out)` is lower
    assert proxy_loss(rounded) < proxy_loss(nearest)


def _layer_and_data(
    out_features: int = 16, in_features: int = 32
) -> tuple[ff.nn.QuantizedLinear, Batches, Batches]:
    """Layer with a per-channel quantizer, plus low-rank activations and output gradients."""
    torch.manual_seed(0)
    rank = 4
    module = ff.nn.QuantizedLinear(in_features, out_features, bias=False)
    module.weight_quantizer = ff.nn.LinearQuantizer(
        num_bits=4, granularity=ff.granularity.PerChannel(channel_dim=0), symmetric=False
    )
    mix_in, mix_out = torch.randn(rank, in_features), torch.randn(rank, out_features)
    inputs = _bundle(*(torch.randn(2, 8, rank) @ mix_in for _ in range(4)))
    gradients = _bundle(*(torch.randn(2, 8, rank) @ mix_out for _ in range(4)))
    return module, inputs, gradients


@pytest.mark.parametrize("factor_dtype", [torch.float32, torch.float64])
def test_yaqa_stores_a_corrected_weight_that_the_quantizer_rounds_to_the_grid(
    factor_dtype: torch.dtype,
) -> None:
    # GIVEN a layer with data, and factors in the given dtype
    module, inputs, gradients = _layer_and_data()
    original = module.weight.detach().clone()
    h_out, h_in = _factors(inputs, gradients)

    # WHEN we run YAQA
    with torch.inference_mode(), ff.strict_quantization(False):
        yaqa(module, (h_out.to(factor_dtype), h_in.to(factor_dtype)), layer_name="fc")

    # THEN the weight changed, and the quantizer rounds it to a point of the grid that is
    # a fixed point of the rounding, and not the nearest rounding of the original weight
    assert not torch.equal(module.weight.detach(), original)
    with torch.inference_mode():
        rounded = _dequantize(module.weight_quantizer, module.weight)
        nearest = _dequantize(module.weight_quantizer, original)
        again = _dequantize(module.weight_quantizer, rounded)
    torch.testing.assert_close(again, rounded)
    assert not torch.equal(rounded, nearest)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_yaqa_stores_a_low_precision_weight_that_rounds_to_the_fixed_point(
    dtype: torch.dtype,
) -> None:
    # GIVEN a layer in a low-precision dtype with data, and the factors of the data
    module, inputs, gradients = _layer_and_data(out_features=64, in_features=128)
    module.to(dtype)
    original = module.weight.detach().float()
    h_out, h_in = _factors(inputs, gradients)

    # WHEN we run YAQA
    with torch.inference_mode(), ff.strict_quantization(False):
        yaqa(module, (h_out, h_in))

    # THEN the rounding of the stored weight is the fixed point of the rounding in that dtype
    with torch.inference_mode():
        rounded = _dequantize(module.weight_quantizer, module.weight).float()
        l_out, _ = unit_upper_ldl(h_out, perc_damp=0.1, max_perc_damp=1.0)
        l_in, _ = unit_upper_ldl(h_in, perc_damp=0.1, max_perc_damp=1.0)
        target = original + _feedback(original - rounded, l_out, l_in)
        again = _dequantize(module.weight_quantizer, target.to(dtype)).float()
    torch.testing.assert_close(again, rounded, rtol=0, atol=0)


def test_yaqa_lowers_the_proxy_loss_below_nearest_rounding() -> None:
    # GIVEN a layer with data, and the quantized weight of nearest rounding
    module, inputs, gradients = _layer_and_data()
    original = module.weight.detach().clone()
    nearest_quantizer = _calibrated_quantizer(ff.granularity.PerChannel(channel_dim=0), original)
    with ff.strict_quantization(False):
        nearest = nearest_quantizer(original).dequantize()
    factors = _factors(inputs, gradients)
    h_out, h_in = factors

    def proxy_loss(weight: torch.Tensor) -> float:
        delta = (original - weight).double()
        return torch.trace(delta @ h_in.double() @ delta.transpose(0, 1) @ h_out.double()).item()

    # WHEN we run YAQA
    with torch.inference_mode(), ff.strict_quantization(False):
        yaqa(module, factors)

    # THEN its proxy loss `tr(Delta H_in Delta^T H_out)` is lower than for nearest rounding
    with torch.inference_mode():
        rounded = _dequantize(module.weight_quantizer, module.weight)
    assert proxy_loss(rounded) < proxy_loss(nearest)


def test_yaqa_rejects_a_weight_quantizer_that_is_not_a_linear_quantizer() -> None:
    # GIVEN a layer that still has its quantizer stub
    _, inputs, gradients = _layer_and_data()
    module = ff.nn.QuantizedLinear(32, 16, bias=False)

    # WHEN / THEN YAQA fails
    with pytest.raises(ValueError, match="LinearQuantizer"):
        yaqa(module, _factors(inputs, gradients))


def test_yaqa_rejects_data_that_does_not_match_the_weight() -> None:
    # GIVEN factors for gradients with the wrong number of features
    module, inputs, _ = _layer_and_data()
    gradients = _bundle(*(torch.randn(2, 8, 5) for _ in range(4)))

    # WHEN / THEN YAQA fails
    with pytest.raises(ValueError, match="has shape"):
        yaqa(module, _factors(inputs, gradients))


def test_yaqa_warns_when_the_rounding_does_not_reach_a_fixed_point(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # GIVEN a layer with data, and a rounding that does not reach a fixed point
    module, inputs, gradients = _layer_and_data()
    monkeypatch.setattr(
        importlib.import_module(yaqa.__module__),
        "_round",
        lambda weights, *_: (weights, None),
    )

    # WHEN we run YAQA
    with torch.inference_mode(), ff.strict_quantization(False), caplog.at_level("WARNING"):
        yaqa(module, _factors(inputs, gradients), layer_name="fc")

    # THEN it warns and names the layer
    assert "[YAQA][fc] The rounding did not reach a fixed point in 47 iterations" in caplog.text


def test_yaqa_names_the_factor_that_is_not_positive_definite() -> None:
    # GIVEN output gradients of zero, so that the output factor is zero
    module, inputs, gradients = _layer_and_data()
    zeros = _bundle(*(torch.zeros_like(g) for (g,), _ in gradients))

    # WHEN / THEN YAQA fails and says which factor
    with pytest.raises(ValueError, match="output factor of layer 'fc'"):
        yaqa(module, _factors(inputs, zeros), layer_name="fc")


def test_sampled_nll_yields_one_scalar_per_batch_with_a_sampled_label_gradient() -> None:
    # GIVEN logits for two batches with extra token dimensions
    torch.manual_seed(0)
    batches = [torch.randn(2, 3, 5, requires_grad=True) for _ in range(2)]

    # WHEN we draw the losses and differentiate each with respect to its logits
    losses = list(sampled_nll(_bundle(*batches)))
    grads = [torch.autograd.grad(loss, batch)[0] for loss, batch in zip(losses, batches)]

    # THEN there is one real scalar loss per batch
    assert len(losses) == 2
    assert all(loss.ndim == 0 and loss.is_floating_point() for loss in losses)

    # THEN each gradient is `softmax - onehot(label)` of a label per token, so
    # subtracting the gradient from the softmax recovers a one-hot encoding
    for loss, batch, grad in zip(losses, batches, grads):
        onehot = batch.detach().softmax(dim=-1) - grad
        torch.testing.assert_close(onehot, onehot.round(), atol=1e-5, rtol=0)
        torch.testing.assert_close(onehot.sum(dim=-1), torch.ones(2, 3))

        # THEN the loss is the sum of the negative log-likelihoods of these labels
        nll = -(batch.detach().log_softmax(dim=-1) * onehot.round()).sum()
        torch.testing.assert_close(loss.detach(), nll)


def test_sampled_nll_samples_from_the_model_distribution() -> None:
    # GIVEN logits that put almost all probability on one class per token
    logits = torch.full((4, 6, 3), -50.0)
    logits[..., 1] = 50.0
    logits.requires_grad_()

    # WHEN we sample labels
    (loss,) = sampled_nll(_bundle(logits))
    (grad,) = torch.autograd.grad(loss, logits)

    # THEN every sampled label is the likely class, so the loss and the gradient vanish
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    torch.testing.assert_close(grad, torch.zeros_like(grad), atol=1e-6, rtol=0)


@pytest.mark.parametrize(
    ("dtype", "expected"), [(torch.float32, torch.float32), (None, torch.bfloat16)]
)
def test_sampled_nll_computes_the_loss_in_the_given_dtype(
    dtype: torch.dtype | None, expected: torch.dtype
) -> None:
    # GIVEN bfloat16 logits
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 5, dtype=torch.bfloat16)

    # WHEN we compute the loss with a dtype, or with `None`
    (loss,) = sampled_nll(_bundle(logits), dtype=dtype)

    # THEN the loss has that dtype, or the dtype of the logits
    assert loss.dtype == expected
