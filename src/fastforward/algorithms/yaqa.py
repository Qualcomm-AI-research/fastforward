# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""YAQA: weight rounding against a Kronecker-factored Hessian of the model-level loss.

Introduced in https://arxiv.org/abs/2505.22988. GPTQ rounds a layer against the Hessian of the
layer output, `E[x x^T]`. YAQA rounds it against the Hessian of the model-level loss, which also
needs the gradient `g` of the loss with respect to the layer output. For a weight `W` of shape
`(m, n)` the Hessian is approximated as `H_out (x) H_in`, with one factor per side of `W`.

The factors are estimated with Sketch B of the paper: one round of power iteration, started from
the identity, on the per-sequence weight gradient `G = sum_t g_t^T x_t`. It needs one pass over
the data. Sketch A of the paper treats tokens as independent and needs several passes. It is not
implemented. There is no incoherence processing either.

The factors are the result of a `FlowOp` that runs `calculate_kronecker_factors`. It computes the
sums behind the factors on one chunk of the calibration data, and `combine_kronecker_factors`
joins the sums of the chunks, so the data need not fit in memory. `finalize_kronecker_factors`
turns the sums into the factors.
"""

from __future__ import annotations

import logging

from typing import Any, Iterable, Iterator

import torch

import fastforward as ff

logger = logging.getLogger(__name__)

# Factor by which the dampening grows after a failed Cholesky factorization.
_DAMP_GROWTH = 10.0

_Batches = Iterable[tuple[tuple[Any, ...], dict[str, Any]]]


@torch.no_grad()
def yaqa(
    module: ff.nn.QuantizedLinear,
    factors: tuple[torch.Tensor, torch.Tensor],
    perc_damp: float = 0.1,
    max_perc_damp: float = 1.0,
    layer_name: str = "",
) -> None:
    """Quantize a QuantizedLinear layer in-place using YAQA.

    The quantization ranges are set once on the original weight and stay fixed. The weight is then
    rounded onto this grid against the Kronecker-factored Hessian of the model-level loss. The
    Hessian is estimated with Sketch B, see `calculate_kronecker_factors`.

    Args:
        module: A QuantizedLinear layer whose weight_quantizer is a LinearQuantizer.
        factors: The factors `(H_out, H_in)` of the layer, of shapes `(m, m)` and `(n, n)`.
            `finalize_kronecker_factors` makes them from the result of
            `calculate_kronecker_factors`. They are factored in their own dtype, and the
            rounding runs in float32.
        perc_damp: First dampening factor to try, as fraction of the mean diagonal value, for each
            factor. If the Cholesky factorization of a factor fails, the dampening of that factor
            grows tenfold until it succeeds or reaches `max_perc_damp`. The default gave the lowest perplexity of 0.01, 0.1 and 1.0 for W4 Llama-3.2-1B.
            The value of the paper (1e-4) assumes incoherence processing, which is not part of
            this implementation. Without it, the small value gives large feedback terms that move
            the weights outside the quantization grid.
        max_perc_damp: Largest dampening factor to try.
        layer_name: Optional name for logging and error messages.

    Raises:
        ValueError: If the weight quantizer is not a LinearQuantizer, if the shapes of the
            factors do not match the weight, or if a factor has non-finite entries or is not
            positive definite at `max_perc_damp`.
    """
    weight_quantizer = module.weight_quantizer
    if not isinstance(weight_quantizer, ff.nn.LinearQuantizer):
        msg = f"weight_quantizer must be a LinearQuantizer, got {type(weight_quantizer).__name__}."
        raise ValueError(msg)

    weights = module.weight.data.clone().float()
    out_features, in_features = weights.shape

    with ff.strict_quantization(False):
        with ff.estimate_ranges(weight_quantizer, ff.range_setting.smoothed_minmax):
            weight_quantizer(weights)
        nearest = weight_quantizer(weights).dequantize()

    h_out, h_in = (factor.to(weights.device) for factor in factors)
    if h_out.shape[0] != out_features or h_in.shape[0] != in_features:
        msg = (
            f"The factors are for {h_out.shape[0]} output and {h_in.shape[0]} input features, "
            f"but the weight of layer '{layer_name}' has shape {tuple(weights.shape)}."
        )
        raise ValueError(msg)

    l_out, damp_out = unit_upper_ldl(
        h_out, perc_damp, max_perc_damp, name=f"output factor of layer '{layer_name}'"
    )
    l_in, damp_in = unit_upper_ldl(
        h_in, perc_damp, max_perc_damp, name=f"input factor of layer '{layer_name}'"
    )
    l_out, l_in = l_out.to(weights.dtype), l_in.to(weights.dtype)
    corrected, iterations = _round(weights, weight_quantizer, l_out, l_in, module.weight.dtype)
    if iterations is None:
        logger.warning(
            "[YAQA][%s] The rounding did not reach a fixed point in %d iterations",
            layer_name,
            out_features + in_features - 1,
        )

    # The weight quantizer rounds the corrected weight when the module runs. The grid stays
    # fixed, and `_round` rounds in the dtype of the weight, so this gives the rounding of the
    # fixed point.
    module.weight.copy_(corrected.to(module.weight.dtype))
    with ff.strict_quantization(False):
        rounded = weight_quantizer(module.weight).dequantize().float()

    # Rounding the feedback target moves it by at most half a step. A larger move means that
    # the target lies outside the grid, and the rounding clamps it. Steps differ per row, so
    # the largest step makes these numbers a lower bound.
    error = weights - rounded
    feedback = _feedback(error, l_out, l_in)
    step = weight_quantizer.scale.detach().max()
    beyond_grid = ((weights + feedback - rounded).abs() > 0.5 * step * 1.001).float().mean()

    logger.info(
        "[YAQA][wbits=%d][%s] proxy_loss=%.6f nearest_proxy_loss=%.6f",
        weight_quantizer.num_bits,
        layer_name,
        _proxy_loss(error, h_out, h_in),
        _proxy_loss(weights - nearest, h_out, h_in),
    )
    logger.info(
        "[YAQA][%s] iterations=%s max_feedback_steps=%.3f beyond_grid=%.3e max_l_out=%.3g "
        "max_l_in=%.3g perc_damp_out=%g perc_damp_in=%g",
        layer_name,
        iterations,
        (feedback.abs().max() / step).item(),
        beyond_grid.item(),
        l_out.abs().max().item(),
        l_in.abs().max().item(),
        damp_out,
        damp_in,
    )


@torch.no_grad()
def calculate_kronecker_factors(
    inputs: _Batches,
    gradients: _Batches,
    *,
    compute_dtype: torch.dtype = torch.float32,
    accumulate_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return the sums behind the Kronecker factors of a linear layer for its weight gradients.

    This is the function of the `FlowOp` that YAQA registers. Sketch B is one round of power
    iteration on the Hessian, started from the identity. It uses the weight gradient of each
    whole sequence. It does not treat the tokens of a sequence as independent, as Sketch A does.

    For every sequence `s` the weight gradient is `G_s = sum_t g_{s,t}^T x_{s,t}`, with shape
    `(m, n)`. The factors are `H_in = sum_s G_s^T G_s / (N m)` and `H_out = sum_s G_s G_s^T / (N n)`,
    with `N` the number of sequences. This function returns the two sums and `N`, and
    `finalize_kronecker_factors` does the division. So a chunked run can call the function once
    for each chunk and join the results with `combine_kronecker_factors`. For the factors to
    be correct, `gradients` must be the gradient of a loss that is a sum over sequences. Then
    every sequence contributes its own weight gradient.

    The weight gradient and its products are computed in `compute_dtype`, and added up in
    `accumulate_dtype`, on the device of the activations. TF32 matmuls are left to the caller's
    global setting. A recipe sets the dtypes with `functools.partial`.

    Args:
        inputs: Layer input activations. Each entry is a tuple of args (tuple of obj) and kwargs
            (dict). The single arg has shape `(batch, ..., n)`.
        gradients: Gradients of the loss with respect to the layer output, in the same layout as
            `inputs` and aligned with it batch by batch. The single arg has shape
            `(batch, ..., m)`.
        compute_dtype: The dtype of the weight gradient of each sequence and of its products.
        accumulate_dtype: The dtype of the sums. A dtype with more precision than
            `compute_dtype` adds each product to the sums in that dtype, as the paper does
            with float64.

    Returns:
        The sums `(H_out, H_in)` of `G_s G_s^T` and `G_s^T G_s`, of shapes `(m, m)` and
        `(n, n)`, and the number of sequences they add up.

    Raises:
        ValueError: If there are no batches, if the number of batches of `inputs` and
            `gradients` differs, if a gradient is `None`, or if the shapes of an input and its
            gradient do not match.
    """
    h_out: torch.Tensor | None = None
    h_in: torch.Tensor | None = None
    num_sequences = 0
    for ((x,), _), ((g,), _) in zip(inputs, gradients, strict=True):
        if g is None:
            msg = "The output gradient is None: the objective does not depend on this layer."
            raise ValueError(msg)
        if x.ndim < 2 or x.shape[:-1] != g.shape[:-1]:
            msg = (
                f"Input shape {tuple(x.shape)} and gradient shape {tuple(g.shape)} must have "
                "at least two dimensions and be equal in all but the last dimension."
            )
            raise ValueError(msg)

        # `(batch, ..., features)` -> `(batch, tokens, features)`
        x = x.to(dtype=compute_dtype).reshape(x.shape[0], -1, x.shape[-1])
        g = g.to(device=x.device, dtype=compute_dtype).reshape(g.shape[0], -1, g.shape[-1])

        if h_out is None or h_in is None:
            h_out = torch.zeros((g.shape[-1],) * 2, device=x.device, dtype=accumulate_dtype)
            h_in = torch.zeros((x.shape[-1],) * 2, device=x.device, dtype=accumulate_dtype)

        # One sequence at a time keeps the memory at `m x n`.
        for x_seq, g_seq in zip(x, g):
            weight_grad = g_seq.transpose(0, 1) @ x_seq
            _add_upper_gram(h_in, weight_grad)
            _add_upper_gram(h_out, weight_grad.transpose(0, 1))
            num_sequences += 1

    if h_out is None or h_in is None:
        msg = "Cannot estimate the Kronecker factors without any sequences."
        raise ValueError(msg)

    return _mirror_upper(h_out), _mirror_upper(h_in), num_sequences


_GRAM_BLOCK_SIZE = 512


def _add_upper_gram(total: torch.Tensor, matrix: torch.Tensor) -> None:
    """Add the upper triangle of `matrix^T @ matrix` to `total`, in blocks of rows.

    The product is symmetric, so the rows of each block only need the columns from the start
    of the block on. That skips most of the lower triangle, almost half of the work for many
    blocks. The parts of `total` below the diagonal are not valid; `_mirror_upper` fills them.

    The product is computed in the dtype of `matrix` and added in the dtype of `total`.
    """
    size = matrix.shape[1]
    for start in range(0, size, _GRAM_BLOCK_SIZE):
        end = min(start + _GRAM_BLOCK_SIZE, size)
        rows, columns = matrix[:, start:end].transpose(0, 1), matrix[:, start:]
        if total.dtype == matrix.dtype:
            total[start:end, start:].addmm_(rows, columns)
        else:
            # `addmm_` needs one dtype, so the block of the product is a temporary tensor.
            total[start:end, start:].add_(rows @ columns)


def _mirror_upper(total: torch.Tensor) -> torch.Tensor:
    """Return the symmetric matrix with the upper triangle of `total`."""
    return total.triu() + total.triu(1).transpose(0, 1)


def combine_kronecker_factors(
    previous: tuple[torch.Tensor, torch.Tensor, int],
    new: tuple[torch.Tensor, torch.Tensor, int],
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Join two results of `calculate_kronecker_factors` into the result on all of their sequences.

    This is the `combine` of the `FlowOp` that YAQA registers.

    Args:
        previous: The sums `(H_out, H_in)` and their number of sequences. The sums are changed
            in place.
        new: The sums `(H_out, H_in)` and their number of sequences.

    Returns:
        `previous`, which now holds the sums and the number of sequences of both.
    """
    previous[0].add_(new[0].to(previous[0].device))
    previous[1].add_(new[1].to(previous[1].device))
    return previous[0], previous[1], previous[2] + new[2]


def finalize_kronecker_factors(
    state: tuple[torch.Tensor, torch.Tensor, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the factors `(H_out, H_in)` of the state that `calculate_kronecker_factors` made.

    This is the `finalize` of the `FlowOp` that YAQA registers.

    Args:
        state: The sums `(H_out, H_in)` and the number of sequences that they add up.

    Returns:
        The factors of shapes `(m, m)` and `(n, n)`. They are new tensors, so the state stays as
        it is for the other layers that read it.
    """
    h_out, h_in, num_sequences = state
    out_features, in_features = h_out.shape[0], h_in.shape[0]
    return h_out / (num_sequences * in_features), h_in / (num_sequences * out_features)


def unit_upper_ldl(
    hessian: torch.Tensor, perc_damp: float, max_perc_damp: float, *, name: str = "matrix"
) -> tuple[torch.Tensor, float]:
    """Factor a damped symmetric positive definite matrix as `L D L^T` with `L` unit upper.

    The matrix is damped as `H + damp * mean(diag(H)) * I`. The factorization starts at
    `damp = perc_damp`. If it fails, `damp` grows by a factor of `_DAMP_GROWTH` (capped at
    `max_perc_damp`) and the original matrix is damped and factored again. Only `L` is returned,
    because only `L` is needed for rounding.

    Args:
        hessian: Symmetric matrix of shape `(n, n)`.
        perc_damp: First dampening factor, as fraction of the mean diagonal value.
        max_perc_damp: Largest dampening factor to try.
        name: What the matrix is, for log and error messages.

    Returns:
        The unit upper triangular factor `L` of shape `(n, n)` and the dampening factor that
        made the factorization succeed.

    Raises:
        ValueError: If `perc_damp` is not in `(0, max_perc_damp]`, if the matrix has non-finite
            entries, or if it is not positive definite at `max_perc_damp`.
    """
    if not 0 < perc_damp <= max_perc_damp:
        msg = f"Expected 0 < perc_damp <= max_perc_damp, got {perc_damp} and {max_perc_damp}."
        raise ValueError(msg)
    if not hessian.isfinite().all():
        msg = f"The {name} has non-finite entries. More dampening cannot fix this."
        raise ValueError(msg)

    mean_diagonal = hessian.diagonal().mean()
    identity = torch.eye(hessian.shape[0], device=hessian.device, dtype=hessian.dtype)
    damp = perc_damp
    while True:
        damped = hessian + damp * mean_diagonal * identity
        # An upper factor `R` with `H = R R^T` is the flipped Cholesky factor of the flipped matrix.
        chol, info = torch.linalg.cholesky_ex(damped.flip((0, 1)))
        if info.item() == 0:
            break
        if damp >= max_perc_damp:
            msg = (
                f"The {name} is not positive definite after dampening with perc_damp={damp} (the "
                f"Cholesky factorization failed at order {info.item()})."
            )
            raise ValueError(msg)
        next_damp = min(damp * _DAMP_GROWTH, max_perc_damp)
        logger.warning(
            "[YAQA] Cholesky of the %s failed with perc_damp=%g, retrying with %g",
            name,
            damp,
            next_damp,
        )
        damp = next_damp
    upper: torch.Tensor = chol.flip((0, 1))

    # Divide each column by its diagonal entry.
    return upper / upper.diagonal(), damp


def _proxy_loss(delta: torch.Tensor, h_out: torch.Tensor, h_in: torch.Tensor) -> float:
    """Return `tr(delta H_in delta^T H_out)`, the loss that YAQA minimizes for a weight error."""
    delta = delta.to(h_out.dtype)
    return ((h_out @ delta) * (delta @ h_in)).sum().item()


def _round(
    weights: torch.Tensor,
    quantizer: ff.nn.LinearQuantizer,
    l_out: torch.Tensor,
    l_in: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, int | None]:
    """Correct `weights` with error feedback from both Kronecker factors, for rounding.

    Solves the fixed point `W_hat = Q(W + L_out^T (W - W_hat) L_in - (W - W_hat))` (Algorithm 1 of
    the paper), starting from `W_hat = Q(W)`. With `L_out = I` this is LDLQ, which is GPTQ without
    act-order. Entry `(i, j)` of the right side depends only on entries `(i', j')` with
    `i' <= i`, `j' <= j` and `(i', j') != (i, j)`, so `m + n - 1` iterations are exact.

    Args:
        weights: Weight of shape `(m, n)` in float32.
        quantizer: Weight quantizer with ranges calibrated on `weights`. Its scale and offset
            stay fixed.
        l_out: Unit upper factor of `H_out`, shape `(m, m)`.
        l_in: Unit upper factor of `H_in`, shape `(n, n)`.
        dtype: The dtype in which the corrected weight is stored. `Q` casts to it before it
            rounds, as the module does when it runs. Without the cast, the stored weight of an
            entry near a rounding boundary can round to the next grid point.

    Returns:
        The corrected weight `W + L_out^T (W - W_hat) L_in - (W - W_hat)` of shape `(m, n)`, so
        that `Q(corrected) = W_hat` also after the cast to `dtype`, and the number of iterations
        until the fixed point. The number is `None` if the iteration did not reach the fixed
        point. The corrected weight is not rounded, and may lie outside the grid, where the
        quantizer clamps it.
    """

    def quantize(w: torch.Tensor) -> torch.Tensor:
        return quantizer(w.to(dtype)).dequantize().to(w.dtype)

    with ff.strict_quantization(False):
        rounded = quantize(weights)
        corrected = weights
        for iterations in range(1, sum(weights.shape)):
            corrected = weights + _feedback(weights - rounded, l_out, l_in)
            updated = quantize(corrected)
            if torch.equal(updated, rounded):
                return corrected, iterations
            rounded = updated
    return corrected, None


def _feedback(error: torch.Tensor, l_out: torch.Tensor, l_in: torch.Tensor) -> torch.Tensor:
    """Return `L_out^T E L_in - E` for unit upper `L_out` and `L_in`.

    The terms are split as `(L_out^T - I) E L_in + E (L_in - I)`, so that entry `(i, j)` does not
    depend on `E[i, j]`. The direct form subtracts `E` from a sum that contains `E`, and the
    rounding error of float32 leaves a part of `E[i, j]` in entry `(i, j)`. With large feedback
    this part can move the target to the next grid point, and the iteration need not converge.
    """
    strict_out = l_out.transpose(0, 1) - torch.eye(l_out.shape[0], device=l_out.device)
    strict_in = l_in - torch.eye(l_in.shape[0], device=l_in.device)
    return strict_out @ error @ l_in + error @ strict_in


def sampled_nll(
    logits: _Batches, dtype: torch.dtype | None = torch.float32
) -> Iterator[torch.Tensor]:
    """Yield the negative log-likelihood of labels sampled from the model, one scalar per batch.

    This is the default objective of `yaqa`. A label is sampled for every token from the softmax
    of the logits, so the gradients of the loss estimate the Fisher information of the model and
    not the Hessian of a loss on labeled data. The loss is a sum over tokens, not a mean, so
    every sequence keeps its own gradient scale.

    The loss is computed on the device of the logits, and the labels are sampled with the global
    random number generator of torch. A model that returns an object, like a Hugging Face model,
    needs a `select` on the `ModelOutputs` that reads the logits tensor from the object. It can
    also move the logits to the device where the loss is to be computed.

    Args:
        logits: The logits of the model, one batch per entry. The single arg has shape
            `(batch, ..., classes)`.
        dtype: The dtype in which the loss is computed. The default float32 keeps the softmax
            of low-precision logits accurate. `None` keeps the dtype of the logits and saves the
            memory of a copy.

    Yields:
        The summed negative log-likelihood of the sampled labels of a batch.
    """
    for (batch_logits,), _ in logits:
        flat = batch_logits.reshape(-1, batch_logits.shape[-1])
        if dtype is not None:
            flat = flat.to(dtype)
        with torch.no_grad():
            labels = torch.multinomial(flat.softmax(dim=-1), num_samples=1).squeeze(-1)
        yield torch.nn.functional.cross_entropy(flat, labels, reduction="sum")
