# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#
# This file implements AdaRound, introduced in "Up or Down? Adaptive Rounding for
# Post-Training Quantization" (https://arxiv.org/abs/2004.10568).

from __future__ import annotations

import contextlib
import logging
import math

from typing import Any, Generator, Iterable, TypeAlias

import optree
import torch

import fastforward as ff

from fastforward.quantization import tiled_tensor
from fastforward.range_setting import RangeEstimator, smoothed_minmax

logger = logging.getLogger(__name__)

# A range estimator, as a type or as an instance, in the form that `estimate_ranges` takes.
_RangeEstimator: TypeAlias = RangeEstimator[Any, Any] | type[RangeEstimator[Any, Any]]

# One minibatch of layer inputs, as args and kwargs of the layer's `forward`.
_Batch: TypeAlias = tuple[tuple[Any, ...], dict[str, Any]]

# Stretch parameters of the rectified sigmoid h(V) = clip(sigmoid(V)(zeta - gamma) + gamma, 0, 1).
_ZETA = 1.1
_GAMMA = -0.1


def _rectified_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Return `h(V)`, the rectified sigmoid of Eq. 23."""
    return torch.clamp(torch.sigmoid(x) * (_ZETA - _GAMMA) + _GAMMA, 0.0, 1.0)


def _reconstruction_error(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the error of the LHS of Eq. 25, flattened to one row per sample."""
    return (output.float() - target.float()).reshape(output.shape[0], -1)


def adaround(
    module: ff.nn.QuantizedLinear | ff.nn.QuantizedConv2d,
    original_inputs: Iterable[tuple[tuple[Any, ...], dict[str, Any]]],
    quantized_inputs: Iterable[tuple[tuple[Any, ...], dict[str, Any]]],
    num_iterations: int = 10_000,
    reg_param: float = 0.01,
    beta_range: tuple[float, float] = (20.0, 2.0),
    warm_start: float = 0.2,
    learning_rate: float = 1e-3,
    range_estimator: _RangeEstimator | None = smoothed_minmax,
    layer_name: str = "",
) -> None:
    """Quantize a layer in-place using AdaRound.

    AdaRound learns, per weight element, whether to round up or down. It replaces the
    fixed rounding of the weight quantizer by `floor(W/s) + h(V)`, where `h` is a
    rectified sigmoid of a learnable tensor `V`. `V` is optimized to minimize the
    reconstruction error of the layer output, plus a regularizer that pushes `h(V)`
    towards 0 or 1. After optimization, a hard rounding is baked into `module.weight`.

    The objective is the asymmetric reconstruction of the paper (Eq. 25). The target is
    the output of this layer in the original model, `Wx`, while the reconstruction runs on
    the activations that the layer receives once the layers before it are quantized,
    `W~x^`. Because the target stays clean, the layer corrects the error that the layers
    before it introduce. Pass the same activations in both arguments to get the symmetric
    layer-wise objective of Eq. 21 instead.

    The weight quantization grid comes from `range_estimator`, or from an earlier calibration
    when `range_estimator` is `None`. AdaRound calibrates the grid one time, before the
    optimization, and then holds it fixed. The layer output is produced by the module's own
    `forward`, so bias, stride, padding, dilation and groups need no special handling, and
    every weight quantizer granularity is supported.

    After the call, `module.weight` holds the hard rounded weight and lies on the grid of the
    weight quantizer. The rounding variable is not kept. Do not estimate the weight ranges again,
    because a new grid throws the learned rounding away.

    Note:
        The caller must not use `torch.inference_mode`, because the optimization needs an
        autograd graph. `torch.no_grad` is fine; gradients are enabled internally.

    Note:
        The objective omits the activation function `f_a` of Eq. 25, thus the reconstruction
        is measured at the output of this layer. The paper puts `f_a` on both branches,
        so that an error which the activation function removes has no cost.

    Args:
        module: A layer whose `weight_quantizer` is a `LinearQuantizer`.
        original_inputs: Input activations of this layer in the original model, an
            iterable. Each entry is a tuple of args (tuple of obj) and kwargs (dict), and
            it is one minibatch of the optimization. The target is made from these.
        quantized_inputs: Input activations of this layer with the layers before it
            quantized, in the same batch order as `original_inputs`. The reconstruction
            uses these. Pass `original_inputs` again for the objective of Eq. 21.
        num_iterations: Number of optimization steps.
        reg_param: Weight of the rounding regularizer, lambda in the paper. Both loss terms
            are normalized per element, thus one value fits every layer shape and sequence
            length. The paper compares the two sums directly, so its lambda equals this
            value times `in_features / tokens_per_sample`.
        beta_range: Start and end value of the regularizer exponent beta. Beta follows a
            cosine schedule from the first to the second value.
        warm_start: Fraction of `num_iterations` during which the regularizer is
            disabled, so that `h(V)` can move freely before it is pushed to 0 or 1.
        learning_rate: Learning rate of the Adam optimizer on `V`.
        range_estimator: Range estimator for the weight quantization grid, as a type or as an
            instance. AdaRound learns the rounding on top of that grid and then holds the grid
            fixed, thus the estimator has a large effect on the result. `ff.range_setting.mse_grid`
            minimizes the error on the weights, which often beats the min-max default. Pass `None`
            to keep the grid that `module` already has, for example a grid from an earlier
            calibration of the complete model.
        layer_name: Optional name for logging.

    Raises:
        ValueError: If `weight_quantizer` is not a `LinearQuantizer`, if `warm_start` is
            outside `[0, 1)`, or if `range_estimator` is `None` while the weight quantizer has
            no grid yet.
        RuntimeError: If the caller uses `torch.inference_mode`.
    """
    if torch.is_inference_mode_enabled():
        msg = (
            "adaround optimizes an auxiliary parameter and requires an autograd graph, but it was "
            "called under `torch.inference_mode`, which cannot be re-enabled. Use "
            "`torch.no_grad` instead."
        )
        raise RuntimeError(msg)

    if not isinstance(weight_quantizer := module.weight_quantizer, ff.nn.LinearQuantizer):
        msg = f"weight_quantizer must be a LinearQuantizer, got {type(weight_quantizer).__name__}."
        raise ValueError(msg)

    if not 0.0 <= warm_start < 1.0:
        msg = f"warm_start must be in [0, 1), got {warm_start}."
        raise ValueError(msg)

    # Estimate quantizers ranges, this is the baseline AdaRound works on top of.
    if range_estimator is None:
        if weight_quantizer.has_uninitialized_params:
            msg = (
                "range_estimator=None keeps the quantization grid of weight_quantizer, but that "
                "quantizer has uninitialized parameters. Calibrate the quantizer first, or pass "
                "a range estimator."
            )
            raise ValueError(msg)
    else:
        with ff.estimate_ranges(weight_quantizer, range_estimator):
            weight_quantizer(module.weight)

    # Pre-calculate the target value (fp activations from 'module').
    targets: list[torch.Tensor] = []
    with ff.disable_quantization(module), torch.no_grad():
        device = next(module.parameters()).device
        for batch in original_inputs:
            args, kwargs = optree.tree_map(lambda x: x.to(device), batch)  # type: ignore[arg-type]
            targets.append(module(*args, **kwargs).detach())

    # Soft rounding override we place on the module.
    rounding = _SoftRounding(weight_quantizer, module.weight)
    quantized_batches = list(quantized_inputs)

    with (
        ff.strict_quantization(False),
        _frozen_parameters(module),
        weight_quantizer.register_override(rounding),
    ):
        _optimize(
            module,
            quantized_batches,
            targets,
            rounding,
            num_iterations=num_iterations,
            reg_param=reg_param,
            beta_range=beta_range,
            warm_start=warm_start,
            learning_rate=learning_rate,
            layer_name=layer_name,
        )

        # Measure the objective for the rounding that the optimization evaluated, and for the
        # hard rounding that the layer keeps. A soft loss that stays high means the optimization
        # itself does not converge. A hard loss above the soft loss means the optimization does
        # converge, but the hard rounding gives the gain back.
        soft_loss = _reconstruction_loss(module, quantized_batches, targets)
        with rounding.hard_rounding():
            hard_loss = _reconstruction_loss(module, quantized_batches, targets)

    with torch.no_grad():
        module.weight.copy_(rounding.hard_weight().to(module.weight.dtype))

    logger.info(
        "[AdaRound][wbits=%d][%s] soft_loss=%.6f hard_loss=%.6f undecided=%.2f%%",
        weight_quantizer.num_bits,
        layer_name,
        soft_loss,
        hard_loss,
        100.0 * rounding.undecided_fraction(),
    )


@torch.enable_grad()
def _optimize(
    module: torch.nn.Module,
    batches: list[_Batch],
    targets: list[torch.Tensor],
    rounding: _SoftRounding,
    *,
    num_iterations: int,
    reg_param: float,
    beta_range: tuple[float, float],
    warm_start: float,
    learning_rate: float,
    layer_name: str,
) -> None:
    """Optimize the rounding variable of `rounding` in-place.

    Each step draws one batch index, reconstructs the layer output from the batch of
    `batches` at that index, compares it with the target of the same index and takes an
    Adam step on the sum of the reconstruction loss and the rounding regularizer.
    """
    optimizer = torch.optim.Adam([rounding.rounding_variable], lr=learning_rate)
    warm_iterations = int(warm_start * num_iterations)
    recon_loss = round_loss = torch.zeros(())
    device = next(module.parameters()).device
    num_rounding_elements = rounding.rounding_variable.numel()

    for iteration in range(num_iterations):
        batch_index = int(torch.randint(len(batches), ()).item())
        args, kwargs = optree.tree_map(lambda x: x.to(device), batches[batch_index])  # type: ignore[arg-type]

        output = module(*args, **kwargs)

        # Calculate the LHS of eq 25. (squared frobenius norm)
        error = _reconstruction_error(output, targets[batch_index])
        recon_loss = error.pow(2).sum(dim=1).mean()

        if iteration < warm_iterations:
            round_loss = torch.zeros_like(recon_loss)
            loss = recon_loss
        else:
            # Calculate the RHS of eq 25. (f_reg)
            #
            # Eq. 25 sums f_reg over the weight elements and compares it with a sum over the
            # output elements of one sample. That ratio scales with in_features / tokens, so a
            # single reg_param cannot fit every layer shape and sequence length. Both terms are
            # therefore a mean over their own elements, and the regularizer takes the scale of
            # the reconstruction term back, which keeps the loss magnitude of Eq. 25.
            beta = _beta(iteration, num_iterations, beta_range, warm_iterations)
            reg_scale = reg_param * error.shape[1] / num_rounding_elements
            round_loss = (
                reg_scale
                * (
                    1 - (2 * _rectified_sigmoid(rounding.rounding_variable) - 1).abs().pow(beta)
                ).sum()
            )
            loss = recon_loss + round_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    logger.debug(
        "[AdaRound][%s] recon_loss=%.6f round_loss=%.6f",
        layer_name,
        recon_loss.item(),
        round_loss.item(),
    )


@torch.no_grad()
def _reconstruction_loss(
    module: torch.nn.Module, batches: list[_Batch], targets: list[torch.Tensor]
) -> float:
    """Return the reconstruction loss of `module`, averaged over every batch.

    This is the LHS of Eq. 25 that `_optimize` minimizes, but over all batches instead of over
    one drawn batch. The quantization state of `module` decides which rounding it measures.
    """
    device = next(module.parameters()).device
    losses = []
    for batch, target in zip(batches, targets):
        args, kwargs = optree.tree_map(lambda x: x.to(device), batch)  # type: ignore[arg-type]
        error = _reconstruction_error(module(*args, **kwargs), target)
        losses.append(error.pow(2).sum(dim=1).mean().item())
    return sum(losses) / len(losses)


class _SoftRounding:
    """Quantizer override that replaces rounding by a learnable soft rounding.

    See Equation 22. The soft quantized weight is `s * (clip(floor(W/s - z) + h(V), n, p) + z)`,
    with `h` the rectified sigmoid and `V` the only learnable tensor. `W`, `s` and `z` are fixed,
    so `floor(W/s - z)` is computed once.

    Args:
        quantizer: The calibrated weight quantizer that this override replaces.
        weight: The unquantized weight of the layer that owns `quantizer`.
    """

    def __init__(self, quantizer: ff.nn.LinearQuantizer, weight: torch.Tensor) -> None:
        self._weight_shape = weight.shape
        self._weight_dtype = weight.dtype
        self._tile_size = quantizer.granularity.tile_size(weight.shape)
        self._min_int = quantizer.integer_minimum
        self._max_int = quantizer.integer_maximum

        rows = tiled_tensor.tiles_to_rows(weight.detach().float(), self._tile_size)
        self._scale = quantizer.scale.detach().float().reshape(-1, 1)
        if quantizer.offset is None:
            self._offset = torch.zeros_like(self._scale)
        else:
            self._offset = quantizer.offset.detach().float().round().reshape(-1, 1)

        scaled = rows / self._scale - self._offset
        self._floor = torch.floor(scaled)

        # Initialize V such that h(V) equals the rounding residual. The soft weight then
        # starts at the unquantized weight and the regularizer decides the direction.
        residual = (scaled - self._floor).clamp(0.0, 1.0)
        alpha = -torch.log((_ZETA - _GAMMA) / (residual - _GAMMA) - 1)
        self.rounding_variable = torch.nn.Parameter(alpha)
        self._hard = False

    def __call__(
        self,
        _context: Any,
        _callback: Any,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """Return the quantized weight for the current rounding, ignoring the incoming weight."""
        if self._hard:
            return self._dequantize(self._hard_rounding())
        return self._dequantize(_rectified_sigmoid(self.rounding_variable))

    @contextlib.contextmanager
    def hard_rounding(self) -> Generator[None, None, None]:
        """Make the override return the hard rounded weight for the duration of the context.

        This measures the layer with the weight that it keeps after the call, while the soft
        rounding is what the optimization sees.
        """
        self._hard = True
        try:
            yield
        finally:
            self._hard = False

    def hard_weight(self) -> torch.Tensor:
        """Return the weight for the learned rounding, rounded hard to the grid."""
        return self._dequantize(self._hard_rounding())

    def _hard_rounding(self) -> torch.Tensor:
        """Return the hard rounding, 1 where the soft rounding leans up and 0 elsewhere."""
        # h(V) >= 0.5 is equivalent to V >= 0, so this rounds each element up exactly
        # where the soft rounding leans up.
        return (self.rounding_variable.detach() >= 0).to(self._floor.dtype)

    def undecided_fraction(self, threshold: float = 0.01) -> float:
        """Return the fraction of elements whose soft rounding did not converge to 0 or 1.

        A large fraction means the hard rounding baked into the weight differs from the
        rounding that the optimization actually evaluated.
        """
        soft = _rectified_sigmoid(self.rounding_variable.detach())
        undecided = (soft > threshold) & (soft < 1.0 - threshold)
        return undecided.sum().item() / soft.numel()

    def _dequantize(self, rounding: torch.Tensor) -> torch.Tensor:
        quantized = torch.clamp(self._floor + rounding, self._min_int, self._max_int)
        dequantized = (quantized + self._offset) * self._scale
        weight = tiled_tensor.rows_to_tiles(dequantized, self._weight_shape, self._tile_size)
        return weight.to(self._weight_dtype)


def _beta(
    iteration: int, num_iterations: int, beta_range: tuple[float, float], warm_iterations: int
) -> float:
    """Return the regularizer exponent for `iteration`, on a cosine schedule.

    Beta decays from `beta_range[0]` at the end of the warm-up to `beta_range[1]` at the
    last iteration. A high beta leaves `h(V)` free, a low beta forces a decision.
    """
    start, end = beta_range
    relative = (iteration - warm_iterations) / max(num_iterations - warm_iterations, 1)
    return end + 0.5 * (start - end) * (1 + math.cos(relative * math.pi))


@contextlib.contextmanager
def _frozen_parameters(module: torch.nn.Module) -> Generator[None, None, None]:
    """Disable gradients for the parameters of `module` for the duration of the context.

    AdaRound optimizes only the rounding variable. Without this, autograd also computes
    gradients for the weight, bias and quantization parameters of the layer.

    Lazy parameters are skipped. An uninitialized quantizer holds an `UninitializedParameter`,
    which rejects `requires_grad_`. Such a quantizer cannot run a forward pass either, thus the
    error must come from the quantizer and not from here.
    """
    requires_grad = [
        (param, param.requires_grad)
        for param in module.parameters()
        if not isinstance(param, torch.nn.UninitializedParameter)
    ]
    try:
        for param, _ in requires_grad:
            param.requires_grad_(False)
        yield
    finally:
        for param, flag in requires_grad:
            param.requires_grad_(flag)
