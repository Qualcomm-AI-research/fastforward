# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#
# AdaScale combines the weight scaling of FlexRound (https://arxiv.org/abs/2306.00317) with the
# learnable weight clipping of OmniQuant (https://arxiv.org/abs/2308.13137).
#
# `adascale` and `adascale_hessian` learn the same parameters with the same optimizer. They
# differ only in the loss.

from __future__ import annotations

import itertools
import logging

from typing import Any, Callable, Iterable, Sequence, TypeAlias

import torch
import torch.utils._pytree as pytree

import fastforward as ff

from fastforward.quantization import affine, tiled_tensor
from fastforward.range_setting import RangeEstimator, smoothed_minmax

logger = logging.getLogger(__name__)

# A range estimator, as a type or as an instance, in the form that `estimate_ranges` takes.
_RangeEstimator: TypeAlias = RangeEstimator[Any, Any] | type[RangeEstimator[Any, Any]]

# One minibatch of layer inputs, as args and kwargs of the layer's `forward`.
_Batch: TypeAlias = tuple[tuple[Any, ...], dict[str, Any]]


# The number of optimization steps, the first ones and the last ones, whose mean loss is logged.
_NUM_LOGGED_STEPS = 16


def adascale(
    module: torch.nn.Module,
    original_inputs: Iterable[_Batch],
    quantized_inputs: Iterable[_Batch],
    num_iterations: int = 1500,
    clip_learning_rate: float = 1e-3,
    scale_learning_rate: float = 5e-4,
    quantized_input_ratio: float = 0.5,
    range_estimator: _RangeEstimator | None = smoothed_minmax,
    layer_name: str = "",
) -> None:
    """Quantize the QuantizedLinear layers of `module` in-place with AdaScale.

    For each layer, AdaScale divides the weight by `exp(S2 + s3)` before rounding (FlexRound)
    and clips the range of each quantizer tile to `[exp(beta) * min, exp(gamma) * max]`
    (OmniQuant). All parameters start at zero, which is round-to-nearest on the calibrated grid.
    Each step takes the next minibatch, in order and cycling, and minimizes the mean squared
    error between two outputs of `module` without quantizers: with the AdaScale weights on the
    mixed inputs, and with the original weights on the original inputs. The mixed inputs take
    each element from the original or from the quantized inputs.

    Args:
        module: A QuantizedLinear layer, or a module that contains them, such as a decoder
            layer. Each layer needs a `LinearQuantizer` as `weight_quantizer`.
        original_inputs: Minibatches of the inputs of `module` in the original model, each a
            tuple of args and kwargs.
        quantized_inputs: The same minibatches in the quantized model.
        num_iterations: Number of optimization steps.
        clip_learning_rate: Learning rate of `gamma` and `beta`.
        scale_learning_rate: Learning rate of `S2` and `s3`.
        quantized_input_ratio: Probability that an element of a mixed input comes from
            `quantized_inputs`.
        range_estimator: Range estimator for the initial weight grids. `None` keeps the
            existing grids.
        layer_name: Name for logging.

    Raises:
        ValueError: If the layers, inputs or `quantized_input_ratio` are invalid.
        RuntimeError: Under `torch.inference_mode`. Use `torch.no_grad` instead.
    """
    # The layers by the name of their weight in `module`. A single layer has the name "weight".
    layers = {
        f"{name}.weight".removeprefix("."): layer
        for name, layer in module.named_modules()
        if isinstance(layer, ff.nn.QuantizedLinear)
    }
    originals, quantized = list(original_inputs), list(quantized_inputs)
    if not layers:
        msg = f"adascale needs a module with QuantizedLinear layers, got {type(module).__name__}."
        raise ValueError(msg)
    if not originals or len(originals) != len(quantized):
        msg = (
            "adascale needs the same number of original and quantized batches, and at least "
            f"one, got {len(originals)} and {len(quantized)}."
        )
        raise ValueError(msg)
    if not 0.0 <= quantized_input_ratio <= 1.0:
        msg = f"quantized_input_ratio must be in [0, 1], got {quantized_input_ratio}."
        raise ValueError(msg)

    scalings = {
        name: AdaScaleParameters.from_layer(layer, range_estimator)
        for name, layer in layers.items()
    }
    device = next(iter(layers.values())).weight.device
    batches = itertools.cycle(zip(originals, quantized))

    def loss() -> torch.Tensor:
        batch = pytree.tree_map_only(torch.Tensor, lambda tensor: tensor.to(device), next(batches))
        (original_args, original_kwargs), (quantized_args, quantized_kwargs) = batch
        with torch.no_grad(), ff.disable_quantization(module):
            target: torch.Tensor = module(*original_args, **original_kwargs)

        # Each element of a mixed input comes from the quantized input with probability
        # `quantized_input_ratio`, else from the original input. Inputs that are not
        # floating-point tensors come from the quantized inputs.
        def mix(quantized_input: Any, original_input: Any) -> Any:
            if not (
                isinstance(quantized_input, torch.Tensor) and quantized_input.is_floating_point()
            ):
                return quantized_input
            from_quantized = (
                torch.rand(quantized_input.shape, device=device) < quantized_input_ratio
            )
            return torch.where(from_quantized, quantized_input, original_input)

        mixed_args, mixed_kwargs = pytree.tree_map(
            mix, (quantized_args, quantized_kwargs), (original_args, original_kwargs)
        )
        # `functional_call` runs `module` with the AdaScale weights in place of its own.
        weights = {
            name: scalings[name].quantized_weight().to(layer.weight.dtype)
            for name, layer in layers.items()
        }
        with ff.disable_quantization(module):
            output: torch.Tensor = torch.func.functional_call(
                module, weights, mixed_args, mixed_kwargs
            )
        return (output.float() - target.float()).pow(2).sum(dim=-1).mean()

    losses = _optimize(
        list(scalings.values()),
        loss,
        num_iterations=num_iterations,
        clip_learning_rate=clip_learning_rate,
        scale_learning_rate=scale_learning_rate,
    )
    for name, layer in layers.items():
        scalings[name].store(layer)
    _log_losses("AdaScale", len(layers), layer_name, losses)


def adascale_hessian(
    module: ff.nn.QuantizedLinear,
    hessian: torch.Tensor,
    num_iterations: int = 1500,
    clip_learning_rate: float = 1e-3,
    scale_learning_rate: float = 5e-4,
    range_estimator: _RangeEstimator | None = smoothed_minmax,
    layer_name: str = "",
) -> None:
    """Quantize a QuantizedLinear layer in-place with AdaScale on the input Hessian.

    Learns the parameters of `adascale`, but minimizes the layer output error in closed form,
    `tr((W - W_q) H (W - W_q)^T)` with `H = 2/N sum_x x x^T`. Each step uses all calibration
    tokens, at a cost independent of their number. The logged losses are halved, which makes
    them equal to those of `adascale` on the same layer and inputs.

    Args:
        module: A QuantizedLinear layer with a `LinearQuantizer` as `weight_quantizer`.
        hessian: The Hessian `H` of the layer inputs, of shape [in_features, in_features].
        num_iterations: Number of optimization steps.
        clip_learning_rate: Learning rate of `gamma` and `beta`.
        scale_learning_rate: Learning rate of `S2` and `s3`.
        range_estimator: Range estimator for the initial weight grid. `None` keeps the
            existing grid.
        layer_name: Name for logging.

    Raises:
        ValueError: If the weight quantizer is invalid.
        RuntimeError: Under `torch.inference_mode`. Use `torch.no_grad` instead.
    """
    scaling = AdaScaleParameters.from_layer(module, range_estimator)
    hessian = hessian.to(device=scaling.weight.device, dtype=torch.float32)

    def loss() -> torch.Tensor:
        # The layer output error `tr((W - W_q) H (W - W_q)^T)`.
        delta = scaling.weight - scaling.quantized_weight()
        return ((delta @ hessian) * delta).sum()

    losses = _optimize(
        [scaling],
        loss,
        num_iterations=num_iterations,
        clip_learning_rate=clip_learning_rate,
        scale_learning_rate=scale_learning_rate,
    )
    scaling.store(module)
    # `H = 2/N sum x x^T` doubles the output error, so halve it to log the loss of `adascale`.
    _log_losses("AdaScale Hessian", 1, layer_name, losses / 2)


def _log_losses(algorithm: str, num_layers: int, layer_name: str, losses: torch.Tensor) -> None:
    """Log the mean loss of the first and of the last steps, if there are steps."""
    if not losses.numel():
        return
    logger.info(
        "[%s][layers=%d][%s] initial_loss=%.6f final_loss=%.6f",
        algorithm,
        num_layers,
        layer_name,
        losses[:_NUM_LOGGED_STEPS].mean().item(),
        losses[-_NUM_LOGGED_STEPS:].mean().item(),
    )


@torch.enable_grad()
def _optimize(
    scalings: Sequence[AdaScaleParameters],
    loss: Callable[[], torch.Tensor],
    *,
    num_iterations: int,
    clip_learning_rate: float,
    scale_learning_rate: float,
) -> torch.Tensor:
    """Optimize the parameters of `scalings` in-place, with one step for each call of `loss`.

    Returns:
        The loss of each step, before its update, on the CPU.
    """
    clip_params = [param for scaling in scalings for param in (scaling.gamma, scaling.beta)]
    scale_params = [
        param for scaling in scalings for param in (scaling.element_scale, scaling.tile_scale)
    ]
    optimizer = torch.optim.Adam([
        {"params": clip_params, "lr": clip_learning_rate},
        {"params": scale_params, "lr": scale_learning_rate},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_iterations, eta_min=0.0
    )
    losses = []
    for _ in range(num_iterations):
        step_loss = loss()
        optimizer.zero_grad()
        # Only the AdaScale parameters get gradients, not the parameters of the model.
        step_loss.backward(inputs=clip_params + scale_params)
        optimizer.step()
        scheduler.step()
        losses.append(step_loss.detach())
    return torch.stack(losses).cpu() if losses else torch.empty(0)


class AdaScaleParameters:
    """The learnable parameters of AdaScale for one weight, and the weight they quantize.

    All parameters start at zero, which is round-to-nearest on the grid of `quantizer`.
    `quantized_weight()` is differentiable in the parameters, and `store` writes the result to
    the layer.

    `element_scale` has the shape of the weight. The other parameters have one row per
    quantizer tile.

    Args:
        quantizer: The calibrated weight quantizer whose grid AdaScale starts from.
        weight: The unquantized weight, in float32.

    Attributes:
        weight: The unquantized weight, in float32.
        element_scale: `S2`, one value per weight element.
        tile_scale: `s3`, one value per tile.
        gamma: The log factor of the maximum of the range, one value per tile.
        beta: The log factor of the minimum of the range, one value per tile.
    """

    def __init__(self, quantizer: ff.nn.LinearQuantizer, weight: torch.Tensor) -> None:
        self.weight = weight
        self._granularity = quantizer.granularity
        self._num_bits = quantizer.num_bits
        self._tile_size = quantizer.granularity.tile_size(weight.shape)
        self._integer_minimum = quantizer.integer_minimum
        self._integer_maximum = quantizer.integer_maximum

        # The calibrated range of each tile, which the clipping factors scale.
        scale = quantizer.scale.detach().float().reshape(-1, 1)
        offset = None if quantizer.offset is None else quantizer.offset.detach().float().reshape(-1)
        range_min, range_max = affine.quantization_range(scale.reshape(-1), offset, self._num_bits)
        self._range_min = torch.as_tensor(range_min).reshape(-1, 1)
        self._range_max = torch.as_tensor(range_max).reshape(-1, 1)
        # Only an asymmetric quantizer learns its offset; others keep the offset they have.
        self._learn_offset = isinstance(quantizer.offset, torch.nn.Parameter)
        self._fixed_offset = offset

        self.element_scale = torch.nn.Parameter(torch.zeros_like(weight))
        self.tile_scale = torch.nn.Parameter(torch.zeros_like(scale))
        self.gamma = torch.nn.Parameter(torch.zeros_like(scale))
        self.beta = torch.nn.Parameter(torch.zeros_like(scale))

    @classmethod
    def from_layer(
        cls, layer: ff.nn.QuantizedLinear, range_estimator: _RangeEstimator | None
    ) -> AdaScaleParameters:
        """Check `layer`, calibrate its weight grid and return the parameters at zero.

        Args:
            layer: A QuantizedLinear layer with a `LinearQuantizer` as `weight_quantizer`.
            range_estimator: Range estimator for the initial weight grid. `None` keeps the
                existing grid.

        Raises:
            ValueError: If `weight_quantizer` is not a `LinearQuantizer`, or if
                `range_estimator` is `None` while the weight quantizer has no grid yet.
            RuntimeError: If the caller uses `torch.inference_mode`.
        """
        if torch.is_inference_mode_enabled():
            msg = (
                "AdaScale optimizes auxiliary parameters and requires an autograd graph, but it "
                "was called under `torch.inference_mode`, which cannot be re-enabled. Use "
                "`torch.no_grad` instead."
            )
            raise RuntimeError(msg)

        quantizer = layer.weight_quantizer
        if not isinstance(quantizer, ff.nn.LinearQuantizer):
            msg = f"weight_quantizer must be a LinearQuantizer, got {type(quantizer).__name__}."
            raise ValueError(msg)

        if range_estimator is not None:
            with ff.estimate_ranges(quantizer, range_estimator):
                quantizer(layer.weight)
        elif quantizer.has_uninitialized_params:
            msg = (
                "range_estimator=None keeps the quantization grid of weight_quantizer, but that "
                "quantizer has uninitialized parameters. Calibrate the quantizer first, or pass "
                "a range estimator."
            )
            raise ValueError(msg)

        return cls(quantizer, layer.weight.detach().float())

    @torch.no_grad()
    def store(self, layer: ff.nn.QuantizedLinear) -> None:
        """Store the learned grid in the weight quantizer and the quantized weight in `layer`."""
        quantizer = layer.weight_quantizer
        assert isinstance(quantizer, ff.nn.LinearQuantizer)
        scale, offset = self.quantization_parameters()
        quantizer.scale.copy_(scale.reshape(quantizer.scale.shape))
        if quantizer.offset is not None and offset is not None:
            quantizer.offset.copy_(offset.reshape(quantizer.offset.shape))
        layer.weight.copy_(self.quantized_weight().to(layer.weight.dtype))

    def quantization_parameters(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return the scale and offset of the clipped range, one value per tile.

        OmniQuant clips the calibrated range `[min, max]` of each tile to
        `[exp(beta) * min, exp(gamma) * max]`. At `beta = gamma = 0` this is the calibrated
        range.
        """
        clipped_min = torch.exp(self.beta) * self._range_min
        clipped_max = torch.exp(self.gamma) * self._range_max

        # The grid divides the clipped range into equal steps, one per integer interval.
        num_steps = self._integer_maximum - self._integer_minimum
        scale = (clipped_max - clipped_min) / num_steps
        scale = scale.clamp_min(torch.finfo(scale.dtype).eps).reshape(-1)

        if not self._learn_offset:
            # A symmetric quantizer keeps its offset, so its grid stays centered on zero.
            return scale, self._fixed_offset
        # The offset puts `integer_minimum` on `clipped_min`.
        offset = clipped_min.reshape(-1) / scale - self._integer_minimum
        return scale, offset

    def adjusted_weight(self) -> torch.Tensor:
        """Return the weight divided by `exp(S2 + s3)`, differentiable.

        FlexRound divides the weight by `exp(S2 + s3)` before rounding, with `S2` per element
        and `s3` per tile. The rounded weight is not multiplied back: the division only
        moves elements across rounding boundaries. At `S2 = s3 = 0` this is the weight.
        """
        # `s3` has one row per tile. Spread each value over the elements of its tile.
        elements_per_tile = self.weight.numel() // self.tile_scale.shape[0]
        tile_scale_rows = self.tile_scale.expand(-1, elements_per_tile)
        tile_scale = tiled_tensor.rows_to_tiles(tile_scale_rows, self.weight.shape, self._tile_size)
        return self.weight / torch.exp(self.element_scale + tile_scale)

    def quantized_weight(self) -> torch.Tensor:
        """Return the dequantized weight of the current parameters, differentiable.

        Rounds `adjusted_weight()` to the clipped grid. At `S2 = s3 = 0` and
        `beta = gamma = 0` this is round-to-nearest.
        """
        # The rounding passes the gradient straight through.
        scale, offset = self.quantization_parameters()
        quantized = affine.quantize(
            self.adjusted_weight(), scale, offset, self._granularity, self._num_bits
        )
        return quantized.dequantize()
