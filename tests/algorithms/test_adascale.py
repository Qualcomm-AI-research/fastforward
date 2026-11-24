# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import fastforward as ff
import pytest
import torch

from fastforward.algorithms.adascale import adascale, adascale_hessian
from fastforward.algorithms.hessian_calculation import finalize_hessian, hessian_sum_and_count


def _quantized_linear(
    granularity: ff.granularity.Granularity, num_bits: int = 4, symmetric: bool = False
) -> ff.nn.QuantizedLinear:
    """Create a QuantizedLinear with an uncalibrated LinearQuantizer on its weight."""
    torch.manual_seed(0)
    module = ff.nn.QuantizedLinear(64, 32, bias=False)
    module.weight_quantizer = ff.nn.LinearQuantizer(
        num_bits=num_bits, granularity=granularity, symmetric=symmetric
    )
    return module


def _weight_quantizer(module: ff.nn.QuantizedLinear) -> ff.nn.LinearQuantizer:
    """Return the weight quantizer of `module`, narrowed to `LinearQuantizer`."""
    quantizer = module.weight_quantizer
    assert isinstance(quantizer, ff.nn.LinearQuantizer)
    return quantizer


def _activations() -> torch.Tensor:
    """Build activations concentrated on one input direction.

    For isotropic activations the output error reduces to the weight error, which
    round-to-nearest already minimizes. A dominant direction makes the grid choice matter.
    """
    torch.manual_seed(1)
    direction = torch.randn(64)
    return torch.randn(256, 1) * direction + 0.05 * torch.randn(256, 64)


def _quantized_activations(activations: torch.Tensor) -> torch.Tensor:
    """Return `activations` with an error, as the inputs of a layer in a quantized model."""
    torch.manual_seed(3)
    return activations + 0.1 * torch.randn_like(activations)


def _nearest_weight(module: ff.nn.QuantizedLinear) -> torch.Tensor:
    """Return the round-to-nearest weight on the grid that AdaScale estimates."""
    quantizer = _weight_quantizer(module)
    with (
        ff.strict_quantization(False),
        ff.estimate_ranges(quantizer, ff.range_setting.smoothed_minmax),
    ):
        quantizer(module.weight)
    with ff.strict_quantization(False), torch.no_grad():
        return quantizer(module.weight).dequantize()


def _nearest_scale(module: ff.nn.QuantizedLinear) -> torch.Tensor:
    """Return the scale that the round-to-nearest calibration of a copy of `module` gives."""
    reference = _quantized_linear(
        _weight_quantizer(module).granularity, symmetric=_weight_quantizer(module).symmetric
    )
    _nearest_weight(reference)
    return _weight_quantizer(reference).scale.detach().clone()


GRANULARITIES = [
    ff.granularity.PerTensor(),
    ff.granularity.PerChannel(channel_dim=0),
    ff.granularity.PerChannel(channel_dim=1),
    ff.granularity.PerBlock(block_dims=1, block_sizes=16, per_channel_dims=0),
    ff.granularity.PerTile((16, 16)),
]
GRANULARITY_IDS = ["per_tensor", "per_channel_dim0", "per_channel_dim1", "per_block", "per_tile"]


_Batch = tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor]]


def _batches(activations: torch.Tensor, num_batches: int = 4) -> list[_Batch]:
    """Split `activations` into minibatches, as the `InputActivations` flow gives them."""
    return [((batch,), {}) for batch in activations.chunk(num_batches)]


class _Block(torch.nn.Module):
    """A residual block of two QuantizedLinear layers with a keyword input."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.up = ff.nn.QuantizedLinear(64, 32)
        self.down = ff.nn.QuantizedLinear(32, 64)
        for layer in (self.up, self.down):
            layer.weight_quantizer = ff.nn.LinearQuantizer(
                num_bits=3, granularity=ff.granularity.PerChannel(channel_dim=0)
            )

    def forward(self, x: torch.Tensor, *, gate: torch.Tensor) -> torch.Tensor:
        residual: torch.Tensor = self.down(torch.relu(self.up(x)))
        return x + gate * residual


def _block_batches(activations: torch.Tensor, num_batches: int = 4) -> list[_Batch]:
    """Split `activations` into minibatches of block inputs, with the keyword input."""
    return [((batch,), {"gate": torch.ones(64)}) for batch in activations.chunk(num_batches)]


def _block_error(block: _Block, reference: _Block, activations: torch.Tensor) -> float:
    """Return the squared output error of `block` against `reference`, without quantizers."""
    gate = torch.ones(64)
    with torch.no_grad(), ff.disable_quantization(block), ff.disable_quantization(reference):
        error = block(activations, gate=gate) - reference(activations, gate=gate)
    return float(error.pow(2).sum(dim=-1).mean())


def _nearest_block() -> _Block:
    """Return a `_Block` with the round-to-nearest weight in each layer."""
    block = _Block()
    with torch.no_grad():
        for layer in (block.up, block.down):
            layer.weight.copy_(_nearest_weight(layer))
    return block


@pytest.mark.slow
def test_adascale_reduces_the_block_output_error_over_nearest_rounding() -> None:
    # GIVEN a block, minibatches of its inputs and the round-to-nearest error at its output
    activations = _activations()
    original = _Block()
    nearest_error = _block_error(_nearest_block(), original, activations)

    # WHEN we quantize the same block with AdaScale on the original inputs
    block = _Block()
    batches = _block_batches(activations)
    with torch.no_grad():
        adascale(block, batches, batches, num_iterations=200)

    # THEN the error at the block output is below the round-to-nearest error
    assert _block_error(block, original, activations) < nearest_error


@pytest.mark.slow
def test_adascale_on_quantized_inputs_reduces_the_error_against_the_original_layer() -> None:
    # GIVEN a 3 bit layer, original and quantized minibatches, and the round-to-nearest error
    # against the original layer
    granularity = ff.granularity.PerChannel(channel_dim=0)
    activations = _activations()
    quantized = _quantized_activations(activations)
    original = _quantized_linear(granularity, num_bits=3)

    def error_against_original(weight: torch.Tensor) -> torch.Tensor:
        outputs = quantized @ weight.T - activations @ original.weight.data.T
        return outputs.pow(2).sum(dim=-1).mean()

    nearest_error = error_against_original(_nearest_weight(original))

    # WHEN we quantize the same layer with AdaScale on the quantized inputs only
    module = _quantized_linear(granularity, num_bits=3)
    with torch.no_grad():
        adascale(
            module,
            _batches(activations),
            _batches(quantized),
            num_iterations=200,
            quantized_input_ratio=1.0,
        )

    # THEN the error against the original layer is below the round-to-nearest error
    assert error_against_original(module.weight.data) < nearest_error


@pytest.mark.parametrize("symmetric", [False, True], ids=["asymmetric", "symmetric"])
def test_adascale_weight_lands_on_the_learned_grid(symmetric: bool) -> None:
    # GIVEN a layer whose weight and quantizer were replaced by an AdaScale result
    module = _quantized_linear(ff.granularity.PerChannel(channel_dim=0), symmetric=symmetric)
    scale = _nearest_scale(module)
    batches = _batches(_activations())
    with torch.no_grad():
        adascale(module, batches, batches, num_iterations=20)

    # WHEN we quantize the stored weight again with the stored quantizer
    with ff.strict_quantization(False), torch.no_grad():
        requantized = module.weight_quantizer(module.weight).dequantize()

    # THEN nothing changes, and the quantizer holds a grid that differs from the initial grid
    torch.testing.assert_close(requantized, module.weight.data)
    assert not torch.allclose(_weight_quantizer(module).scale, scale)


@pytest.mark.parametrize("symmetric", [False, True], ids=["asymmetric", "symmetric"])
@pytest.mark.parametrize("granularity", GRANULARITIES, ids=GRANULARITY_IDS)
def test_adascale_without_optimization_matches_nearest_rounding(
    granularity: ff.granularity.Granularity, symmetric: bool
) -> None:
    # GIVEN a layer and the round-to-nearest weight of its calibrated quantizer
    nearest = _nearest_weight(_quantized_linear(granularity, symmetric=symmetric))

    # WHEN we run AdaScale with no optimization steps, so all parameters keep their zero
    # initial value
    module = _quantized_linear(granularity, symmetric=symmetric)
    batches = _batches(_activations())
    with torch.no_grad():
        adascale(module, batches, batches, num_iterations=0)

    # THEN the stored weight is exactly the round-to-nearest weight, for every granularity
    torch.testing.assert_close(module.weight.data, nearest)


def test_adascale_keeps_the_existing_grid_for_a_none_range_estimator() -> None:
    # GIVEN a layer whose quantizer is calibrated on data that is not its own weight
    module = _quantized_linear(ff.granularity.PerChannel(channel_dim=0))
    quantizer = _weight_quantizer(module)
    with (
        ff.strict_quantization(False),
        ff.estimate_ranges(quantizer, ff.range_setting.smoothed_minmax),
    ):
        quantizer(2.0 * module.weight.data)
    scale = quantizer.scale.detach().clone()
    batches = _batches(_activations())

    # WHEN we run AdaScale without optimization steps and without a range estimator
    with torch.no_grad():
        adascale(module, batches, batches, num_iterations=0, range_estimator=None)

    # THEN the grid is untouched
    torch.testing.assert_close(quantizer.scale.detach(), scale)


@pytest.mark.slow
def test_adascale_with_a_zero_ratio_ignores_the_quantized_inputs() -> None:
    # GIVEN a layer, original minibatches and quantized minibatches
    granularity = ff.granularity.PerChannel(channel_dim=0)
    activations = _activations()
    originals = _batches(activations)
    quantized = _batches(_quantized_activations(activations))

    # WHEN we run AdaScale with a quantized input ratio of 0, and again with the original
    # inputs in both arguments, with the same random state
    module = _quantized_linear(granularity, num_bits=3)
    reference = _quantized_linear(granularity, num_bits=3)
    with torch.no_grad():
        torch.manual_seed(0)
        adascale(module, originals, quantized, num_iterations=20, quantized_input_ratio=0.0)
        torch.manual_seed(0)
        adascale(reference, originals, originals, num_iterations=20, quantized_input_ratio=0.0)

    # THEN both runs store the same weight
    torch.testing.assert_close(module.weight.data, reference.weight.data)


@pytest.mark.slow
def test_adascale_with_a_zero_ratio_ignores_the_quantized_keyword_inputs() -> None:
    # GIVEN block minibatches, and the same minibatches with another keyword input
    activations = _activations()
    originals = _block_batches(activations)
    quantized = [(args, {"gate": 2 * kwargs["gate"]}) for args, kwargs in originals]

    # WHEN we run AdaScale with a quantized input ratio of 0, and again with the original
    # inputs in both arguments, with the same random state
    block, reference = _Block(), _Block()
    with torch.no_grad():
        torch.manual_seed(0)
        adascale(block, originals, quantized, num_iterations=20, quantized_input_ratio=0.0)
        torch.manual_seed(0)
        adascale(reference, originals, originals, num_iterations=20, quantized_input_ratio=0.0)

    # THEN both runs store the same weights
    for layer, reference_layer in ((block.up, reference.up), (block.down, reference.down)):
        torch.testing.assert_close(layer.weight.data, reference_layer.weight.data)


@pytest.mark.parametrize(
    ("original_is_quantized", "quantized_is_quantized"),
    [(False, True), (True, False), (True, True)],
    ids=["plain_and_quantized", "quantized_and_plain", "both_quantized"],
)
def test_adascale_mixes_quantized_tensor_inputs_under_strict_quantization(
    original_is_quantized: bool, quantized_is_quantized: bool
) -> None:
    # GIVEN plain minibatches, the same minibatches as the `QuantizedTensor` outputs of an
    # 8 bit activation quantizer, and those dequantized
    granularity = ff.granularity.PerChannel(channel_dim=0)
    activations = _activations()
    activation_quantizer = ff.nn.LinearQuantizer(num_bits=8)
    with (
        ff.strict_quantization(False),
        ff.estimate_ranges(activation_quantizer, ff.range_setting.smoothed_minmax),
    ):
        activation_quantizer(activations)
    plain = _batches(activations)
    with torch.no_grad():
        quantized: list[_Batch] = [((activation_quantizer(x),), {}) for (x,), _ in plain]
    dequantized: list[_Batch] = [((x.dequantize(),), {}) for (x,), _ in quantized]

    # WHEN we run AdaScale under strict quantization with `QuantizedTensor` minibatches as the
    # original inputs, the quantized inputs or both, and again with the dequantized minibatches
    # in their place, with the same random state
    module = _quantized_linear(granularity, num_bits=3)
    reference = _quantized_linear(granularity, num_bits=3)
    with torch.no_grad(), ff.strict_quantization(True):
        torch.manual_seed(0)
        adascale(
            module,
            quantized if original_is_quantized else plain,
            quantized if quantized_is_quantized else plain,
            num_iterations=20,
        )
        torch.manual_seed(0)
        adascale(
            reference,
            dequantized if original_is_quantized else plain,
            dequantized if quantized_is_quantized else plain,
            num_iterations=20,
        )

    # THEN both runs store the same weight
    torch.testing.assert_close(module.weight.data, reference.weight.data)


def test_adascale_leaves_the_biases_and_the_gradient_state_unchanged() -> None:
    # GIVEN a block whose bias of one layer does not require a gradient, outside
    # `torch.no_grad`
    block = _Block()
    block.up.bias.requires_grad_(False)
    biases = [block.up.bias.data.clone(), block.down.bias.data.clone()]
    batches = _block_batches(_activations())

    # WHEN we run AdaScale on the block
    adascale(block, batches, batches, num_iterations=2)

    # THEN the biases are untouched, no parameter has a gradient, and each keeps its
    # requires_grad flag
    torch.testing.assert_close([block.up.bias.data, block.down.bias.data], biases)
    assert all(param.grad is None for param in block.parameters())
    assert not block.up.bias.requires_grad
    assert block.down.bias.requires_grad


def test_adascale_rejects_a_none_range_estimator_without_a_grid() -> None:
    # GIVEN a layer whose quantizer has no grid yet
    module = _quantized_linear(ff.granularity.PerChannel(channel_dim=0))
    batches = _batches(_activations())

    # WHEN we run AdaScale without a range estimator, THEN it reports the missing grid
    with pytest.raises(ValueError, match="uninitialized"), torch.no_grad():
        adascale(module, batches, batches, num_iterations=1, range_estimator=None)


def test_adascale_rejects_a_non_linear_weight_quantizer() -> None:
    # GIVEN a layer whose weight quantizer is still a stub
    module = ff.nn.QuantizedLinear(64, 32, bias=False)
    batches = _batches(_activations())

    # WHEN we run AdaScale, THEN it reports the unsupported quantizer
    with pytest.raises(ValueError, match="LinearQuantizer"), torch.no_grad():
        adascale(module, batches, batches, num_iterations=1)


def test_adascale_rejects_a_module_without_quantized_linear_layers() -> None:
    # GIVEN a block without QuantizedLinear layers
    block = torch.nn.Sequential(torch.nn.Linear(64, 64))
    batches = _batches(_activations())

    # WHEN we run AdaScale on the block, THEN it reports the missing layers
    with pytest.raises(ValueError, match="QuantizedLinear layers"), torch.no_grad():
        adascale(block, batches, batches, num_iterations=1)


@pytest.mark.parametrize("num_quantized", [0, 3], ids=["empty", "mismatched"])
def test_adascale_rejects_inputs_without_matching_batches(num_quantized: int) -> None:
    # GIVEN a layer, four original batches and a different number of quantized batches
    module = _quantized_linear(ff.granularity.PerChannel(channel_dim=0))
    activations = _activations()
    quantized = _batches(activations)[:num_quantized]
    originals = _batches(activations) if num_quantized else []

    # WHEN we run AdaScale, THEN it reports the number of batches
    with pytest.raises(ValueError, match="number of original and quantized batches"):
        with torch.no_grad():
            adascale(module, originals, quantized, num_iterations=1)


@pytest.mark.parametrize("ratio", [-0.1, 1.1])
def test_adascale_rejects_a_ratio_outside_the_unit_interval(ratio: float) -> None:
    # GIVEN a layer and its minibatches
    module = _quantized_linear(ff.granularity.PerChannel(channel_dim=0))
    batches = _batches(_activations())

    # WHEN we run AdaScale with an invalid ratio, THEN it reports the valid range
    with pytest.raises(ValueError, match=r"quantized_input_ratio must be in \[0, 1\]"):
        with torch.no_grad():
            adascale(module, batches, batches, num_iterations=1, quantized_input_ratio=ratio)


def test_adascale_raises_under_inference_mode() -> None:
    # GIVEN a block, its minibatches and a caller that uses inference mode
    block = _Block()
    batches = _block_batches(_activations())

    # WHEN we run AdaScale on the block, THEN it reports that inference mode cannot be used
    with torch.inference_mode(), pytest.raises(RuntimeError, match="inference_mode"):
        adascale(block, batches, batches, num_iterations=1)


def _hessian(activations: torch.Tensor) -> torch.Tensor:
    """Return the Hessian of `activations` as the `hessian_original` flow gives it."""
    state = hessian_sum_and_count([((activations,), {})])
    return finalize_hessian(state)


@pytest.mark.slow
@pytest.mark.parametrize("symmetric", [False, True], ids=["asymmetric", "symmetric"])
def test_adascale_hessian_reduces_the_output_error_over_nearest_rounding(symmetric: bool) -> None:
    # GIVEN a 3 bit layer, activations with a dominant direction and the round-to-nearest error
    granularity = ff.granularity.PerChannel(channel_dim=0)
    activations = _activations()
    original = _quantized_linear(granularity, num_bits=3, symmetric=symmetric)

    def output_error(weight: torch.Tensor) -> torch.Tensor:
        return (activations @ (weight - original.weight.data).T).pow(2).sum(dim=-1).mean()

    nearest_error = output_error(_nearest_weight(original))

    # WHEN we quantize the same layer with AdaScale on the Hessian of the activations
    module = _quantized_linear(granularity, num_bits=3, symmetric=symmetric)
    with torch.no_grad():
        adascale_hessian(module, _hessian(activations), num_iterations=500)

    # THEN the output error is far below the round-to-nearest error
    assert output_error(module.weight.data) < 0.2 * nearest_error
