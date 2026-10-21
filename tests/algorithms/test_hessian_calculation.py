# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from collections.abc import Iterable
from typing import Any, Literal

import pytest
import torch

from fastforward.algorithms.hessian_calculation import (
    calculate_hessian,
    hessian_settings,
    invert_hessian,
)


def _batches(
    tensors: Iterable[torch.Tensor],
) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    """Wrap each tensor as one batch: an entry of args and kwargs."""
    return [((tensor,), {}) for tensor in tensors]


def test_calculate_hessian_does_not_modify_input_activations() -> None:
    # GIVEN float32 activations, so the dtype cast is a no-op and the reshape/transpose
    # yield views
    activations = [torch.randn(2, 8, 32) for _ in range(3)]
    expected = [activation.clone() for activation in activations]

    # WHEN we build the Hessian
    calculate_hessian(_batches(activations))

    # THEN the caller's activations are unchanged
    for activation, original in zip(activations, expected):
        torch.testing.assert_close(activation, original, rtol=0, atol=0)


def test_calculate_hessian_is_repeatable_for_a_shared_dataset() -> None:
    # GIVEN one dataset read twice, as the orchestrator hands the same tensor objects
    # to every function that reads an activation (q_proj/k_proj/v_proj share their input)
    dataset = _batches(torch.randn(2, 8, 32) for _ in range(4))

    # WHEN two functions build a Hessian from it in turn
    first = calculate_hessian(dataset)
    second = calculate_hessian(dataset)

    # THEN the second reader sees the same Hessian as the first
    torch.testing.assert_close(first, second)


def test_calculate_hessian_without_activations_is_refused() -> None:
    with pytest.raises(ValueError, match="without activations"):
        calculate_hessian([])


def test_calculate_hessian_of_a_convolution_matches_the_weight_layout() -> None:
    # GIVEN a convolution with a kernel that is not square, and with stride, padding and
    # dilation, and conv inputs
    torch.manual_seed(0)
    conv = torch.nn.Conv2d(
        3, 4, kernel_size=(3, 2), stride=(2, 1), padding=(1, 0), dilation=(1, 2), bias=False
    )
    images = [torch.randn(2, 3, 7, 6), torch.randn(1, 3, 7, 6)]

    # WHEN we build the Hessian from its settings
    hessian = calculate_hessian(_batches(images), **hessian_settings(conv))

    # THEN it has one row per weight in a flattened filter, and it orders them as the weight
    # does: the weight maps the Hessian of the patches to the Hessian of the outputs
    assert hessian.shape == (3 * 3 * 2, 3 * 3 * 2)
    with torch.no_grad():
        outputs = torch.cat([conv(image).permute(0, 2, 3, 1).reshape(-1, 4) for image in images])
    weight = conv.weight.detach().flatten(1)
    expected = 2 / outputs.shape[0] * outputs.T @ outputs
    torch.testing.assert_close(weight @ hessian @ weight.T, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("padding_mode", ["reflect", "replicate", "circular"])
def test_hessian_settings_refuses_a_padding_that_is_not_zeros(
    padding_mode: Literal["reflect", "replicate", "circular"],
) -> None:
    # GIVEN a convolution that pads with other values than zeros, which `unfold` cannot show
    conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1, padding_mode=padding_mode)

    # WHEN/THEN its settings are refused
    with pytest.raises(NotImplementedError, match="zero padding"):
        hessian_settings(conv)


def test_invert_hessian_does_not_modify_the_hessian() -> None:
    # GIVEN a Hessian
    hessian = calculate_hessian(_batches(torch.randn(2, 8, 16) for _ in range(2)))
    expected = hessian.clone()

    # WHEN one layer inverts it
    invert_hessian(hessian, perc_damp=0.01)

    # THEN the shared Hessian is unchanged, can be used for other layers
    torch.testing.assert_close(hessian, expected, rtol=0, atol=0)
