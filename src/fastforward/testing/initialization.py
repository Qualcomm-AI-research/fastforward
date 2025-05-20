# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Utilities for initializing quantizers in testing functions."""

from typing import Any, Callable

import torch

import fastforward as ff

from fastforward.quantization.granularity import Granularity
from fastforward.quantization.quant_init import QuantizerCollection


def initialize_quantizers_to_linear_quantizer(
    quant_model: torch.nn.Module,
    activation_quantizers: QuantizerCollection | str,
    parameter_quantizers: QuantizerCollection | str,
    granularity_activations: Granularity | None = None,
    granularity_parameters: Granularity | None = None,
    num_bits_activations: int = 8,
    num_bits_parameters: int = 8,
) -> Callable[..., None]:
    """Utility to perform basic quantizer initialization for testing modules."""
    if isinstance(activation_quantizers, str):
        activation_quantizers = ff.find_quantizers(quant_model, activation_quantizers)
    if isinstance(parameter_quantizers, str):
        parameter_quantizers = ff.find_quantizers(quant_model, parameter_quantizers)

    activation_quantizers.initialize(
        ff.nn.LinearQuantizer,
        num_bits=num_bits_activations,
        granularity=granularity_activations or ff.PerTensor(),
    )
    parameter_quantizers.initialize(
        ff.nn.LinearQuantizer,
        num_bits=num_bits_parameters,
        granularity=granularity_parameters or ff.PerTensor(),
    )

    def estimate_fn(*args: Any, **kwargs: Any) -> None:
        with ff.estimate_ranges(quant_model, ff.range_setting.smoothed_minmax):
            quant_model(*args, **kwargs)

    return estimate_fn
