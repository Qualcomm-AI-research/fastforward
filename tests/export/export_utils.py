# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Any, TypeAlias

import fastforward as ff
import torch

from fastforward.quantization.granularity import Granularity
from fastforward.quantization.quant_init import QuantizerCollection

QuantizedModelFixture: TypeAlias = tuple[torch.nn.Module, QuantizerCollection, QuantizerCollection]


def activate_quantizers(
    quant_model: torch.nn.Module,
    args: torch.Tensor | tuple[torch.Tensor, ...],
    activation_quantizers: QuantizerCollection,
    parameter_quantizers: QuantizerCollection,
    param_granularity: Granularity = ff.PerTensor(),
    kwargs: None | dict[str, Any] = None,
) -> None:
    activation_quantizers.initialize(ff.nn.LinearQuantizer, num_bits=8)
    parameter_quantizers.initialize(
        ff.nn.LinearQuantizer, num_bits=8, granularity=param_granularity
    )

    kwargs = kwargs or {}

    if not isinstance(args, tuple):
        args = (args,)

    with ff.estimate_ranges(quant_model, ff.range_setting.smoothed_minmax):
        quant_model(*args, **kwargs)
