# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""
!!! experimental
    Please be aware that autoquant is an experimental feature. Use it with caution and
    expect changes as we continue the development of the feature.

    We encourage you to report any issues or feature requests.
"""  # noqa: D205, D212

import torch

from fastforward._autoquant.autoquant import autoquant_with_defaults
from fastforward._quantops import optable


def autoquantize(
    module: torch.nn.Module, operator_table: optable.OperatorTable | None = None
) -> None:
    """Create Python source code for quantized version of `module`.

    Args:
        module: The module to quantize.
        operator_table: The operator table that defines the non-quantized to
            quantized operator mapping.

    """
    print(autoquant_with_defaults(module, operator_table).build().code)
