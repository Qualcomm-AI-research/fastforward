# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#
import torch

from fastforward._autoquant.autoquant import autoquant_with_defaults
from fastforward._quantops import optable


def autoquantize(
    module: torch.nn.Module, operator_table: optable.OperatorTable | None = None
) -> None:
    """Create Python source code for quantized version of `module`.

    Note:
        This functionality is experimental and currently under active
        development.

    Args:
        module: The module to quantize.
        operator_table: The operator table that defines the non-quantized to
            quantized operator mapping.

    """
    print(autoquant_with_defaults(module, operator_table).build().code)
