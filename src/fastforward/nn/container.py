# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import torch

from fastforward.nn import QuantizedModule


class QuantizedSequential(QuantizedModule, torch.nn.Sequential):
    """Quantized implementation of torch.nn.Sequential."""


class QuantizedModuleList(QuantizedModule, torch.nn.ModuleList):
    """Quantized implementation of torch.nn.ModuleList."""


class QuantizedModuleDict(QuantizedModule, torch.nn.ModuleDict):
    """Quantized implementation of torch.nn.ModuleDict."""


class QuantizedParameterList(QuantizedModule, torch.nn.ParameterList):
    """Quantized implementation of torch.nn.ParameterList."""


class QuantizedParameterDict(QuantizedModule, torch.nn.ParameterDict):
    """Quantized implementation of torch.nn.ParameterDict."""
