# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import torch

from torch import nn


class SAM3ModelInspired(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def get_model() -> torch.nn.Module:
    return SAM3ModelInspired()