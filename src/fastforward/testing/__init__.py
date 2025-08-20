# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import random

import numpy as np
import torch

from . import autoquant as autoquant
from . import metrics as metrics
from . import string as string


def is_close_to_rounding(tensor: torch.Tensor, eps: float = 0.0001) -> bool:
    """Return True if one of the tensor elements close to 0.5.

    When this holds, further processing might generate a round error.
    """
    frac = torch.frac(tensor)
    return bool(torch.any(torch.abs(frac - 0.5) < eps))


def seed_prngs(random_seed: int) -> None:
    """Seeds the common PRNGs."""
    assert 0 <= random_seed <= 2**64
    torch.manual_seed(random_seed)
    np.random.seed(random_seed % 2**32)
    random.seed(random_seed)
