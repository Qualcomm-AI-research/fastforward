# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import torch


def is_close_to_rounding(tensor: torch.Tensor, eps: float = 0.0001) -> bool:
    """Return True if one of the tensor elements close to 0.5,
    which might generate a round error.
    """
    frac = torch.frac(tensor)
    return bool(torch.any(torch.abs(frac - 0.5) < eps))
