# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Second half of the import-collision fixture.

Defines a module-level `residual_op` whose name is deliberately shared with
`collision_alpha.modeling_alpha.residual_op`, with different semantics so that
picking the wrong one changes the computed result.

See `collision_alpha.modeling_alpha` for the full rationale.
"""

import torch
import torch.nn as nn


def residual_op(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x - y


class BetaBlock(nn.Module):
    """Counterpart to `collision_alpha.modeling_alpha.AlphaBlock`.

    See that class for why `residual_op` is referenced rather than called, and
    why explanation belongs at class level rather than inside `forward`.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        op = residual_op
        return op(x, x)
