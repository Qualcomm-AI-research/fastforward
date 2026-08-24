# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""E2E autoquant fixture for colliding imports in the generated module.

Mirrors the Ministral-3 failure mode: a multimodal checkpoint pairs a text
decoder with a vision tower, and both `transformers` modules define a
module-level helper named `eager_attention_forward`. Autoquant collected both
symbols and emitted two plain `from ... import eager_attention_forward` lines,
so the second binding silently shadowed the first and the generated decoder
called the vision tower's implementation.

Here `AlphaBlock` and `BetaBlock` come from sibling packages that each define
`residual_op`. As in the real model the helper is *referenced* rather than
called directly, so it stays an import instead of being replaced by a quantized
twin. The generated module must therefore bind both, under distinct names.
"""

import torch
import torch.nn as nn

from tests.autoquant.test_data.collision_alpha.modeling_alpha import AlphaBlock
from tests.autoquant.test_data.collision_beta.modeling_beta import BetaBlock


class ImportCollisionModel(nn.Module):
    """Wrapper pairing two blocks whose helper imports collide by name."""

    def __init__(self) -> None:
        super().__init__()
        self.alpha = AlphaBlock()
        self.beta = BetaBlock()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.beta(self.alpha(x))


def get_model() -> torch.nn.Module:
    return ImportCollisionModel()
