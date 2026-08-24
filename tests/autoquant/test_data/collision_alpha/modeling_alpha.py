# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""First half of the import-collision fixture.

Defines a module-level `residual_op` whose name is deliberately shared with
`collision_beta.modeling_beta.residual_op`. This mirrors
`eager_attention_forward` being defined by both the `ministral3` text decoder
and the `pixtral` vision tower in `transformers`.

Lives in its own package so the module-qualified alias autoquant allocates
(`collision_alpha_...` / `collision_beta_...`) identifies the originating
module, exactly as `pixtral_eager_attention_forward` does for the real model.
"""

import torch
import torch.nn as nn


def residual_op(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y


class AlphaBlock(nn.Module):
    """References `residual_op` as a value rather than calling it directly.

    Autoquant only inspects the callee of a call expression, so a helper that is
    merely referenced stays an import instead of being replaced by a quantized
    twin. That is the condition under which the name collision is observable.

    Note: comments and docstrings inside `forward` are copied verbatim into the
    generated module, so keep any explanation at class level to avoid coupling
    `import_collision_model.expected.py` to this file's prose.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        op = residual_op
        return op(x, x)
