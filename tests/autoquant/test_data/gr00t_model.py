# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from __future__ import annotations

"""GR00T-inspired E2E autoquant fixture.

This fixture mirrors a simplified GR00T-like spatial-alignment step where
`torch.nn.functional.pad` is used before downstream processing.

Full chain under test (pad-only):
  input -> Gr00tPadBlock(F.pad only) -> output

Scope note:
- This fixture intentionally isolates only `F.pad`.
- Additional GR00T-inspired layers will be added later in separate changes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    *_: object,
) -> torch.Tensor:
    emb = timesteps[:, None].float()
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Gr00tPadBlock(nn.Module):
    """GR00T relation: `diffusers.models.embeddings.Timesteps` relies on
    `get_timestep_embedding`, which uses `torch.nn.functional.pad` for
    odd embedding dimensions. The `get_timestep_embedding` must be defined outsideof the class.

    Single issue under test: `F.pad` conversion only.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        timesteps = x.flatten().to(dtype=torch.float32)
        return get_timestep_embedding(timesteps, 257)


def _gr00t_inherited_alias_impl(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return F.pad(x, (0, 1), mode="constant", value=0.0)


class Gr00tBaseInheritedOps(nn.Module):
    """Defines `forward` via inherited alias to module-level helper."""

    forward = _gr00t_inherited_alias_impl


class Gr00tInheritedScopeBlock(Gr00tBaseInheritedOps):
    """Child inherits aliased forward without rebinding helper name."""


class Gr00tModelInspired(nn.Module):
    """Top-level GR00T-referenced wrapper used by E2E harness."""

    def __init__(self):
        super().__init__()
        self.pad = Gr00tPadBlock()
        self.inherited_scope = Gr00tInheritedScopeBlock()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x)
        x = self.inherited_scope(x)
        return x


def get_model() -> torch.nn.Module:
    return Gr00tModelInspired()
