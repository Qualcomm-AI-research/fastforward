# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import torch

from torch import nn


class SAM3SuperParent(nn.Module):
    """Base class used as the super() call target in subclass tests."""

    def forward(self, *, parent_kw):
        return parent_kw


class SAM3SuperChild(SAM3SuperParent):
    """Tests that a bare super() call is rewritten to super(ClassName, self).

    In copied methods, bare super() must be anchored to the original class name
    so that MRO resolution works correctly inside the generated quantized class,
    which inherits from both the quantized mixin and SAM3SuperChild.
    """

    def forward(self, *, child_kw):
        return super().forward(parent_kw=child_kw)


class SAM3AlreadyExplicitSuper(SAM3SuperParent):
    """Tests that an already-explicit super(ClassName, self) call is left unchanged.

    If the source method already uses the two-argument form, the autoquant pass
    must not double-rewrite it or otherwise alter it.
    """

    def forward(self, *, child_kw):
        return super(SAM3AlreadyExplicitSuper, self).forward(parent_kw=child_kw)


class SAM3ModelInspired(nn.Module):
    """Showcases extracted from SAM3 model."""
    def __init__(self) -> None:
        super().__init__()
        self.super_child = SAM3SuperChild()
        self.explicit_super = SAM3AlreadyExplicitSuper()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.super_child(child_kw=x)
        _ = self.explicit_super(child_kw=x)
        return x


def get_model() -> torch.nn.Module:
    return SAM3ModelInspired()
