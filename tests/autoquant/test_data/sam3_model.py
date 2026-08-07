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

    In copied methods, bare super() must be anchored to the generated class so
    that MRO resolution reaches the quantized counterpart of SAM3SuperParent, 
    which the generated class inherits from ahead of
    SAM3SuperChild itself.
    """

    def forward(self, *, child_kw):
        return super().forward(parent_kw=child_kw)


class SAM3AlreadyExplicitSuper(SAM3SuperParent):
    """Tests that an already-explicit super(ClassName, self) call is re-anchored too.

    An explicit call that anchors to the owning class is equivalent to a bare
    super() call once the method is copied, and must be re-anchored to the
    generated class for the same reason.
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
