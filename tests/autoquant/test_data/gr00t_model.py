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

import importlib.abc
import importlib.machinery
import linecache
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Gr00tFakeModuleFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    LEAF_MODULES = {
        "fake_diffusers.models.activations",
        "fake_torch.nn.modules.activation",
    }

    GELU_SOURCE = '''
import torch
import torch.nn as nn


class GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
'''

    @classmethod
    def _is_fake_package(cls, fullname: str) -> bool:
        return any(module.startswith(f"{fullname}.") for module in cls.LEAF_MODULES)

    def find_spec(
        self,
        fullname: str,
        path: object | None,
        target: object | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        is_leaf = fullname in self.LEAF_MODULES
        is_package = self._is_fake_package(fullname)
        if not is_leaf and not is_package:
            return None

        return importlib.machinery.ModuleSpec(
            fullname,
            self,
            is_package=is_package,
            origin=f"<fake-loader:{fullname}>",
        )

    def exec_module(self, module: object) -> None:
        module_dict = module.__dict__
        module_name = module_dict["__name__"]

        filename = f"<fake-loader:{module_name}>"
        module_dict["__file__"] = filename

        if self._is_fake_package(module_name):
            module_dict["__path__"] = []
            module_dict["__package__"] = module_name
            return

        module_dict["__package__"] = module_name.rpartition(".")[0]
        linecache.cache[filename] = (
            len(self.GELU_SOURCE),
            None,
            self.GELU_SOURCE.splitlines(keepends=True),
            filename,
        )
        exec(compile(self.GELU_SOURCE, filename, "exec"), module_dict)


if not any(isinstance(finder, _Gr00tFakeModuleFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _Gr00tFakeModuleFinder())

from fake_diffusers.models.activations import GELU as DiffusersGELU
from fake_torch.nn.modules.activation import GELU as TorchGELU



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


class Gr00tDuplicateGELUBlock(nn.Module):
    """Two layers share class name `GELU` but originate from different modules."""

    def __init__(self) -> None:
        super().__init__()
        self.diffusers_gelu = DiffusersGELU()
        self.torch_gelu = TorchGELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.diffusers_gelu(x)
        x = self.torch_gelu(x)
        return x


class Gr00tModelInspired(nn.Module):
    """Top-level GR00T-referenced wrapper used by E2E harness."""

    def __init__(self):
        super().__init__()
        self.pad = Gr00tPadBlock()
        self.inherited_scope = Gr00tInheritedScopeBlock()
        self.duplicate_gelu = Gr00tDuplicateGELUBlock()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x)
        x = self.inherited_scope(x)
        x = self.duplicate_gelu(x)
        return x


def get_model() -> torch.nn.Module:
    return Gr00tModelInspired()
