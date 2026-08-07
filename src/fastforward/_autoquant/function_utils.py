# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import logging
import types

from typing import Any, Callable, cast

import torch

logger = logging.getLogger(__name__)


def unwrap_method_owner_member(member: Any) -> Any:
    if isinstance(member, (classmethod, staticmethod)):
        return member.__func__
    return member


def resolve_method_owner_and_name(
    module_type: type[torch.nn.Module],
    accessed_name: str,
    func: Callable[..., Any],
) -> tuple[type[torch.nn.Module], str]:
    """Resolve class scope/member name for source lookup of method tasks.

    Methods can be inherited or aliased (for example ``forward = helper_func``),
    where ``func.__name__`` differs from the call-site member name. Resolve by
    member identity across the MRO so source is loaded from the defining scope.
    """
    for owner in module_type.__mro__:
        owner_member = unwrap_method_owner_member(owner.__dict__.get(accessed_name, None))
        if owner_member is func:
            return owner, accessed_name

    for owner in module_type.__mro__:
        for name, member in owner.__dict__.items():
            if unwrap_method_owner_member(member) is func:
                return owner, name

    return module_type, accessed_name


def resolve_function_source_member_name(
    module: types.ModuleType,
    accessed_name: str,
    func: Callable[..., Any],
) -> str:
    """Resolve module member name for source lookup of helper-function tasks.

    Helpers can be referenced through aliases (e.g. ``mod.alias(...)`` where
    ``mod.alias is mod.actual_func``). Source lookup should follow the canonical
    defining name when available while preserving call-site alias separately.
    """
    module_dict = getattr(module, "__dict__", None)
    if isinstance(module_dict, dict):
        if module_dict.get(accessed_name, None) is func:
            return accessed_name

        for name, member in module_dict.items():
            if member is func:
                return cast(str, name)

    return accessed_name
