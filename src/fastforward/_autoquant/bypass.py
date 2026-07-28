# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Centralized bypass policy for AutoQuant.

Some control/override-style torch APIs must be preserved as-is in the generated
code: they are not tensor operations, they do not have a quantized counterpart,
and descending into their implementation produces noisy wrappers that obscure
the user's original code.

Extending the policy is a one-line addition to `_BYPASS_MIN_TORCH_VERSION`.
"""

import importlib
import warnings

from typing import Any

import libcst
import libcst.helpers
import torch

from packaging.version import Version

# Fully qualified names of callables that AutoQuant must leave untouched, mapped
# to the first torch version that provides them. An entry that the installed
# torch predates is skipped: it cannot appear in user code on that version.
_BYPASS_MIN_TORCH_VERSION: dict[str, str] = {
    "torch.utils.checkpoint.checkpoint": "2.4",
    "torch.overrides.handle_torch_function": "2.4",
    "torch.compiler.is_exporting": "2.7",
    "torch._check_with": "2.4",
}

_BYPASS_QUALIFIED_NAMES: frozenset[str] = frozenset(_BYPASS_MIN_TORCH_VERSION)


def _resolve_qualified_name(qualified_name: str) -> Any:
    """Resolve a dotted name to the object it denotes.

    Imports the longest importable prefix before attribute access, because a
    submodule (e.g. `torch.utils.checkpoint`) only becomes an attribute of its
    parent package once something imports it.

    Raises:
        ImportError: if no prefix of `qualified_name` is an importable module.
        AttributeError: if a prefix imports but the remaining attributes are absent.
    """
    parts = qualified_name.split(".")
    for split in range(len(parts), 0, -1):
        try:
            obj: Any = importlib.import_module(".".join(parts[:split]))
        except ModuleNotFoundError:
            continue
        for part in parts[split:]:
            obj = getattr(obj, part)
        return obj
    msg = f"No prefix of {qualified_name!r} is an importable module"
    raise ImportError(msg)


def _resolve_bypass_callables() -> frozenset[Any]:
    installed_torch = Version(torch.__version__.split("+", 1)[0])
    resolved: set[Any] = set()
    for qualified_name, min_torch_version in _BYPASS_MIN_TORCH_VERSION.items():
        if installed_torch.release < Version(min_torch_version).release:
            continue
        try:
            resolved.add(_resolve_qualified_name(qualified_name))
        except (ImportError, AttributeError) as e:
            warnings.warn(
                f"AutoQuant bypass entry {qualified_name!r} could not be resolved "
                f"({type(e).__name__}: {e}); calls to this op will not be bypassed "
                f"via callable identity."
            )
    return frozenset(resolved)


_BYPASS_CALLABLES: frozenset[Any] = _resolve_bypass_callables()


def is_bypassed_callable(ref: Any) -> bool:
    """Return True when `ref` is a callable that AutoQuant must not rewrite."""
    return ref in _BYPASS_CALLABLES


def is_bypassed_call_syntax(func_expr: libcst.BaseExpression) -> bool:
    """Return True when the callsite target syntactically matches a bypass op.

    Matches by dotted-name (e.g. `torch.utils.checkpoint.checkpoint`).
    Aliased imports are covered by `is_bypassed_callable` at points that have
    scope resolution.
    """
    full_name = libcst.helpers.get_full_name_for_node(func_expr)
    return full_name is not None and full_name in _BYPASS_QUALIFIED_NAMES
