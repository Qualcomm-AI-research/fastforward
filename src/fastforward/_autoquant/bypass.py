# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Centralized bypass policy for AutoQuant.

Some control/override-style torch APIs must be preserved as-is in the generated
code: they are not tensor operations, they do not have a quantized counterpart,
and descending into their implementation produces noisy wrappers that obscure
the user's original code.

Extending the policy is a one-line addition to `_BYPASS_QUALIFIED_NAMES`.
"""

import importlib
import warnings

from typing import Any

import libcst
import libcst.helpers

# Fully qualified names of callables that AutoQuant must leave untouched. The
# leaf function is resolved lazily at import time so that missing attributes on
# older torch versions do not break AutoQuant.
_BYPASS_QUALIFIED_NAMES: frozenset[str] = frozenset({
    "torch.utils.checkpoint.checkpoint",
    "torch.overrides.handle_torch_function",
    "torch.compiler.is_exporting",
    "torch._check_with",
})


def _resolve_bypass_callables() -> frozenset[Any]:
    resolved: set[Any] = set()
    for qualified_name in _BYPASS_QUALIFIED_NAMES:
        root, *parts = qualified_name.split(".")
        try:
            obj: Any = importlib.import_module(root)
            for part in parts:
                obj = getattr(obj, part)
        except (ImportError, AttributeError) as e:
            warnings.warn(
                f"AutoQuant bypass entry {qualified_name!r} could not be resolved "
                f"({type(e).__name__}: {e}); calls to this op will not be bypassed "
                f"via callable identity."
            )
            continue
        resolved.add(obj)
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
