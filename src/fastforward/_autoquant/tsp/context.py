# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Source identity shared by all type-inference backends."""

import contextlib
import contextvars

from typing import Iterator

_module_context: contextvars.ContextVar[tuple[str, str | None]] = contextvars.ContextVar(
    "fastforward_autoquant_type_module_context", default=("_ff_evaluation_module__", None)
)


def current_module() -> tuple[str, str | None]:
    """Return the source module name and original path, when available."""
    return _module_context.get()


@contextlib.contextmanager
def module_context(module_name: str, module_path: str | None) -> Iterator[None]:
    """Preserve source identity and restore it after nested analysis or errors."""
    token = _module_context.set((module_name, module_path))
    try:
        yield
    finally:
        _module_context.reset(token)
