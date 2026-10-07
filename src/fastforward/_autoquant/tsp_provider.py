# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Backend-agnostic type-support provider for autoquant.

`fastforward.autoquantize(..., use_type_inference=True)` only controls
*whether* type inference is used; it never exposes *which* type checker
backs it. The backend is selected once, at import time, through the
`FASTFORWARD_TYPE_BACKEND` environment variable, so switching backends is a
developer/CI concern, not a public API concern.

Mypy uses the shared TSP metadata provider with an in-process client. A
remote backend supplies a TspClient launch command and registers its provider
and call scope here; source-range queries, metadata, and lifecycle management
remain in fastforward._autoquant.tsp.
"""

import contextlib
import dataclasses
import os

from typing import Callable

import libcst

from fastforward._autoquant.mypy import type_provider as _mypy_backend
from fastforward._autoquant.tsp.types import TypeInfo as TypeInfo

ENV_VAR = "FASTFORWARD_TYPE_BACKEND"
DEFAULT_BACKEND = "mypy"


@dataclasses.dataclass(frozen=True)
class _Backend:
    """A registered type-inference backend and its call-scoped cache."""

    provider: type[libcst.VisitorMetadataProvider[TypeInfo]]
    scoped_cache: Callable[[], contextlib.AbstractContextManager[None]]


_BACKENDS: dict[str, _Backend] = {
    "mypy": _Backend(
        provider=_mypy_backend.MypyTypeProvider,
        scoped_cache=_mypy_backend.mypy_call_scoped_cache,
    ),
}


def _select_backend() -> _Backend:
    name = os.environ.get(ENV_VAR, DEFAULT_BACKEND)
    try:
        return _BACKENDS[name]
    except KeyError:
        supported = ", ".join(sorted(_BACKENDS))
        msg = f"Unknown {ENV_VAR}={name!r}. Supported backends: {supported}."
        raise ValueError(msg) from None


_active_backend = _select_backend()

TypeProvider = _active_backend.provider
"""The `libcst.VisitorMetadataProvider[TypeInfo]` selected by `FASTFORWARD_TYPE_BACKEND`."""

type_inference_scope = _active_backend.scoped_cache
"""Call-scoped cache context manager for the selected backend."""
