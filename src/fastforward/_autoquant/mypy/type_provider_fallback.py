# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import contextlib

from typing import Iterator

import libcst

from fastforward._autoquant.tsp.context import module_context
from fastforward._autoquant.tsp.types import TypeInfo as TypeInfo

mypy_module_context = module_context


class MypyTypeProvider(libcst.VisitorMetadataProvider[TypeInfo]):
    """Fallback for MypyTypeProvider if Mypy is not installed."""


@contextlib.contextmanager
def mypy_call_scoped_cache() -> Iterator[None]:
    """No-op fallback for mypy_call_scoped_cache if Mypy is not installed."""
    yield
