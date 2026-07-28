# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import contextlib
import dataclasses

from typing import Iterator

import libcst


@dataclasses.dataclass
class TypeInfo:
    """Fallback for TypeInfo if Mypy is not installed."""


class MypyTypeProvider(libcst.VisitorMetadataProvider[TypeInfo]):
    """Fallback for MypyTypeProvider if Mypy is not installed."""


@contextlib.contextmanager
def mypy_call_scoped_cache() -> Iterator[None]:
    """No-op fallback for mypy_call_scoped_cache if Mypy is not installed."""
    yield
