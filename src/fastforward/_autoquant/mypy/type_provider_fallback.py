# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import dataclasses

import libcst


@dataclasses.dataclass
class TypeInfo:
    """Fallback for TypeInfo if Mypy is not installed."""


class MypyTypeProvider(libcst.VisitorMetadataProvider[TypeInfo]):
    """Fallback for MypyTypeProvider if Mypy is not installed."""
