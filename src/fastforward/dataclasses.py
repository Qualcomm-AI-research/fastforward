# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import dataclasses

from typing import Any


def nocopy_asdict(obj: Any) -> dict[str, Any]:
    """Create a dictionary of dataclass fields without copying values."""
    if not dataclasses.is_dataclass(obj):
        msg = "nocopy_asdict() should be called on dataclass instances"
        raise TypeError(msg)

    return {field.name: getattr(obj, field.name) for field in dataclasses.fields(obj)}
