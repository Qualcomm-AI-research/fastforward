# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import dataclasses
import pathlib

from typing import Any, Callable, Optional

from fastforward._quantops import symtypes


@dataclasses.dataclass(frozen=True)
class Parameter:
    param_type: symtypes.Type
    name: str
    default_value: Optional[str]


@dataclasses.dataclass
class OperatorMetadata:
    fallback: str
    specification_file: pathlib.Path | None = None
    line_number: int | None = None
    cast_output: Optional[str] = None
    dispatch_op: Callable[..., Any] | None = None


@dataclasses.dataclass(frozen=True)
class Operator:
    identifier: str
    parameters: tuple[Parameter, ...]
    return_type: symtypes.Type | None
    metadata: Optional[OperatorMetadata] = None
