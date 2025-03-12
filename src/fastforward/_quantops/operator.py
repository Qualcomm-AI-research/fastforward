# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import dataclasses
import pathlib

from collections.abc import Iterator
from typing import Any, Callable

from fastforward._import import fully_qualified_name
from fastforward._quantops import symtypes
from fastforward.nn import functional


@dataclasses.dataclass(frozen=True)
class Parameter:
    param_type: symtypes.Type
    name: str
    default_value: str | None

    @property
    def quantized(self) -> bool:
        return symtypes.QuantizedTensor in self.param_type


@dataclasses.dataclass
class OperatorMetadata:
    fallback: str
    specification_file: pathlib.Path | None = None
    line_number: int | None = None
    cast_output: str | None = None
    dispatch_op: Callable[..., Any] | None = None


@dataclasses.dataclass(frozen=True)
class Operator:
    identifier: str
    parameters: tuple[Parameter, ...]
    return_type: symtypes.Type | None
    metadata: OperatorMetadata | None = None

    def dispatch_op(self) -> Callable[..., Any] | None:
        if metadata := self.metadata:
            return metadata.dispatch_op
        return None

    def dispatch_qualified_name(self) -> str | None:
        """Returns the fully qualified name of the dispatch op.

        If no dispatch op is set, returns None
        """
        if not (dispatch_op := self.dispatch_op()):
            return None
        if getattr(functional, dispatch_op.__name__, None) == dispatch_op:
            return f"fastforward.nn.functional.{dispatch_op.__name__}"
        return fully_qualified_name(dispatch_op)

    @property
    def returns_quantized(self) -> bool:
        return_type = self.return_type
        return return_type is not None and symtypes.QuantizedTensor in return_type

    @property
    def num_output_quantizers(self) -> int:
        return 1 if self.returns_quantized else 0
