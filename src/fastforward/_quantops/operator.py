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
    """Represents a Python function signature and metadata.

    This representation is used to represent Python functions, quantized
    version of functions, and collect metadata related for dispatching and
    autoquant.

    Attributes:
        identifier: The identifier of the operator.
        parameters: The parameters of the operator.
        return_type: The return type of the operator, if any.
        metadata: The metadata of the operator, if any.
    """

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

    def bind_partial(self, *args: Any, **kwargs: Any) -> Iterator[tuple[Parameter, Any]]:
        """Bind arguments and keyword arguments to operator signature.

        Given a tuple of `args` and dictionary of `kwargs`, assign each
        provided argument to a parameter in the operator signature, if
        possible.

        If not all arguments can be assigned to a unique parameter, a
        `TypeError` or `ValueError` is raised. This function succeeds if an
        insufficient number of arguments is supplied and returns a partial
        binding.

        Note that this function does not perform any form of type checking. It
        simply binds argument to parameters based on the position and use of a
        keyword argument.

        Returns:
            Iterator of `Parameter` and argument pairs. The argument element is
            an argument provided to this method.
        """
        strict_quant_param = Parameter(symtypes.Bool, "strict_quantization", "None")
        params: dict[str, Parameter] = {strict_quant_param.name: strict_quant_param}

        if self.num_output_quantizers > 1:
            raise NotImplementedError("Support for multiple quantized outputs is pending")
        if self.num_output_quantizers == 1:
            quantizer_param = Parameter(symtypes.Quantizer, "output_quantizer", "None")
            params[quantizer_param.name] = quantizer_param

        for param in reversed(self.parameters):
            params[param.name] = param
        param_names = {k for k in params.keys()}

        bound_params: dict[str, tuple[Parameter, Any]] = {}
        len_args = len(args) + len(kwargs)
        if len(params) < len_args:
            msg = f"{self.identifier}() takes {len(params)} arguments but {len_args} were given"
            raise ValueError(msg)

        for arg in args:
            name, param = params.popitem()
            bound_params[name] = (param, arg)
        for kw, arg in kwargs.items():
            if kw not in param_names:
                msg = f"{self.identifier}() got an unexpected keyword '{kw}'"
                raise TypeError(msg)
            if kw not in params:
                msg = f"{self.identifier}() got multiple values for argument '{kw}'"
                raise TypeError(msg)
            param = params.pop(kw)
            bound_params[kw] = (param, arg)

        yield from bound_params.values()
