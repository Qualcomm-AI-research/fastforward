# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import dataclasses
import itertools
import logging
import pathlib

from collections.abc import Iterator
from typing import Any, Callable

import libcst

from typing_extensions import Self

from fastforward._import import fully_qualified_name
from fastforward._quantops import symtypes
from fastforward.nn import functional

logger = logging.getLogger(__name__)


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
        intermediate_quantizers: Names of intermediate quantizers expected as input to this operator.
        metadata: The metadata of the operator, if any.
    """

    identifier: str
    parameters: tuple[Parameter, ...]
    return_type: symtypes.Type | None
    intermediate_quantizers: tuple[str, ...] = ()
    metadata: OperatorMetadata | None = None

    @classmethod
    def from_spec(
        cls,
        spec: str,
        fallback: str,
        intermediate_quantizers: tuple[str, ...] = (),
        **metadata: Any,
    ) -> Self:
        try:
            # use libcst to parse function signature
            funcdef = libcst.parse_statement(f"def {spec}: pass")
            assert isinstance(funcdef, libcst.FunctionDef)
        except libcst.ParserSyntaxError as e:
            msg = f"'{spec}' is not a valid operator spec"
            raise ValueError(msg) from e

        identifier = funcdef.name.value
        return_type = (
            symtypes.type_from_cst_expression(funcdef.returns.annotation)
            if funcdef.returns is not None
            else None
        )
        params: list[Parameter] = []

        if funcdef.params.kwonly_params or funcdef.params.posonly_params:
            msg = (
                "Spec for '%s' uses keyword-only or positional-only arguments. "
                + "This is not supported and all arguments are treated as 'normal' arguments"
            )
            logger.warning(msg, (funcdef.name.value))

        for param in itertools.chain(
            funcdef.params.params, funcdef.params.kwonly_params, funcdef.params.posonly_params
        ):
            name = param.name.value
            if param.annotation is None:
                msg = "All parameters must have a valid annotation"
                raise ValueError(msg)
            param_type = symtypes.type_from_cst_expression(param.annotation.annotation)
            default = _get_default_value(param.default) if param.default is not None else None

            params.append(Parameter(name=name, param_type=param_type, default_value=default))

        return cls(
            identifier,
            tuple(params),
            return_type=return_type,
            intermediate_quantizers=intermediate_quantizers,
            metadata=OperatorMetadata(fallback=fallback, **metadata),
        )

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
        return self._bind_partial(args, kwargs, include_quantization_params=True)

    def _bind_partial(
        self, args: Any, kwargs: Any, include_quantization_params: bool
    ) -> Iterator[tuple[Parameter, Any]]:
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

        Args:
            args: positional arguments to bind to this operator's signature.
            kwargs: keyword arguments to bind to this operator's signature.
            include_quantization_params: If True, adds quantization-related
                parameters (strict_quantization, output_quantizer, and intermediate
                quantizers) to the signature before binding. Otherwise, bind arguments
                to the original spec.

        Returns:
            Iterator of `Parameter` and argument pairs. The argument element is
            an argument provided to this method.
        """
        params: dict[str, Parameter] = {}
        if include_quantization_params:
            strict_quant_param = Parameter(symtypes.Bool, "strict_quantization", "None")
            params[strict_quant_param.name] = strict_quant_param

            if self.num_output_quantizers > 1:
                raise NotImplementedError("Support for multiple quantized outputs is pending")
            if self.num_output_quantizers == 1:
                quantizer_param = Parameter(symtypes.Quantizer, "output_quantizer", "None")
                params[quantizer_param.name] = quantizer_param

            for intermediate in self.intermediate_quantizers:
                quantizer_param = Parameter(symtypes.Quantizer, intermediate, "None")
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

    def validate_arguments(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
        """Validate if operator can be called with given arguments.

        This method only validates that the number of provided arguments and
        keyword arguments matches the operator's expectations. No value or type
        checking is performed.

        Args:
            args: List of positional arguments to validate.
            kwargs: Dictionary of keyword arguments to validate.

        Returns:
            bool: True if the operator can be called with the provided arguments,
                False otherwise.
        """
        try:
            binding = self._bind_partial(args, kwargs, include_quantization_params=False)
            bound_params = {param.name for param, _ in binding}
        except (ValueError, TypeError):
            return False

        if len(bound_params) != len(args) + len(kwargs):
            return False
        for param in self.parameters:
            if param.name not in bound_params and param.default_value is None:
                return False

        return True


def _get_default_value(expr: libcst.BaseExpression) -> str:
    match expr:
        case libcst.SimpleString():
            return str(expr.evaluated_value)
        case _:
            return libcst.Module([]).code_for_node(expr)
