# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import abc
import dataclasses
import functools
import inspect
import types

from collections.abc import Iterator
from typing import Any, Callable, Generic, TypeVar

import torch

from typing_extensions import Self, override

import fastforward as ff

from fastforward.common import maybe_tensor_apply
from fastforward.quantized_tensor import QuantizedTensor


@dataclasses.dataclass
class QuantizationParameters:
    """Container for quantization parameters.

    Each quantization expects a specific `QuantizationParameters`.
    """

    def with_changes(self, **changes: Any) -> Self:
        """Replace given keywords and return a new `QuantizationParameters`."""
        return dataclasses.replace(self, **changes)

    def _apply(self, fn: Callable[[Any], Any]) -> Self:
        """Apply `fn` to every parameter.

        Create a new `QuantizationParameters` using the results of each application.
        """
        new_values = {k: fn(v) for k, v in ff.dataclasses.nocopy_asdict(self).items()}
        return type(self)(**new_values)

    @override
    def __format__(self, format_spec: str, /) -> str:
        return repr(self)


QuantParams = TypeVar("QuantParams", bound=QuantizationParameters)
QuantParams_co = TypeVar("QuantParams_co", bound=QuantizationParameters, covariant=True)


class QuantizationFunction(Generic[QuantParams_co], abc.ABC):
    """Base class for QuantizationFunctions."""

    # mypy does not allow covariant parameters. It's a coarse way of not
    # allowing mutable covariant functions. This is not a problem in this case,
    # hence the error (misc) is suppressed.
    @classmethod
    @abc.abstractmethod
    def quantize(cls, data: torch.Tensor, params: QuantParams_co) -> QuantizedTensor:  # type: ignore[misc]
        """Abstract method for quantization.

        Implementers are expected to implement the quantize function that takes
        input data and a `QuantizationParameters` object and return a
        `QuantizedTensor` that is `data` quantized following `params`.
        """

    @classmethod
    @abc.abstractmethod
    def dequantize(cls, data: torch.Tensor, params: QuantParams_co) -> torch.Tensor:  # type: ignore[misc]
        """Abstract method for dequantization.

        Implementers are expected to implement the dequantize function that takes
        quantized data and a `QuantizationParameters` object and return a
        tensor that represents the dequantized data.
        """


@dataclasses.dataclass(frozen=True)
class QuantizationContext(Generic[QuantParams_co]):
    """A container for `QuantizationFunction`s and corresponding parameters.

    Together these form the quantization context and contain all information to
    perform quantization and/or dequantization.
    """

    quantization_fn: type[QuantizationFunction[QuantParams_co]]
    quantization_params: QuantParams_co

    def with_changes(
        self,
        quantization_fn: type[QuantizationFunction[QuantParams_co]] | None = None,
        **changes: Any,
    ) -> Self:
        """Create new context replacing `quantization_fn` or parameters.

        Args:
            quantization_fn: `QuantizationFunction` for new
                `QuantizationContext`. If `None`, use the current function.
            **changes: Fields to change in `QuantizationParams`.

        Returns:
            New `QuantizationContext`.
        """
        params = self.quantization_params.with_changes(**changes)
        context_changes: dict[str, Any] = {"quantization_params": params}
        if quantization_fn is not None:
            context_changes["quantization_fn"] = quantization_fn
        return dataclasses.replace(self, **context_changes)

    def _apply(self, fn: Callable[[Any], Any]) -> Self:
        """Apply `fn` to every parameter.

        Create a new `QuantizationParameters` using the results of each application.
        """
        params = self.quantization_params._apply(fn)
        return dataclasses.replace(self, quantization_params=params)

    def clone_parameters(self) -> Self:
        """Clone each tensor in `quantization_params`.

        Return a new context with updated parameters.
        """
        fn = functools.partial(maybe_tensor_apply, fn=torch.clone)
        return self._apply(fn)

    def detach_parameters(self) -> Self:
        """Detach each tensor in `quantization_params`.

        Return a new context with updated parameters.
        """
        fn = functools.partial(maybe_tensor_apply, fn=torch.detach)
        return self._apply(fn)

    def contiguous_parameters(self) -> Self:
        """Update each tensor in `quantization_params` to be contiguous."""
        fn = functools.partial(maybe_tensor_apply, fn=torch.Tensor.contiguous)
        ctx = self._apply(fn)

        current_params = ff.dataclasses.nocopy_asdict(self.quantization_params)
        new_params = ff.dataclasses.nocopy_asdict(ctx.quantization_params)

        for k in current_params:
            if current_params[k] is not new_params[k]:
                return ctx
        return self

    def to(self, device: torch.device | str) -> Self:
        """Update each tensor in `quantization_params` to `device`."""

        def _to_device(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.to(device=device)

        fn = functools.partial(maybe_tensor_apply, fn=_to_device)
        return self._apply(fn)

    def attach(self, data: torch.Tensor) -> QuantizedTensor:
        """Attach to `data` to this context.

        Creates a `QuantizedTensor` that uses data as raw_data and this context
        as quantization context.
        """
        if ff.get_export_mode():
            # The QuantizedTensor can be used once issue #166 is resolved (waiting for changes
            # on the torch side).
            return self.quantization_fn.dequantize(data, self.quantization_params)  # type: ignore[return-value]
        return QuantizedTensor(data, self)


@dataclasses.dataclass
class _QuantParamsWithDict(QuantizationParameters, abc.ABC):
    @abc.abstractmethod
    def quantize_params(self) -> dict[str, Any]: ...

    @abc.abstractmethod
    def dequantize_params(self) -> dict[str, Any]: ...


_QP = TypeVar("_QP", bound=_QuantParamsWithDict)


class _ImplicitQuantizationFunction(Generic[_QP], QuantizationFunction[_QP]):
    """BaseType for generated QuantizationFunction using `create_quantization_function`."""

    _quantize_impl: Callable[..., torch.Tensor]
    _dequantize_impl: Callable[..., torch.Tensor]

    def __init_subclass__(cls) -> None:
        if not hasattr(cls, "_quantize_impl"):
            msg = f"A subclass of {cls.__name__} must define '_quantize_impl'"
            raise TypeError(msg)
        if not hasattr(cls, "_dequantize_impl"):
            msg = f"A subclass of {cls.__name__} must define '_dequantize_impl'"
            raise TypeError(msg)
        return super().__init_subclass__()

    @classmethod
    def quantize(cls, data: torch.Tensor, params: _QP) -> QuantizedTensor:
        """Quantize `data` with `params`. Returns a `QuantizedTensor`."""
        quantized_data = cls._quantize_impl(data, **params.quantize_params())
        context = QuantizationContext(cls, params)
        return ff.QuantizedTensor(quantized_data, context)

    @classmethod
    def dequantize(cls, data: torch.Tensor, params: _QP) -> torch.Tensor:
        """Dequantize `data` with `params`. Returns a `torch.Tensor`."""
        return cls._dequantize_impl(data, **params.dequantize_params())


def create_quantization_function(
    cls_name: str,
    quantize: Callable[..., torch.Tensor],
    dequantize: Callable[..., torch.Tensor],
) -> tuple[
    type[_QuantParamsWithDict],
    _ImplicitQuantizationFunction[_QuantParamsWithDict],
    Callable[..., QuantizedTensor],
]:
    """Generate types and function for quantizer.

    Given a `quantize` and `dequantize` function, create a new
    `QuantizationFunction` and associated `QuantizationParameters` to use
    within fastforward. Moverover, a new helper function is returned that calls
    the newly created `QuantizationFunction` with appropriate bookkeeping.

    The newly produced `QuantizationFunction` and `QuantizationParameters`
    subclasses are deduced from the signature of `quantize` and `dequantize`.
    For this reason, any annotation or default value of parameters with the
    same name between quantize and dequantze must match.

    Args:
        cls_name: The name of the newly created `QuantizationFunction` subclass.
        quantize: A quantization function. Must return a `torch.Tensor` that is
            then wrapped as a `QuantizedTensor`.
        dequantize: A dequantization function. Must return a `torch.Tensor`.

    """
    QuantParams = _make_quantization_parameters_type(cls_name, quantize, dequantize)

    QuantFunc: type[_ImplicitQuantizationFunction[_QuantParamsWithDict]] = types.new_class(
        cls_name,
        (_ImplicitQuantizationFunction[QuantParams],),  # type: ignore[valid-type]
        {},
        lambda ns: ns.update({
            "_quantize_impl": staticmethod(quantize),
            "_dequantize_impl": staticmethod(dequantize),
        }),
    )

    def quantize_func(data: torch.Tensor, **kwargs: Any) -> QuantizedTensor:
        """Quantize `data`."""
        return QuantFunc.quantize(data, QuantParams(**kwargs))

    return QuantParams, QuantFunc, quantize_func  # type: ignore[return-value]


def _make_quantization_parameters_type(
    cls_name: str,
    quantize: Callable[..., torch.Tensor],
    dequantize: Callable[..., torch.Tensor],
) -> type[_QuantParamsWithDict]:
    """Create a `QuantizationParameters` type for the quantize and dequantize signatures."""
    quantize_params = {
        param: (annotation, field)
        for param, annotation, field in _parameters_from_function(quantize)
    }
    dequantize_params = {
        param: (annotation, field)
        for param, annotation, field in _parameters_from_function(dequantize)
    }

    for param, (dequantize_annotation, dequantize_field) in dequantize_params.items():
        if param not in quantize_params:
            continue

        quantize_annotation, quantize_field = quantize_params[param]
        if quantize_annotation != dequantize_annotation:
            msg = (
                "The type annotation for '{param}' must be the same in both the "
                + "quantize and dequantize function"
            )
            raise TypeError(msg)
        if quantize_field.default != dequantize_field.default:
            msg = (
                "The default value for '{param}' must be the same in both the "
                + "quantize and dequantize function"
            )
            raise ValueError(msg)

    fields = (
        (param, annotation, field)
        for param, (annotation, field) in {**quantize_params, **dequantize_params}.items()
    )

    def _quantize_params(self: _QuantParamsWithDict) -> dict[str, Any]:
        return {name: getattr(self, name) for name in quantize_params}

    def _dequantize_params(self: _QuantParamsWithDict) -> dict[str, Any]:
        return {name: getattr(self, name) for name in dequantize_params}

    return dataclasses.make_dataclass(
        cls_name=f"{cls_name}Params",
        fields=fields,
        bases=(_QuantParamsWithDict,),
        namespace={"quantize_params": _quantize_params, "dequantize_params": _dequantize_params},
    )


def _parameters_from_function(
    quant_func: Callable[..., torch.Tensor],
) -> Iterator[tuple[str, Any, dataclasses.Field[Any]]]:
    signature = inspect.signature(quant_func)
    params = iter(signature.parameters.values())
    next(params)  # skip the data param
    for param in params:
        annotation = param.annotation if param.annotation is not inspect.Parameter.empty else Any
        field_args = {}
        if param.default is not inspect.Parameter.empty:
            field_args["default"] = param.default
        if param.kind not in (param.KEYWORD_ONLY, param.POSITIONAL_OR_KEYWORD):
            msg = (
                "All parameters must be keyword only or positional or keyword parameters "
                + f"{param.name} is {param.kind.description}"
            )
            raise TypeError(msg)
        yield param.name, annotation, dataclasses.field(**field_args)
