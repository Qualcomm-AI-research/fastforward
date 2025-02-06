# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import abc
import dataclasses
import functools

from typing import Any, Callable, Generic, TypeVar

import torch

from typing_extensions import Self, override

import fastforward as ff

from fastforward.common import maybe_tensor_apply
from fastforward.quantized_tensor import QuantizedTensor


@dataclasses.dataclass
class QuantizationParameters:
    """Container for quantization parameters.

    Each quantization expects a specific `QuantizationParamers`.
    """

    def with_changes(self, **changes: Any) -> Self:
        """Replace given keywords and return a new `QuantizationParameters`."""
        return dataclasses.replace(self, **changes)

    def _apply(self, fn: Callable[[Any], Any]) -> Self:
        """Apply `fn` to every parameter.

        Create a new `QuantizationParameters` using the results of each application.
        """
        new_values = {k: fn(v) for k, v in dataclasses.asdict(self).items()}
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

        Implementors are expected to implement the quantize function that takes
        input data and a `QuantizationParameters` object and return a
        `QuantizedTensor` that is `data` quantized following `params`.
        """

    @classmethod
    @abc.abstractmethod
    def dequantize(cls, data: torch.Tensor, params: QuantParams_co) -> torch.Tensor:  # type: ignore[misc]
        """Abstract method for dequantization.

        Implementors are expected to implement the dequantize function that takes
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

        current_params = dataclasses.asdict(self.quantization_params)
        new_params = dataclasses.asdict(ctx.quantization_params)

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
