# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

"""Collection of types, protocols and base implementations for range setting methods."""

import abc
import contextlib

from typing import (
    Any,
    Callable,
    Generator,
    Generic,
    Iterator,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    runtime_checkable,
)

import torch

from typing_extensions import override

from fastforward.quantization import granularity
from fastforward.quantized_tensor import QuantizedTensor


@runtime_checkable
class RangeSettable(Protocol):
    """Interface for range settable quantization modules.

    The quantization parameters for modules that implement this interface can
    be specified through a quantization range. This is a generalization over
    particular quantizer parameterization and is used by range setting methods.
    """

    @property
    def granularity(Protocol) -> granularity.Granularity:
        """The quantization granularity for a quantizer.

        The granularity specifies which part of the input tensor is quantized
        using which quantization parameters. Examples of granularities are
        per-tensor and per-channel.

        Note:
            Granularities will soon be replaced by tile specifications.

        Returns:
            quantization granulirity
        """
        raise NotImplementedError

    @property
    def quantization_range(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Quantization range for a quantizer specified through a minimum and maximum threshold.

        Returns:
            Minimum threshold for quantization range as N-d tensor or None of
                no range is specified.
            Maximum threshold for quantization range as N-d tensor or None of
                no range is specified.
        """
        raise NotImplementedError

    @quantization_range.setter
    def quantization_range(self, __range: tuple[torch.Tensor, torch.Tensor]) -> None:
        raise NotImplementedError


@runtime_checkable
class SupportsRangeBasedOperator(RangeSettable, Protocol):
    """Interface for quantizers that can create a quantization operator for a range.

    Interface for Quantizers that can specify a quantization function given a
    min/max thresholded quantization range. This is used by range setting
    methods that require access to multiple parameterized quantization function
    during estimation time.

    This interface extends the `RangeSettable` interface.
    """

    @property
    def symmetric(self) -> bool:
        """Return boolean indicating if quantizer is symmetric.

        Returns:
            Boolean indicating if quantization operator is symmetric or not
        """
        raise NotImplementedError

    def operator_for_range(
        self, __min: torch.Tensor, __max: torch.Tensor, __data_shape: torch.Size
    ) -> Callable[[torch.Tensor], QuantizedTensor]:
        """Return quantization operator using the given range.

        Returns a callable that specifies a quantization operator for the given min/max thresholded
        quantization range and specific data_shape.

        It is assumed that the returned operator is independent of self and not altered during
        its life and users of this method may assume a unique reference to the returned operator.

        Args:
            __min: N-d tensor for minimum threshold of quantization range(s).
            __max: N-d tensor for maximum threshold of quantization range(s).
            __data_shape: Shape of input to quantization operator.

        Note:
            Implementers of this protocol can use different names for the function arguments

        Returns:
            Quantization operator for given quantization range and data shape.
        """
        raise NotImplementedError


_T = TypeVar("_T")
_Module = TypeVar("_Module", bound=torch.nn.Module)


class RangeEstimator(abc.ABC, Generic[_T, _Module]):
    """Abstract base class for range estimator methods.

    Subclasses of this base class implement preparation and cleanup methods for
    range estimation that setup a module for range estimation before passing
    data through the model and clean up any range estimation related settings
    afterwards.

    Since many range estimation methods operate 'locally' on leaf modules for
    which separate prepare and cleanup steps are appropriate, `split_module`
    may take a particular module and yield child modules for which `prepare`
    and `cleanup` is called separately.
    """

    @abc.abstractmethod
    def prepare(self, module: _Module) -> _T:
        """Prepare module for a particular range estimation method.

        Any metadata that is returned from this method is fed back to cleanup
        after range estimation.

        Args:
            module: Module to prepare for range estimation

        Returns:
            None or any Metadata that may be used during cleanup.
        """
        ...

    @abc.abstractmethod
    def cleanup(self, module: _Module, metadata: _T) -> None:
        """Clean up any range estimation specific settings from module.

        After the conclusion of this method, it is assumed that module is in
        the same state as before `prepare` was called, except for some
        quantization specific parameters that where estimated during the range
        estimation step.

        Args:
            module: The module to clean up
            metadata: Any metadata that was returned from the prepare method for `module`
        """
        ...

    @abc.abstractmethod
    def split_module(self, module: torch.nn.Module) -> Iterator[_Module]:
        """Split the module into one or more submodules.

        For each module yielded from this function, prepare and cleanup are
        called separately. If only a single prepare and cleanup should be
        performed from module, only yield module once.

        Yields:
            Submodules for module (or module itself) for which a separate prepare/cleanup
                step is performed.
        """
        ...


_QuantizerType = TypeVar("_QuantizerType", bound=RangeSettable)


class SimpleEstimatorStep(abc.ABC, Generic[_QuantizerType]):
    """Base class for simple range estimator step.

    SimpleEstimatorStep provides a forward pass in which estimate_step is called. During
    the first execution of the forward pass, setup_estimator is called.

    Args:
        disable_quantization: If True, during range estimation, the ranges are estimated
            based on a non-quantized forward pass, i.e., the output of all quantizers will
            not be quantized. If False, the quantizers will produce quantized tensors which
            are propagated through the network.
    """

    def __init__(self, *args: Any, disable_quantization: bool = False, **kwargs: Any) -> None:
        self._initialized = False
        self._disable_quantization = disable_quantization
        super().__init__(*args, **kwargs)

    def setup_estimator(self, data: torch.Tensor) -> None:
        """Perform setup for the estimator.

        The first data batch is passed. this method is only called once during
        the first estimator step.

        Args:
            data: The first batch passed to the estimator.
        """

    @abc.abstractmethod
    def estimate_step(self, quantizer: _QuantizerType, data: torch.Tensor) -> None:
        """Given quantizer and data, update the quantization parameters of `quantizer` based on data.

        Args:
            quantizer: `Quantizer` module for which parameters should be updated
            data: Data that update should be based on
        """
        ...

    @override
    def forward(  # type: ignore[misc]
        self,
        quantizer: _QuantizerType,
        callback: Callable[[torch.Tensor], torch.Tensor],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> torch.Tensor:
        def _data(data: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            return data

        data = _data(*args, **kwargs)
        if not self._initialized:
            self.setup_estimator(data)
            self._initialized = True

        self.estimate_step(quantizer, data)

        if self._disable_quantization:
            return data
        return callback(data)


@contextlib.contextmanager
def estimate_ranges(
    model_or_layers: torch.nn.Module | Sequence[torch.nn.Module],
    estimator: RangeEstimator[_T, _Module] | type[RangeEstimator[_T, _Module]],
    *args: Any,
    **kwargs: Any,
) -> Generator[None, None, None]:
    """Context manager to setup `model_or_layers` for range estimation.

    Within the context, any data that is passed to `model_or_layers` will
    trigger a range estimation step. Each module will be setup by `estimator`
    and cleaned up at the conclusion of the context.

    If `estimator` is a `RangeEstimator` type (in contrast to an instance), it is initialized
    with any extra arguments passed to this function. If it is an instance, it is used as is.

    Example:
        This function may be used as follows:

        ```python
        with estimate_ranges([first_module, second_module], SomeEstimatorClass):
            for batch in data_loader:
                first_module(data)
                second_module(data)

        # Here all quantizers that executed in the previous context will have their
        # ranges initialized and are ready for further use.
        ```
    """
    if isinstance(model_or_layers, torch.nn.Module):
        model_or_layers = [model_or_layers]

    if isinstance(estimator, type):
        estimator = estimator(*args, **kwargs)
    elif args or kwargs:
        raise ValueError(
            "`estimator` is already initialized so no `args` or `kwargs` can be given."
        )

    prepared_modules: list[tuple[_Module, _T]] = []
    for module in model_or_layers:
        for module_part in estimator.split_module(module):
            metadata = estimator.prepare(module_part)
            prepared_modules.append((module_part, metadata))
    try:
        yield
    finally:
        for module, metadata in prepared_modules:
            estimator.cleanup(module, metadata)
