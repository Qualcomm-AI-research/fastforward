# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import Any, Callable, Generator, Iterator, Protocol, Sequence

import torch

from fastforward.nn import QuantizedModule
from fastforward.nn.quantizer import Quantizer
from fastforward.quantization.local_error import LocalErrorMethod, RunnerContext, execution_context


def _data_from_args(data: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    """Extract the data tensor from the arguments.

    Args:
        data: The data tensor.
        args: Additional positional arguments.
        kwargs: Additional keyword arguments.

    Returns:
        torch.Tensor: The data tensor.
    """
    return data


class _Handle(Protocol):
    """Protocol for a handle that can be removed."""

    def remove(self) -> Any:
        """Remove the handle."""


class OutputMSEOverride:
    """Override class for output Mean Squared Error (MSE) calculation.

    Attributes:
        passthrough: Flag to enable passthrough mode.
    """

    def __init__(self) -> None:
        self.passthrough = False

    def __call__(
        self,
        quantizer: Quantizer,
        callback: Callable[[torch.Tensor], torch.Tensor],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """Call the override with the given quantizer, callback, and arguments.

        Args:
            quantizer: The quantizer instance.
            callback: The callback function.
            args: Positional arguments for the callback.
            kwargs: Keyword arguments for the callback.

        Returns:
            torch.Tensor: The result of the callback or the data tensor if passthrough is enabled.
        """
        del quantizer

        data = _data_from_args(*args, **kwargs)
        if self.passthrough:
            return data

        return callback(*args, **kwargs)


class OutputMSE(LocalErrorMethod):
    """Local error method for output Mean Squared Error (MSE) minimization."""

    def __init__(
        self, modules: Sequence[QuantizedModule], optimizer_factory: torch.optim.Optimizer
    ) -> None:
        self._modules = modules
        self._override = OutputMSEOverride()
        self._handles: list[_Handle] = []
        self._optimizer_factory = optimizer_factory

    def _all_quantizers(self) -> Iterator[Quantizer]:
        """Iterate over all quantizers in the modules.

        Yields:
            Iterator[Quantizer]: An iterator over quantizers.
        """
        for module in self._modules:
            for quantizer in module.quantizers(skip_stubs=True):
                yield quantizer

    def _all_outputs(self) -> Iterator[Quantizer]:
        """Iterate over all output quantizers in the modules.

        Yields:
            Iterator[Quantizer]: An iterator over output quantizers.
        """
        for module in self._modules:
            for quantizer in module.quantizers(skip_stubs=False):
                if (metadata := quantizer.quant_metadata) and metadata.output_quantizer:
                    yield quantizer

    def prepare(self, ctx: RunnerContext) -> None:
        """Prepare the context by registering hooks for quantizers.

        Args:
            ctx: The runner context.
        """

        def hook(_: torch.nn.Module, args: tuple[Any, ...]) -> torch.Tensor:
            return ctx.communicate(args[0])

        for quantizer in self._all_quantizers():
            metadata = quantizer.quant_metadata
            if metadata and metadata.parameter_quantizer:
                self._handles.append(quantizer.register_override(self._override))

        for output_quantizer in self._all_outputs():
            self._handles.append(output_quantizer.register_forward_pre_hook(hook))

    def cleanup(self) -> None:
        """Clean up by removing all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @execution_context
    def alternative_context(self) -> Generator[None, None, None]:
        """Provide an alternative context with passthrough mode enabled.

        Yields:
            Generator[None, None, None]: A generator for the alternative context.
        """
        self._override.passthrough = True
        try:
            yield
        finally:
            self._override.passthrough = False

    def conclude_partition(self) -> None:
        """Conclude the current partition by zeroing the gradients."""
        self._optimizer_factory.zero_grad()

    def update(self, quantized: torch.Tensor, unquantized: torch.Tensor) -> None:
        """Update the model parameters based on the MSE between quantized and unquantized tensors.

        Args:
            quantized: The quantized tensor.
            unquantized: The unquantized tensor.
        """
        self._optimizer_factory.zero_grad()
        error = (quantized - unquantized).pow(2).sum()
        error.backward()
        self._optimizer_factory.step()

    def propagate(self, replay_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Propagate the replay value to the default and alternative context.

        Args:
            replay_value: The replay value tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The detached and dequantized replay value.
        """
        return replay_value.detach(), replay_value.dequantize().detach()
