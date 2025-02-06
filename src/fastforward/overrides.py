# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import contextlib

from typing import Any, Callable, Generator, Iterable

import torch

import fastforward as ff

from fastforward import forward_override as override
from fastforward.nn.quantized_module import named_quantizers
from fastforward.quantization.quant_init import QuantizerCollection


@contextlib.contextmanager
def disable_quantization(model: torch.nn.Module) -> Generator[None, None, None]:
    """Disable quantization for all quantizers in `model` within context.

    The global `strict_quantization` flag is also set to `False` during the context.

    Args:
        model: The model for which all quantizers are disabled.
    """
    override = DisableQuantizationOverride()
    quantizers = [quantizer for _, quantizer in named_quantizers(model)]
    override.attach_to(quantizers)
    try:
        with ff.strict_quantization(False):
            yield
    finally:
        override.detach()


class DisableQuantizationOverride:
    """Override to disable quantization.

    Attach `DisableQuantizationOverride` instance as quantizer override to
    disable quantization. Quantization is enabled/disabled using the
    `enable_quantization` and `disable_quantization` methods for all quantizers
    the instance is attached to.

    Quantizers can be 'bulk' attached to using the `attach_to` and `detach`
    methods. Note that `detach` will only detach from quantizers which
    where attached to using the `attach_to` methods, i.e.,  if an instance of
    `DisableQuantizationOverride` is registered as override to a quantizers
    using different means, it must be detached separately.
    """

    def __init__(self) -> None:
        self._quantization_enabled = False
        self._handles: list[override.OverrideHandle] = []

    @property
    def quantization_enabled(self) -> bool:
        """True if quantization is enabled, False otherwise."""
        return self._quantization_enabled

    def enable_quantization(self, enabled: bool = True) -> None:
        """Enable quantization.

        More specifically, this instance will not disable quantization for any
        quantizers it is attached to. Other instance, or other methods may
        still disable quantization.

        Args:
            enabled: True if quantization must be enabled, False if it must be disabled.
        """
        self._quantization_enabled = enabled

    def disable_quantization(self) -> None:
        """Disable quantization.

        See the docstring of `enable_quantization` for more information.
        """
        self.enable_quantization(enabled=False)

    def __call__(
        self,
        _context: Any,
        callback: Callable[..., torch.Tensor],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> torch.Tensor:
        if self._quantization_enabled:
            return callback(*args, **kwargs)
        else:
            return _extract_data_from_args(*args, **kwargs)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(quantization_enabled={self._quantization_enabled})"

    def attach_to(
        self, quantizers: ff.nn.Quantizer | QuantizerCollection | Iterable[ff.nn.Quantizer]
    ) -> None:
        """Attach this override to one or more quantizers.

        Args:
            quantizers: Either a single quantizer, a `QuantizerCollection`
                obtained using `ff.find_quantizers` or an iterable of `Quantizer`s.
                Attach this override to each.
        """
        if isinstance(quantizers, ff.nn.Quantizer):
            self._handles.append(quantizers.register_override(self))
        elif isinstance(quantizers, QuantizerCollection):
            for quantizer in quantizers.modules():
                assert isinstance(quantizer, ff.nn.Quantizer)
                self.attach_to(quantizer)
        else:
            for quantizer in quantizers:  # type: ignore[union-attr]
                self.attach_to(quantizer)

    def detach(self) -> None:
        """Detach this override.

        Detach from all quantizers it was attached to using the
        `attach_to` method.
        """
        for handle in self._handles:
            handle.remove()
        self._handles = []


def _extract_data_from_args(data: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    return data
