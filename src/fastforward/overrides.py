# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import contextlib

from typing import Any, Callable, Generator, Iterable

import torch

from typing_extensions import override as typing_override

import fastforward as ff

from fastforward import forward_override
from fastforward.nn.quantized_module import named_quantizers
from fastforward.quantization.quant_init import QuantizerCollection

# make mypy treat override as exported
override = forward_override


@contextlib.contextmanager
def disable_quantization(model: torch.nn.Module) -> Generator[None, None, None]:
    """Disable quantization for all quantizers in `model` within context.

    The global `strict_quantization` flag is also set to `False` during the context.

    Args:
        model: The model for which all quantizers are disabled.
    """
    handles: list[override.OverrideHandle] = []
    for _, quantizer in named_quantizers(model):
        handles.append(quantizer.register_override(DisableQuantizationOverride()))

    try:
        with ff.strict_quantization(False):
            yield
    finally:
        for handle in handles:
            handle.remove()


@contextlib.contextmanager
def enable_quantization(model: torch.nn.Module) -> Generator[None, None, None]:
    """Enable quantization for all quantizers in `model` within context.

    Note that this context manager does not change the `strict_quantization` flag.
    To also (temporarily) change the `strict_quantization` flag use
    `fastforward.quantization.strict_quantization.strict_quantization_for_module`

    Args:
        model: The model for which all quantizers are enabled.
    """
    with contextlib.ExitStack() as exit_stack:
        for _, quantizer in named_quantizers(model):
            for quantizer_override in quantizer.overrides:
                if isinstance(quantizer_override, DisableQuantizationOverride):
                    exit_stack.enter_context(quantizer_override.enable_quantization())
        yield


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

    def enable_quantization(self, enabled: bool = True) -> "_QuantizationContext":
        """Enable quantization.

        More specifically, this instance will not disable quantization for any
        quantizers it is attached to. Other instance, or other methods may
        still disable quantization.

        Args:
            enabled: True if quantization must be enabled, False if it must be disabled.
        """
        context = _QuantizationContext(self, self._quantization_enabled)
        self._quantization_enabled = enabled
        return context

    def disable_quantization(self) -> "_QuantizationContext":
        """Disable quantization.

        See the docstring of `enable_quantization` for more information.
        """
        return self.enable_quantization(enabled=False)

    def __call__(
        self,
        _context: Any,
        callback: Callable[..., torch.Tensor],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """Override function for quantizer disabling."""
        if self._quantization_enabled:
            return callback(*args, **kwargs)
        else:
            return _extract_data_from_args(*args, **kwargs)

    @typing_override
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


class _QuantizationContext(contextlib.AbstractContextManager[None]):
    def __init__(
        self, override: DisableQuantizationOverride, quantization_enabled_reset_status: bool
    ) -> None:
        self._override = override
        self._reset_quantization_status = quantization_enabled_reset_status

    def __enter__(self) -> None: ...

    def __exit__(self, type, value, traceback) -> None:  # type: ignore[no-untyped-def]
        self._override._quantization_enabled = self._reset_quantization_status


def _extract_data_from_args(data: torch.Tensor, *_args: Any, **_kwargs: Any) -> torch.Tensor:
    return data
