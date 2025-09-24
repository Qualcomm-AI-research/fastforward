# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import contextlib

from collections.abc import Iterator, Sequence
from typing import Any, Callable

import torch

import fastforward as ff


class _FreezeParametersOverride:
    """Override that quantizes parameters in-place and replaces quantizers with stubs."""

    def __init__(self, module: torch.nn.Module, remove_quantizer: bool) -> None:
        self._module = module
        self._remove_quantizer = remove_quantizer

    def __call__(
        self,
        quantizer: "ff.nn.Quantizer",
        callback: Callable[..., Any],
        args: tuple[Any],
        kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """Quantize input data in-place and replace the quantizer with a stub.

        Args:
            quantizer: The quantizer being overridden
            callback: Quantization callback function
            args: Positional arguments containing input data
            kwargs: Keyword arguments

        Returns:
            Dequantized tensor result
        """
        # Extract and quantize input data
        input_data = _extract_input(*args, **kwargs)
        quantized = callback(input_data).dequantize()

        if quantized is input_data:
            # When the quantizer callback returns the input unchanged (e.g., when
            # the quantizer is disabled), freezing has no effect. In this case,
            # we skip both the quantizer removal and the in-place parameter update
            # since they would be no-ops.
            return input_data

        # Update parameter in-place if input is a Parameter
        if isinstance(input_data, torch.nn.Parameter):
            with torch.no_grad():
                input_data.copy_(quantized)

        # Replace quantizer with stub to freeze quantization
        if self._remove_quantizer:
            for name, other_quantizer in ff.nn.quantized_module.named_quantizers(
                self._module, recurse=False
            ):
                if other_quantizer is quantizer:
                    stub = ff.nn.QuantizerStub(_metadata=quantizer.quant_metadata)
                    setattr(self._module, name, stub)
                    break

        return input_data


def _extract_input(data: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    del args, kwargs
    return data


@contextlib.contextmanager
def freeze_parameters(
    modules: torch.nn.Module | Sequence[torch.nn.Module], remove_quantizers: bool = True
) -> Iterator[None]:
    """Freeze model parameters during forward pass.

    Registers overrides on all quantizers to quantize parameters, replace
    original values with quantized versions, and convert quantizers to stubs.
    The user of this function is expected to perform a single forward pass
    within this context.

    Args:
        modules: PyTorch module or sequence of modules whose parameters should
            be frozen with quantized values.
        remove_quantizers: If true, all quantizers for which parameters are frozen are replaced
            by `QuantizerStub`s.

    Returns:
        A context manager. Within this context manager, a forward pass will
        freeze quantized parameters. At exit of the context manager, all freeze
        parameter overrides are removed.

    Warning:
        This context manager freezes parameters by quantizing them in-place during
        a forward pass. Key behaviors:

        - Only `torch.nn.Parameter` inputs are frozen in-place; transformed parameters
          (e.g., scaled) are no longer a `Parameter` and are not frozen.
        - Frozen parameters retain their original dtype but contain quantized values.
        - All active hooks and overrides are applied during freezing.
        - Quantizers are replaced with `QuantizerStub`s (unless `remove_quantizers=False`).
        - No-op quantizers (where output equals input by identity) are not frozen or removed;
          for example, disabled quantizers will not be removed and their parameter inputs
          are not changed

    Note:
        Strict quantization is temporarily disabled as overrides return regular tensors.
    """
    hooks: list[ff.overrides.override.OverrideHandle] = []
    modules = [modules] if isinstance(modules, torch.nn.Module) else modules
    for module in modules:
        for submodule in module.modules():
            for _, quantizer in ff.nn.quantized_module.named_quantizers(submodule, recurse=False):
                override = _FreezeParametersOverride(submodule, remove_quantizer=remove_quantizers)
                quantizer.register_override(override)

    try:
        # Disable strict quantization because override returns 'normal' tensors.
        with ff.strict_quantization(False):
            yield
    finally:
        for hook in hooks:
            hook.remove()
