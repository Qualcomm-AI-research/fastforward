# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import weakref

from typing import Any

import torch

from torch.utils.hooks import RemovableHandle

import fastforward as ff

strict_quantization = ff.flags.strict_quantization


class ModuleStrictQuantHandle:
    """A handle to manage strict quantization settings for specific modules.

    This class attaches pre-forward and post-forward hooks to the provided modules
    to enable or disable strict quantization during their execution.
    """

    def __init__(self, enable_strict_quantization: bool) -> None:
        """Initialize the ModuleStrictQuantHandle.

        Args:
            enable_strict_quantization: Flag to enable or disable strict quantization.
        """
        self._enable_strict_quantization = enable_strict_quantization
        self._handles: list[RemovableHandle] = []
        self._module_original_strict_mode: weakref.WeakKeyDictionary[torch.nn.Module, bool] = (
            weakref.WeakKeyDictionary()
        )

    def _pre_forward_hook(self, module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        """Pre-forward hook to set the strict quantization mode before module execution.

        Args:
            module: The module being executed.
            args: The arguments passed to the module.
        """
        self._module_original_strict_mode[module] = ff.get_strict_quantization()
        ff.set_strict_quantization(self._enable_strict_quantization)

    def _post_forward_hook(
        self, module: torch.nn.Module, args: tuple[Any, ...], output: Any
    ) -> None:
        """Post-forward hook to restore the original strict quantization mode after module execution.

        Args:
            module: The module being executed.
            args: The arguments passed to the module.
            output: The output of the module.
        """
        if (strict_mode := self._module_original_strict_mode.get(module, None)) is not None:
            ff.set_strict_quantization(strict_mode)

    def attach(self, module: torch.nn.Module) -> None:
        """Attach the pre-forward and post-forward hooks to a module.

        Args:
            module: The module to attach hooks to.
        """
        pre_handle = module.register_forward_pre_hook(self._pre_forward_hook)
        post_handle = module.register_forward_hook(self._post_forward_hook)
        self._handles.append(pre_handle)
        self._handles.append(post_handle)

    def remove(self) -> None:
        """Remove all attached hooks.
        """
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type, exc_value, traceback):  # type: ignore[no-untyped-def]
        """Exit the context manager and remove all hooks.
        """
        self.remove()


def strict_quantization_for_module(
    strict: bool = True, *modules: torch.nn.Module
) -> ModuleStrictQuantHandle:
    """Enable or disable strict quantization on a per module basis.

    This function attaches pre-forward and post-forward hooks to the provided modules
    to set the global strict quantization setting before executing the model and reset it
    to its original state after the module concludes. It can be used directly or as a context manager.

    Args:
        strict: The strict quantization state to use for modules.
        modules: The modules to apply the strict quantization state to.

    Returns:
        ModuleStrictQuantHandle: A handle to manage the strict quantization settings.

    Examples:
        >>> network = Network()
        >>> with strict_quantization_for_module(False, network.layer3, network.layer5):
        >>>     network(torch.randn(3, 3))

        >>> network = Network()
        >>> handle = strict_quantization_for_module(False, network.layer1, network.layer2, network.layer3)
        >>> network(torch.randn(3, 3))
        >>> handle.remove()
    """
    handle = ModuleStrictQuantHandle(enable_strict_quantization=strict)
    for module in modules:
        handle.attach(module)
    return handle
