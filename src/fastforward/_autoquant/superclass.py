# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import logging

import torch

import fastforward as ff

from fastforward._import import fully_qualified_name
from fastforward.nn.quantized_module import QuantizedModule

logger = logging.getLogger(__name__)


class SuperclassResolver:
    def __init__(self, pre_quantized_modules: set[type[torch.nn.Module]]) -> None:
        self._pre_quantized_modules = pre_quantized_modules
        self._module_map = ff.nn.quantized_module_map()
        self._super_base_cache: dict[
            type[torch.nn.Module], tuple[type[torch.nn.Module], type[QuantizedModule] | None] | None
        ] = {}

    def resolve_superclass(
        self, mod_type: type[torch.nn.Module]
    ) -> tuple[type[torch.nn.Module], type[QuantizedModule] | None] | None:
        if mod_type in self._super_base_cache:
            return self._super_base_cache[mod_type]

        result = None
        base_type = _quantizable_superclass(mod_type)
        if base_type is not None:
            quantized_type = None
            if base_type in self._pre_quantized_modules:
                quantized_type = self._module_map.get(base_type)
                if quantized_type is None:
                    logger.warning(
                        "Cannot resolve a quantized counterpart for superclass '%s' of '%s'. "
                        + "super() calls in the generated code may not be quantized.",
                        fully_qualified_name(base_type),
                        fully_qualified_name(mod_type),
                    )
            if quantized_type is not None or base_type not in self._pre_quantized_modules:
                result = (base_type, quantized_type)

        self._super_base_cache[mod_type] = result
        return result


def ancestor_chain(
    module_type: type[torch.nn.Module], anchor_name: str
) -> list[type[torch.nn.Module]] | None:
    """Find the `torch.nn.Module` ancestors of `module_type` up to and including `anchor_name`.

    An explicit `super(Ancestor, self)` call deliberately skips every class
    between `module_type` and `Ancestor` in the MRO. Quantizing that call
    correctly requires a quantized counterpart for every skipped class, not
    just `Ancestor`: the generated classes mirror the original MRO so that
    re-anchoring the call to the generated `Ancestor` counterpart still skips
    the same classes. This returns that segment of the MRO, in order, so the
    caller can quantize (or stub) each one and chain them together.

    Args:
        module_type: The class whose `super(Ancestor, self)` call is being
            resolved.
        anchor_name: `__name__` of `Ancestor`, as it appears in the call.

    Returns:
        The `torch.nn.Module` ancestors of `module_type` (skipping non-Module
        mixins), starting from its immediate superclass, up to and including
        the first one named `anchor_name`. `None` if no ancestor in
        `module_type`'s MRO has that name.
    """
    chain: list[type[torch.nn.Module]] = []
    for base in module_type.__mro__[1:]:
        if not issubclass(base, torch.nn.Module):
            continue
        chain.append(base)
        if base.__name__ == anchor_name:
            return chain
    return None


def method_provider_chain(
    module_type: type[torch.nn.Module],
    anchor_type: type[torch.nn.Module],
    method_name: str,
) -> list[type[torch.nn.Module]]:
    """Find the ancestors after `anchor_type` up to the one providing `method_name`.

    `super(Anchor, self).method()` resolves *past* `Anchor` itself, so the
    implementation it reaches lives on a later class in the MRO. The generated
    counterpart of that provider must therefore also be part of the chain that
    `ancestor_chain` mirrors: without it, the counterpart of `Anchor` has no
    quantized base, and the re-anchored call falls through `QuantizedModule`
    into the *original* implementations, re-running the unquantized method
    bodies that come earlier in the generated class's MRO.

    Args:
        module_type: The class whose `super(Anchor, self)` call is being
            resolved.
        anchor_type: The class named in the call, as returned by
            `ancestor_chain`.
        method_name: Name of the method invoked on the `super()` object.

    Returns:
        The `torch.nn.Module` ancestors of `module_type` strictly after
        `anchor_type` (skipping non-Module mixins), up to and including the
        first one that defines `method_name` itself. Empty if no quantizable
        ancestor provides it, i.e. the search reaches `torch.nn.Module` or an
        existing `QuantizedModule` first.
    """
    mro = module_type.__mro__
    chain: list[type[torch.nn.Module]] = []
    for base in mro[mro.index(anchor_type) + 1 :]:
        if base is torch.nn.Module or issubclass(base, QuantizedModule):
            return []
        if not issubclass(base, torch.nn.Module):
            continue
        chain.append(base)
        if method_name in vars(base):
            return chain
    return []


def _quantizable_superclass(module_type: type[torch.nn.Module]) -> type[torch.nn.Module] | None:
    """Find the superclass whose implementations `module_type`'s `super()` calls reach.

    Returns the first `torch.nn.Module` in the MRO after `module_type` itself,
    which is what a `super()` call inside `module_type` resolves against.
    `torch.nn.Module` itself is not a meaningful quantization target and is
    excluded, as are classes that are already `QuantizedModule`s.
    """
    for base in module_type.__mro__[1:]:
        if base is torch.nn.Module:
            return None
        if not issubclass(base, torch.nn.Module):
            continue
        if issubclass(base, QuantizedModule):
            return None
        return base
    return None
