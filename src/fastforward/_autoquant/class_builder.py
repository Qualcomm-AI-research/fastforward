# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import collections
import logging

from collections.abc import Mapping, Sequence

import torch

from fastforward._autoquant import pybuilder
from fastforward._autoquant.pysource.scope import ImportSymbol
from fastforward._import import fully_qualified_name

logger = logging.getLogger(__name__)


class QuantizedClassNameAllocator:
    """Allocates collision-safe base aliases and quantized class names."""

    def __init__(self, module_types: Sequence[type[torch.nn.Module]]) -> None:
        self._type_name_totals = collections.Counter(mod_type.__name__ for mod_type in module_types)
        self._type_name_next_index: collections.Counter[str] = collections.Counter()
        self._used_quantized_class_names: set[str] = set()
        self._used_base_import_aliases: set[str] = set()

    @staticmethod
    def _alloc_unique_name(preferred: str, used_names: set[str]) -> str:
        if preferred not in used_names:
            used_names.add(preferred)
            return preferred

        index = 1
        while True:
            candidate = f"{preferred}_{index}"
            if candidate not in used_names:
                used_names.add(candidate)
                return candidate
            index += 1

    def for_module_type(self, mod_type: type[torch.nn.Module]) -> tuple[str, str]:
        type_name = mod_type.__name__
        total = self._type_name_totals[type_name]
        index = self._type_name_next_index[type_name]
        self._type_name_next_index[type_name] += 1

        if total > 1:
            preferred_base_alias = f"__ffaq_base_{type_name}_{index}"
            preferred_quantized_name = f"Quantized{type_name}_{index}"
        else:
            preferred_base_alias = type_name
            preferred_quantized_name = f"Quantized{type_name}"

        return (
            self._alloc_unique_name(preferred_base_alias, self._used_base_import_aliases),
            self._alloc_unique_name(preferred_quantized_name, self._used_quantized_class_names),
        )


class ClassBuilderStore:
    """Creates and memoizes one `QuantizedModuleBuilder` per module type."""

    def __init__(self, class_name_allocator: QuantizedClassNameAllocator) -> None:
        self._class_builders: dict[type[torch.nn.Module], pybuilder.QuantizedModuleBuilder] = {}
        self._class_name_allocator = class_name_allocator

    @property
    def builders(self) -> Mapping[type[torch.nn.Module], pybuilder.QuantizedModuleBuilder]:
        """The builders created so far, keyed by the module type they quantize."""
        return self._class_builders

    def ensure_class_builder(
        self, mod_type: type[torch.nn.Module]
    ) -> pybuilder.QuantizedModuleBuilder:
        if mod_type in self._class_builders:
            return self._class_builders[mod_type]

        base_alias, quantized_name = self._class_name_allocator.for_module_type(mod_type)
        class_builder = _cls_builder_for_module(
            mod_type,
            quantized_name=quantized_name,
            base_alias=base_alias,
        )
        self._class_builders[mod_type] = class_builder
        return class_builder


def _cls_builder_for_module(
    module_type: type[torch.nn.Module],
    quantized_name: str,
    base_alias: str,
) -> pybuilder.QuantizedModuleBuilder:
    qualified_class_name = fully_qualified_name(module_type)
    base_module_name, base_class_name = qualified_class_name.rsplit(".", 1)
    import_alias = base_alias if base_alias != base_class_name else None
    return pybuilder.QuantizedModuleBuilder(
        quantized_name,
        bases=(base_alias,),
        required_imports=(
            ImportSymbol(name=base_class_name, module=base_module_name, asname=import_alias),
        ),
        origin=module_type,
    )


def class_builders_in_definition_order(
    class_builders: Mapping[type[torch.nn.Module], pybuilder.QuantizedModuleBuilder],
) -> list[pybuilder.QuantizedModuleBuilder]:
    """Order class builders so a quantized base precedes the classes using it.

    A generated class that inherits from another generated class must appear
    after it, otherwise the emitted module raises `NameError` on import.
    Discovery order does not guarantee this, so the classes are topologically
    sorted over the generated-base relation. Insertion order is otherwise
    preserved.

    Args:
        class_builders: Builders keyed by the module type they quantize.

    Returns:
        The builders of `class_builders`, reordered.
    """
    name_to_type = {builder.name: mod_type for mod_type, builder in class_builders.items()}
    ordered: list[pybuilder.QuantizedModuleBuilder] = []
    visiting: set[type[torch.nn.Module]] = set()
    visited: set[type[torch.nn.Module]] = set()

    def _visit(mod_type: type[torch.nn.Module]) -> None:
        if mod_type in visited:
            return
        if mod_type in visiting:
            # A cycle cannot occur for a valid class hierarchy. Bail out rather
            # than recurse endlessly if one is somehow encountered.
            logger.warning(
                "Cyclic quantized base dependency detected at '%s'", fully_qualified_name(mod_type)
            )
            return
        visiting.add(mod_type)

        base_name = class_builders[mod_type].quantized_base
        base_type = name_to_type.get(base_name) if base_name is not None else None
        if base_type is not None:
            _visit(base_type)

        visiting.discard(mod_type)
        visited.add(mod_type)
        ordered.append(class_builders[mod_type])

    for mod_type in class_builders:
        _visit(mod_type)

    return ordered
