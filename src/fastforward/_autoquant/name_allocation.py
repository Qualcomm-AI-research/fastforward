# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


import collections
import logging

from collections.abc import Sequence

import libcst
import torch

from .cst import nodes

logger = logging.getLogger(__name__)


class QuantizedClassNameAllocator:
    """Allocates collision-safe base aliases and quantized class names."""

    def __init__(self, module_types: Sequence[type[torch.nn.Module]]) -> None:
        self._type_name_totals = collections.Counter(mod_type.__name__ for mod_type in module_types)
        self._type_name_next_index: collections.Counter[str] = collections.Counter()
        self._used_quantized_class_names: set[str] = set()
        self._used_base_import_aliases: set[str] = set()

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
            alloc_unique_name(preferred_base_alias, self._used_base_import_aliases),
            alloc_unique_name(preferred_quantized_name, self._used_quantized_class_names),
        )


class RenameHelperRefsTransformer(libcst.CSTTransformer):
    """Replace stale Name references to renamed helpers.

    After autoquant renames helper functions (e.g. ``group_norm`` →
    ``quantized_group_norm``), non-call references to the old name become
    undefined.  This pass rewrites all such ``Name`` nodes.  Call-target
    renames are already handled by ``_ResolveQuantizedCallsTransformer``.

    Only standalone ``Name`` nodes are renamed — attribute accesses like
    ``torch.group_norm`` are left untouched because the ``attr`` part of an
    ``Attribute`` node is not an independent reference. ``Name`` subclasses that
    carry deferred-resolution metadata are also left untouched.
    """

    def __init__(self, name_map: dict[str, libcst.BaseExpression]) -> None:
        self._name_map = name_map
        self._inside_attr: set[int] = set()

    def visit_Attribute(self, node: libcst.Attribute) -> bool:
        self._inside_attr.add(id(node.attr))
        return True

    def leave_Attribute(
        self,
        original_node: libcst.Attribute,
        updated_node: libcst.Attribute,
    ) -> libcst.BaseExpression:
        self._inside_attr.discard(id(original_node.attr))
        return updated_node

    def leave_Name(
        self,
        original_node: libcst.Name,
        updated_node: libcst.Name,
    ) -> libcst.BaseExpression:
        if isinstance(updated_node, (nodes.QuantizerReference, nodes.AbstractClassReference)):
            # Both subclass `libcst.Name`. Replacing them with a plain `Name`
            # would discard the metadata used to resolve them at build time.
            return updated_node
        if id(original_node) in self._inside_attr:
            return updated_node
        if replacement := self._name_map.get(updated_node.value):
            return replacement.deep_clone()
        return updated_node


def alloc_unique_name(preferred: str, used_names: set[str]) -> str:
    """Return `preferred`, or the first free `{preferred}_{index}` variant.

    The returned name is added to `used_names`.
    """
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
