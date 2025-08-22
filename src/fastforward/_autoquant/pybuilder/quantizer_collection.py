# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#
import collections
import contextlib
import dataclasses

from collections.abc import Generator
from typing import Any, Callable, Iterator, TypeAlias

import libcst
import libcst.helpers

from fastforward._autoquant.cst import nodes
from fastforward._autoquant.function_context import FunctionContext
from fastforward.type_common import MethodType

_FuncRef: TypeAlias = Callable[..., Any]


class QuantizerReferenceCollection:
    """Manages quantizer references and their metadata within different function contexts.

    This class is responsible for creating, tracking, and disambiguating quantizer references
    across various contexts (e.g., methods, functions, static methods). It handles
    the lifecycle of quantizer references from creation to name disambiguation, ensuring
    unique naming within appropriate scopes.

    The collection maintains metadata for each quantizer including its context, call site,
    and disambiguated name. It supports context-aware operations through a context stack
    and provides utilities to query quantizers by their associated functions or modules.

    Args:
        quantizer_name_prefix: The prefix to use when generating disambiguated
            quantizer names. Defaults to "quantizer_".
    """

    def __init__(self, quantizer_name_prefix: str = "quantizer_") -> None:
        self._quantizer_metadata: dict[int, QuantizerMetadata] = {}
        self._next_ref_id = 0
        self._quantization_context: FunctionContext | None = None
        self._quantizer_name_prefix = quantizer_name_prefix

    def create_reference(self, name: str) -> nodes.QuantizerReference:
        """Create a new quantizer reference with associated metadata.

        Args:
            name: The base name for the quantizer reference.

        Returns:
            A new QuantizerReference object with a unique reference ID and the
            specified name.
        """
        refid = self._next_ref_id
        self._next_ref_id += 1

        calling_func = getattr(self._quantization_context, "func", None)
        call_site = QuantizerCallSite(refid, func=calling_func)

        self._quantizer_metadata[refid] = QuantizerMetadata(
            refid=refid,
            name=name,
            context=self._quantization_context,
            call_site=call_site,
        )
        return nodes.QuantizerReference(name, refid=refid)

    def create_quantizer_expression(self, name: str) -> libcst.BaseExpression:
        """Create a LibCST expression for accessing a quantizer based on the current context.

        In most contexts, this method should be prefered over `create_reference`.

        Args:
            name: The base name for the quantizer.

        Returns:
            A LibCST BaseExpression representing how to access the quantizer:
            - For instance methods: An attribute access expression like "self.quantizer_name"
            - For other contexts: A direct quantizer reference
        """
        ctx = self._quantization_context
        ref = self.create_reference(name)
        if ctx is None or ctx.method_type is not MethodType.METHOD:
            return ref
        else:
            # if instance var is unknown, self is a good guess
            instance_var = ctx.instance_var or "self"
            return libcst.helpers.parse_template_expression(
                "{owner}.{quantizer}", owner=libcst.Name(instance_var), quantizer=ref
            )

    def _metadata_for_reference(self, ref: nodes.QuantizerReference) -> "QuantizerMetadata":
        if ref.refid not in self._quantizer_metadata:
            msg = f"'{ref} is not a member of '{self}'"
            raise KeyError(msg)
        return self._quantizer_metadata[ref.refid]

    @contextlib.contextmanager
    def push_context(self, context: FunctionContext) -> Generator[None, None, None]:
        """Context manager for temporarily setting the quantization context."""
        original_context = self._quantization_context
        self._quantization_context = context
        try:
            yield
        finally:
            self._quantization_context = original_context

    def disambiguate_all_names(self) -> None:
        """Disambiguate all quantizer names to ensure uniqueness within their respective scopes.

        Note:
            This method modifies the internal state of all QuantizerMetadata objects by
            setting their `disambiguated_name` field. Once called, subsequent calls to
            `disambiguate_reference()` will return the disambiguated names without
            recomputing them.
        """
        quantizer_name_groups: dict[Any, dict[str, list[QuantizerMetadata]]] = (
            collections.defaultdict(lambda: collections.defaultdict(list))
        )
        for metadata in self._quantizer_metadata.values():
            ctx = metadata.context
            if ctx is None:
                # When no context is available, default to the global quantizer group.
                # We use None as the index key, which also handles cases where
                # ctx.module is None, ensuring consistent fallback behavior.
                quantizer_name_groups[None][metadata.name].append(metadata)
            elif ctx.method_type is MethodType.METHOD:
                # When the context represents a model instance, group all quantizers
                # belonging to that instance together to prevent naming conflicts
                quantizer_name_groups[ctx.torch_module][metadata.name].append(metadata)
            else:
                # When the context represents a normal function, classmethod or
                # staticmethod, group all quantizers per function/method.
                quantizer_name_groups[ctx.func][metadata.name].append(metadata)

        def get_name(name: str, idx: int, count: int) -> str:
            full_name = f"{self._quantizer_name_prefix}{name}"
            if count > 1:
                full_name = f"{full_name}_{idx + 1}"
            return full_name

        for group in quantizer_name_groups.values():
            for name, name_metadata in group.items():
                for i, metadata in enumerate(name_metadata):
                    metadata.disambiguated_name = get_name(name, i, len(name_metadata))

    def disambiguate_reference(self, ref: nodes.QuantizerReference | int) -> str:
        """Get the disambiguated name for a quantizer reference.

        Args:
            ref: Either a QuantizerReference object or an integer reference ID
                to look up the quantizer metadata.

        Returns:
            The disambiguated name string for the quantizer reference.

        Raises:
            KeyError: If the provided reference is not found in the collection.
        """
        if isinstance(ref, int):
            metadata = self._quantizer_metadata[ref]
        else:
            metadata = self._metadata_for_reference(ref)

        if metadata.disambiguated_name is None:
            self.disambiguate_all_names()
            assert metadata.disambiguated_name is not None

        return metadata.disambiguated_name

    def local_quantizers_for_func(self, func: _FuncRef) -> Iterator[nodes.QuantizerReference]:
        """Yield quantizer references that are local to the specified function.

        A quantizer is considered local to a function if:
        1. It was created within the context of that function
        2. The function is not an instance method (instance methods have no local
           quantizers as all quantizers are defined on the instance)
        3. The quantizer is actually used within the function (not forwarded to
           another function)

        Args:
            func: The function reference to find local quantizers for.

        Yields:
            QuantizerReference objects that are local to the specified function.
        """
        for metadata in self._quantizer_metadata.values():
            ctx = metadata.context
            if ctx is None or ctx.func is not func:
                # Context of quantizer is not clear, cannot be determinded to
                # be a local quantizer of func.
                continue
            if ctx.method_type is MethodType.METHOD:
                # Instance methods have no local quantizers as all quantizers
                # are defined on the instance.
                continue

            call_site = metadata.call_site
            if call_site is not None and call_site.func is not func:
                # Quantizer is 'forwarded' to other function. It is not used in
                # `func` and thus not a local quantizer.
                continue

            yield nodes.QuantizerReference(metadata.name, refid=metadata.refid)

    def instance_quantizers_for_module(self, module: type) -> Iterator[nodes.QuantizerReference]:
        """Yield quantizer references that belong to instance methods of the specified module.

        This method finds all quantizers that were created within the context of instance
        methods belonging to the given module. These quantizers are typically stored
        as instance attributes on the module and are accessible via attribute access
        (e.g., self.quantizer_name).

        Args:
            module: The torch module type to find instance quantizers for.

        Yields:
            QuantizerReference objects that belong to instance methods of the specified
            module.
        """
        for metadata in list(self._quantizer_metadata.values()):
            ctx = metadata.context
            if (
                ctx is not None
                and ctx.torch_module is module
                and ctx.method_type is MethodType.METHOD
            ):
                yield nodes.QuantizerReference(metadata.name, refid=metadata.refid)


@dataclasses.dataclass
class QuantizerMetadata:
    """Metadata container for quantizer references within the collection."""

    # Unique integer identifier for the quantizer reference.
    refid: int

    # The base name of the quantizer before disambiguation.
    name: str

    # The function context in which the quantizer was created, or None if
    # created outside any specific context.
    context: FunctionContext | None

    # The final unique name assigned to the quantizer after disambiguation, or
    # None if disambiguation hasn't been performed yet.
    disambiguated_name: str | None = None

    # Information about where the quantizer is actually used/called, which may
    # differ from where it was created. None if call site information is not
    # available.
    call_site: "QuantizerCallSite | None" = None


@dataclasses.dataclass
class QuantizerCallSite:
    """Information about where a quantizer is actually used or called."""

    # The unique identifier of the quantizer reference being called.
    refid: int

    # The function where the quantizer is actually used/called, or None if the
    # call site is not within a specific function context.
    func: _FuncRef | None
