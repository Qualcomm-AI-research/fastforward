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

_FuncRef: TypeAlias = Callable[..., None]


class QuantizerReferenceCollection:
    def __init__(self, quantizer_name_prefix: str = "quantizer_") -> None:
        self._quantizer_metadata: dict[int, QuantizerMetadata] = {}
        self._next_ref_id = 0
        self._quantization_context: FunctionContext | None = None
        self._quantizer_name_prefix = quantizer_name_prefix

    def create_reference(
        self, name: str, from_ref: nodes.QuantizerReference | None = None
    ) -> nodes.QuantizerReference:
        refid = self._next_ref_id
        self._next_ref_id += 1

        if from_ref is not None:
            call_site = self._metadata_for_reference(from_ref).call_site
        else:
            calling_func = getattr(self._quantization_context, "func", None)
            call_site = QuantizerCallSite(refid, func=calling_func)

        self._quantizer_metadata[refid] = QuantizerMetadata(
            refid=refid,
            name=name,
            context=self._quantization_context,
            call_site=call_site,
        )
        return nodes.QuantizerReference(name, refid=refid)

    def create_quantizer_expression(
        self, name: str, from_ref: nodes.QuantizerReference | None = None
    ) -> libcst.BaseExpression:
        ctx = self._quantization_context
        ref = self.create_reference(name, from_ref=from_ref)
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
        original_context = self._quantization_context
        self._quantization_context = context
        try:
            yield
        finally:
            self._quantization_context = original_context

    def disambiguate_all_names(self) -> None:
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
    refid: int
    name: str
    context: FunctionContext | None
    disambiguated_name: str | None = None

    # The function in which the quantizer is acutally called
    call_site: "QuantizerCallSite | None" = None


@dataclasses.dataclass
class QuantizerCallSite:
    refid: int
    func: _FuncRef | None
