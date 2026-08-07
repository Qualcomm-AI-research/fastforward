# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


import logging

from typing import Mapping

import libcst

from fastforward._autoquant.cst import nodes, super_calls
from fastforward._autoquant.cst.passes import QuantizedCounterpartReplacer
from fastforward._autoquant.function_context import FunctionContext
from fastforward._autoquant.pybuilder import QuantizerReferenceCollection
from fastforward._autoquant.pysource.scope import ImportSymbol, find_required_imports
from fastforward._quantops import OperatorTable
from fastforward.type_common import MethodType

from .cst.quantizer_analysis.transformer import QuantizerFunctionTransformer
from .pass_manager import PassManager
from .pybuilder import QuantizedFunctionBuilder
from .pysource import PySource

logger = logging.getLogger(__name__)


class _AnchorSuperCalls(libcst.CSTTransformer):
    """Rewrite `super()` calls that target a quantized class to the generated quantized class.

    Both the bare `super()` form and the explicit `super(OwnerClass, self)` form
    are rewritten: both need to resolve to the *generated* class's superclass.

    An explicit call anchored to a different class (e.g. skipping a level of the
    MRO) is rewritten only if `super_anchors` maps its method to a quantized
    class counterpart; otherwise it is left untouched.

    A method present in `super_anchors` has its rewritten callsites marked as a
    `QuantizedSuperCall`, so downstream quantizer analysis knows the result is
    already quantized.
    """

    def __init__(
        self,
        owner_class_name: str,
        anchor: libcst.BaseExpression,
        self_name: str,
        super_anchors: Mapping[str, str | None] = {},
    ) -> None:
        self._owner_class_name = owner_class_name
        self._anchor = anchor
        self._self_name = self_name
        self._super_anchors = super_anchors

    def leave_Call(
        self,
        original_node: libcst.Call,
        updated_node: libcst.Call,
    ) -> libcst.BaseExpression:
        del original_node

        func = updated_node.func
        if not isinstance(func, libcst.Attribute):
            return updated_node

        super_call = func.value
        if not isinstance(super_call, libcst.Call):
            return updated_node

        if not isinstance(super_call.func, libcst.Name) or super_call.func.value != "super":
            return updated_node

        method_name = func.attr.value
        if super_calls.super_call_targets_owner(
            super_call, self._owner_class_name, self._self_name
        ):
            new_anchor: libcst.BaseExpression = self._anchor.deep_clone()
        elif (override := self._super_anchors.get(method_name)) is not None:
            # A deliberate MRO skip whose anchor was itself quantized -- reanchor
            # to that generated class so the skip is preserved in quantized form.
            new_anchor = libcst.Name(override)
        else:
            # Anchored to something other than the owning class, and no
            # generated counterpart was resolved for it: the call target is
            # left as written.
            return updated_node

        super_call = super_call.with_changes(
            args=(
                libcst.Arg(new_anchor),
                libcst.Arg(libcst.Name(self._self_name)),
            )
        )
        updated_node = updated_node.with_changes(func=func.with_changes(value=super_call))

        if method_name in self._super_anchors:
            return nodes.QuantizedSuperCall(**nodes.node_asdict(updated_node))

        return updated_node


def _owner_class_name(src: PySource) -> str | None:
    parts = src.qualified_name.split(".")
    if len(parts) < 2:
        return None
    return parts[-2]


def _rewrite_super_calls(
    cst: libcst.FunctionDef,
    src: PySource,
    func_ctx: FunctionContext,
    super_anchors: Mapping[str, str | None] = {},
) -> libcst.FunctionDef:
    """Rewrite `super()` calls that target a quantized class to the generated quantized class.

    Both the bare `super()` form and the explicit `super(OwnerClass, self)` form
    are rewritten: both need to resolve to the *generated* class's superclass.

    An explicit call anchored to a different class (e.g. skipping a level of the
    MRO) is rewritten only if `super_anchors` maps its method to a quantized
    class counterpart; otherwise it is left untouched.

    A method present in `super_anchors` has its rewritten callsites marked as a
    `QuantizedSuperCall`, so downstream quantizer analysis knows the result is
    already quantized.

    Args:
        cst: The method to rewrite.
        src: Source reference for `cst`, used to find the original class name.
        func_ctx: Context in which function appears in code.
        super_anchors: Maps each super-delegated method that resolves to a
            quantized implementation to the generated class its callsites
            should be anchored to, or `None` to anchor them to the generated
            class the method is copied into. See `convert_function`.
    """
    if func_ctx.method_type is not MethodType.METHOD or func_ctx.instance_var is None:
        return cst

    owner = _owner_class_name(src)
    if owner is None:
        return cst

    # `AbstractClassReference` defers the name because the generated class name
    # is not yet known here: it is resolved when the class is built.
    anchor: libcst.BaseExpression = nodes.AbstractClassReference(owner)
    rewritten = cst.visit(_AnchorSuperCalls(owner, anchor, func_ctx.instance_var, super_anchors))
    assert isinstance(rewritten, libcst.FunctionDef)
    return rewritten


def convert_function(
    src: PySource,
    optable: OperatorTable,
    func_ctx: "FunctionContext",
    quantizer_refs: QuantizerReferenceCollection,
    super_anchors: Mapping[str, str | None] = {},
) -> QuantizedFunctionBuilder:
    """Convert a single function or method to its quantized counterpart.

    The source of `src` is rewritten in three steps: `super()` callsites are
    re-anchored for the generated quantized class (see `_rewrite_super_calls`),
    operator calls are replaced by their quantized equivalents using `optable`
    as 'ground-truth', and the quantizers those replacements need are created.
    `func_ctx` determines how the function is treated -- in particular whether
    it is a method, and which parameter is its instance variable.

    Quantizers discovered during conversion are registered in `quantizer_refs`
    under the context the caller has pushed, so this is expected to be called
    within the matching `quantizer_refs.push_context(func_ctx)`.

    Args:
        src: `PySource` object that represents the function to be quantized.
        optable: The `OperatorTable` that is used as 'ground-truth' for
            operator replacement.
        func_ctx: Context in which the function appears in code, including its
            method type. Use `MethodType.NO_METHOD` for non-method functions.
        quantizer_refs: Quantizer reference collection used for autoquant.
        super_anchors: The methods that, when called through `super()`, resolve
            to a quantized implementation. Membership means the result is
            treated as already quantized. The value is the generated class name
            that a deliberate-MRO-skip `super(Ancestor, self)` callsite should
            be re-anchored to, or `None` when the call targets the owning class
            and is therefore anchored to the generated class the method is
            copied into.

    Returns:
        A `QuantizedFunctionBuilder` holding the converted `FunctionDef` and the
        imports it requires. The builder is not yet attached to a class or
        module builder (the caller does that) but the quantizers it needs
        have already been added to `quantizer_refs`.
    """
    logger.info(
        "convert_function: converting %s (%s)",
        src.qualified_name,
        func_ctx.method_type.name if func_ctx.method_type is not None else None,
    )
    src_cst = src.cst(NodeType=libcst.FunctionDef)
    # Anchor `super()` before the conversion passes so that quantizer analysis
    # sees which callsites already yield a quantized value.
    src_cst = _rewrite_super_calls(
        src_cst,
        src=src,
        func_ctx=func_ctx,
        super_anchors=super_anchors,
    )
    required_imports_extra: set[ImportSymbol] = set()
    dst_cst = autoquantize_funcdef(
        src_cst=src_cst,
        optable=optable,
        func_ctx=func_ctx,
        quantizer_refs=quantizer_refs,
        required_imports_extra=required_imports_extra,
    )

    assert isinstance(dst_cst, libcst.FunctionDef)
    required_imports = _infer_imports(src, dst_cst)
    required_imports |= required_imports_extra

    return QuantizedFunctionBuilder(dst_cst, required_imports, origin=func_ctx)


def autoquantize_funcdef(
    src_cst: libcst.FunctionDef,
    optable: OperatorTable,
    func_ctx: FunctionContext,
    quantizer_refs: QuantizerReferenceCollection,
    required_imports_extra: set[ImportSymbol] | None = None,
) -> libcst.FunctionDef:
    """Autoquantize a single `FuncDef` with given `optable`."""
    counterpart_replacer = QuantizedCounterpartReplacer(
        optable=optable,
        func_ctx=func_ctx,
        quantizer_refs=quantizer_refs,
    )
    pm = PassManager(
        passes=[
            counterpart_replacer,
            QuantizerFunctionTransformer(quantizer_refs=quantizer_refs),
        ]
    )
    converted = pm(src_cst)

    if required_imports_extra is not None:
        required_imports_extra.update(counterpart_replacer.required_imports_extra)

    return converted


def _infer_imports(src: PySource, cst: libcst.FunctionDef) -> set[ImportSymbol]:
    if (scope := src.scope()) is None:
        # Cannot infer import without scope info. Don't error to avoid autoquant failure
        return set()
    return find_required_imports(cst, scope, src.module().qualified_name)
