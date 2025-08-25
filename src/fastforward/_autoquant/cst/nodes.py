# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""This module implements various CST Nodes used for code analysis.

These CSTNodes are used to ease further code analysis without destroying any
information in the CST.

Each of the nodes implemented in this module will cooperate with a 'vanilla'
CST and can be used during code generation.
"""

import dataclasses

from typing import Any, Callable, Sequence

import libcst

from libcst._nodes.internal import visit_optional, visit_required, visit_sequence
from typing_extensions import override

from fastforward._quantops.operator import Operator


@dataclasses.dataclass(slots=True, frozen=True)
class GeneralAssignment(libcst.BaseSmallStatement):
    """A universal node that wraps different types of assignments.

    Any transformation is implemented separately and
    this node is only used to 'communicate' between different passes.

    We copy all fields of the original assignment statement that may be updated by subsequent
    processing. Other fields which are irrelevant for quantization are hidden away in the
    `original` field and should not be modified, but are stored for later so-called deflation.
    """

    original: libcst.AnnAssign | libcst.Assign | libcst.AugAssign
    targets: Sequence[libcst.BaseAssignTargetExpression]
    annotation: libcst.Annotation | None
    value: libcst.BaseExpression | None
    semicolon: libcst.Semicolon | libcst.MaybeSentinel = libcst.MaybeSentinel.DEFAULT

    def _codegen_impl(self, *args: Any, **kwargs: Any) -> None:
        self._validate()
        deflated_node = self._deflate()
        deflated_node._codegen_impl(*args, **kwargs)

    def _validate(self) -> None:
        assert (self.value is None) == (self.original.value is None), (
            "Should have a value if and only if original had a value."
        )
        match self.original:
            case libcst.AnnAssign():
                assert bool(self.value) == bool(self.original.value), (
                    "Should have a value if and only if original had a value."
                )
                assert self.annotation is not None, "Expected an annotation."
                assert len(self.targets) == 1, "Should not have additional augmentation targets."
            case libcst.Assign():
                assert bool(self.value) == bool(self.original.value), (
                    "Should have a value if and only if original had a value."
                )
                assert len(self.targets) == len(self.original.targets), (
                    "Should not have new assignment targets."
                )
            case libcst.AugAssign():
                assert bool(self.value) == bool(self.original.value), (
                    "Should have a value if and only if original had a value."
                )
                assert len(self.targets) == 1, "Should not have additional augmentation targets."
                assert self.value is not None, "Expected a value to assign to."
            case _:
                raise TypeError(f"Unexpected node type: {type(self.original)}")

    def _deflate(self) -> libcst.CSTNode:
        match self.original:
            case libcst.AnnAssign():
                return self.original.with_changes(
                    target=self.targets[0],
                    annotation=self.annotation,
                    value=self.value,
                )
            case libcst.Assign():
                assign_targets = tuple(
                    assign_target.with_changes(target=base_target)
                    for assign_target, base_target in zip(
                        self.original.targets, self.targets, strict=True
                    )
                )
                return self.original.with_changes(targets=assign_targets, value=self.value)
            case libcst.AugAssign():
                return self.original.with_changes(target=self.targets[0], value=self.value)
            case _:
                raise TypeError(f"Unexpected node type: {type(self.original)}")

    def _visit_and_replace_children(self, visitor: libcst.CSTVisitorT) -> libcst.CSTNode:
        return type(self)(
            self.original,
            visit_sequence(self, "targets", self.targets, visitor) if self.targets else (),
            visit_optional(self, "annotation", self.annotation, visitor),
            visit_optional(self, "value", self.value, visitor),
        )


@dataclasses.dataclass(slots=True, frozen=True)
class ReplacementCandidate(libcst.BaseExpression):
    """A marker node that wraps an original BaseExpression.

    Any transformation is implemented separately and
    this node is only used to 'communicate' between different passes.
    """

    original: libcst.BaseExpression

    def _codegen_impl(self, *args: Any, **kwargs: Any) -> None:
        self.original._codegen(*args, **kwargs)

    def _visit_and_replace_children(self, visitor: libcst.CSTVisitorT) -> libcst.CSTNode:
        return ReplacementCandidate(visit_required(self, "original", self.original, visitor))

    def _safe_to_use_with_word_operator(self, *args: Any, **kwargs: Any) -> bool:
        return self.original._safe_to_use_with_word_operator(*args, **kwargs)

    def _check_left_right_word_concatenation_safety(self, *args: Any, **kwargs: Any) -> bool:
        return self.original._check_left_right_word_concatenation_safety(*args, **kwargs)


@dataclasses.dataclass(slots=True, frozen=True)
class QuantizedCall(libcst.Call):
    """A metadata node that carries extra information and wraps a `libcst.Call`.

    This wrapper node contains extra information on the quantized
    operation that is 'called'. This can be helpful in further analysis.
    """

    original_name: str = dataclasses.field(kw_only=True)
    operator: Operator | None = dataclasses.field(kw_only=True, default=None)
    func_ref: Callable[..., Any] | None = dataclasses.field(kw_only=True, default=None)

    def _visit_and_replace_children(self, visitor: libcst.CSTVisitorT) -> "QuantizedCall":
        # This method must be implemented to prevent a 'downcast' to a
        # libcst.Call node during arbitrary transformer application
        visited_call = libcst.Call._visit_and_replace_children(self, visitor)
        return QuantizedCall(
            **node_asdict(visited_call), original_name=self.original_name, operator=self.operator
        )


@dataclasses.dataclass(slots=True, frozen=True)
class UnresolvedQuantizedCall(libcst.Call):
    """A metadata node that carries extra information and wraps a `libcst.Call`.

    This wrapper node represents a quantized function call that has been identified
    but not yet fully resolved or transformed. Unlike `QuantizedCall`, this node
    maintains the original function reference and name for later processing during
    the quantization pipeline. It serves as an intermediate representation that
    preserves metadata needed for subsequent analysis and transformation steps.
    """

    original_name: str = dataclasses.field(kw_only=True)
    func_ref: Callable[..., Any] = dataclasses.field(kw_only=True)

    def _visit_and_replace_children(self, visitor: libcst.CSTVisitorT) -> "UnresolvedQuantizedCall":
        # This method must be implemented to prevent a 'downcast' to a
        # libcst.Call node during arbitrary transformer application
        visited_call = libcst.Call._visit_and_replace_children(self, visitor)
        return UnresolvedQuantizedCall(
            **node_asdict(visited_call), original_name=self.original_name, func_ref=self.func_ref
        )


@dataclasses.dataclass(slots=True, frozen=True)
class QuantizerReference(libcst.Name):
    """Node that references a specific quantizer.

    The quantizer referenced by this node is identified by `quantizer_info`.
    Each `QuantizerReference` that shares the same `quantizer_info` references
    the same quantizer.

    The `QuantizerInfo` object stores extra information on the quantizer, which
    can be changed at any time. Although CST nodes are immutable, the
    `quantizer_info` object is not. Moreover, in regular CST nodes, a full copy
    is made during visitor traversal. This node will copy all elements except
    for the `quantizer_info` object. This means that multiple nodes in multiple
    trees may reference the same `QuantizerInfo` object.
    """

    refid: int = dataclasses.field(kw_only=True, repr=False)

    @override
    def _visit_and_replace_children(self, visitor: libcst.CSTVisitorT) -> "QuantizerReference":
        return QuantizerReference(
            lpar=visit_sequence(self, "lpar", self.lpar, visitor),
            value=self.value,
            rpar=visit_sequence(self, "rpar", self.rpar, visitor),
            refid=self.refid,
        )

    @override
    def deep_clone(self: "QuantizerReference") -> "QuantizerReference":
        node = QuantizerReference(
            lpar=tuple(elem.deep_clone() for elem in self.lpar),
            value=self.value,
            rpar=tuple(elem.deep_clone() for elem in self.rpar),
            refid=self.refid,
        )
        return node


@dataclasses.dataclass(slots=True, frozen=True)
class AbstractClassReference(libcst.Name):
    """Subclass of Name representing a deferred class reference resolved at module build time.

    Used when the class name is not available during initial processing but can be determined later.

    Example:
        When creating quantized counterpart classes, we do not know the final
        class name during initial CST construction. AbstractClassReference
        allows us to reference this yet-to-be-determined class name in the CST,
        with the actual name resolved when the module is built.
    """

    def _visit_and_replace_children(self, visitor: libcst.CSTVisitorT) -> "AbstractClassReference":
        # This method must be implemented to prevent a 'downcast' to a
        # libcst.Call node during arbitrary transformer application
        visited_call = libcst.Name._visit_and_replace_children(self, visitor)
        return AbstractClassReference(**node_asdict(visited_call))


def node_asdict(node: libcst.CSTNode) -> dict[str, Any]:
    return {field.name: getattr(node, field.name) for field in dataclasses.fields(node)}
