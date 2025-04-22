# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""This module implements various CST Nodes used for code analysis.

These CSTNodes are used to ease further code analysis without destroying any
information in the CST.

Each of the nodes implemented in this module will cooperate with a 'vanilla'
CST and can be used during code generation.
"""

import dataclasses

from dataclasses import dataclass, field
from typing import Any, Sequence

import libcst

from libcst._nodes.internal import CodegenState, visit_optional, visit_required, visit_sequence
from typing_extensions import Self, override

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


@dataclass(slots=True, frozen=True)
class QuantizedCall(libcst.Call):
    """A metadata node that carries extra information and wraps a `libcst.Call`.

    This wrapper node contains extra information on the quantized
    operation that is 'called'. This can be helpful in further analysis.
    """

    original_name: str = dataclasses.field(kw_only=True)
    operator: Operator = dataclasses.field(kw_only=True)


@dataclass(slots=True, frozen=True)
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

    @dataclass(eq=False)
    class QuantizerInfo:
        """Quantizer specific information."""

        base_name: str

    quantizer_info: QuantizerInfo = field(init=False)

    @classmethod
    def from_quantizer_info(cls, quantizer_info: QuantizerInfo) -> "QuantizerReference":
        """Construct a new `QuantizerReference` from a `QuantizerInfo` object."""
        quantizer_ref = cls(quantizer_info.base_name)
        quantizer_ref._set_quantizer_info(quantizer_info)
        return quantizer_ref

    def __post_init__(self) -> None:
        self._set_quantizer_info(QuantizerReference.QuantizerInfo(self.value))

    def _set_quantizer_info(self, name_info: QuantizerInfo) -> None:
        # In LibCST, CST nodes are immutable and new nodes are created after
        # visiting/transforming. This means we cannot identify a node in a tree
        # across visits or transforms by means of the node's id. Here, we
        # introduces a `QuantizerInfo` object that holds both contextual
        # information about the quantizer  and acts as a 'identification
        # token'. I.e., every `QuantizerReference` node that has the same
        # `quantizer_info` object references the same quantizer.
        object.__setattr__(self, "quantizer_info", name_info)

    @override
    def _visit_and_replace_children(self, visitor: libcst.CSTVisitorT) -> "QuantizerReference":
        node = QuantizerReference(
            lpar=visit_sequence(self, "lpar", self.lpar, visitor),
            value=self.value,
            rpar=visit_sequence(self, "rpar", self.rpar, visitor),
        )
        node._set_quantizer_info(self.quantizer_info)
        return node

    @override
    def with_changes(self, **changes: Any) -> Self:
        if "value" in changes:
            raise ValueError(f"Cannot change the 'value' attribute of a {type(self).__name__}")
        return super().with_changes(**changes)

    @override
    def _codegen_impl(self, state: CodegenState) -> None:
        with self._parenthesize(state):
            state.add_token(self.quantizer_info.base_name)

    @override
    def deep_clone(self: "QuantizerReference") -> "QuantizerReference":
        node = QuantizerReference(
            lpar=tuple(elem.deep_clone() for elem in self.lpar),
            value=self.value,
            rpar=tuple(elem.deep_clone() for elem in self.rpar),
        )
        node._set_quantizer_info(self.quantizer_info)
        return node
