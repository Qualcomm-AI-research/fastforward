# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

"""This module implements various CST Nodes. These are used to ease further code
analysis without destroying any information in the CST.

Each of the nodes implemented in this module will cooperate with a 'vanilla'
CST and can be used during code generation.
"""

import dataclasses

from dataclasses import dataclass
from typing import Any, Sequence

import libcst

from libcst._nodes.internal import visit_optional, visit_required, visit_sequence


@dataclasses.dataclass(slots=True, frozen=True)
class GeneralAssignment(libcst.BaseSmallStatement):
    """A universal node that wraps different types of assignments which are candidate for
    further transformation. This step is also called inflation.

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
    """A marker node that wraps an original BaseExpression that is a candidate for
    further transformation.

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
    """A metadata node that carries extra information and wraps a `libcst.Call`
    node. This wrapper node contains extra information on the quantized
    operation that is 'called'. This can be helpful in further analysis.
    """

    original_name: str = dataclasses.field(kw_only=True)
    operator: Any = dataclasses.field(kw_only=True)
