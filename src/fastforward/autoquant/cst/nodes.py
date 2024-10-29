"""
This module implements various CST Nodes. These are used to ease further code
analysis without destroying any information in the CST.

Each of the nodes implemented in this module will cooperate with a 'vanilla'
CST and can be used during code generation.
"""

import dataclasses

from typing import Any

import libcst

from libcst._nodes.internal import visit_required

from fastforward import _quantops


@dataclasses.dataclass(slots=True, frozen=True)
class ReplacementCandidate(libcst.BaseExpression):
    """
    A marker node that wraps an original BaseExpression that is a candidate for
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


@dataclasses.dataclass(slots=True, frozen=True)
class CandidateAssignment(libcst.BaseSmallStatement):
    """
    A marker node that wraps an original `libcst.Assign` node that is a candidate
    for further transformation.

    Any transformation is implemented separately and
    this node is only used to 'communicate' between different passes.
    """

    original: libcst.Assign

    def _codegen_impl(self, *args: Any, **kwargs: Any) -> None:
        self.original._codegen(*args, **kwargs)

    def _visit_and_replace_children(self, visitor: libcst.CSTVisitorT) -> libcst.CSTNode:
        return type(self)(visit_required(self, "original", self.original, visitor))


@dataclasses.dataclass(slots=True, frozen=True)
class CandidateReturn(libcst.BaseSmallStatement):
    """
    A marker node that wraps an original `libcst.Return` node that is a candidate
    for further transformation.

    Any transformation is implemented separately and
    this node is only used to 'communicate' between different passes.
    """

    original: libcst.Return

    def _codegen_impl(self, *args: Any, **kwargs: Any) -> None:
        self.original._codegen(*args, **kwargs)

    def _visit_and_replace_children(self, visitor: libcst.CSTVisitorT) -> libcst.CSTNode:
        return type(self)(visit_required(self, "original", self.original, visitor))


@dataclasses.dataclass(slots=True, frozen=True)
class CandidateYield(libcst.BaseExpression):
    """
    A marker node that wraps an original `libcst.Yield` node that is a candidate
    for further transformation.

    Any transformation is implemented separately and
    this node is only used to 'communicate' between different passes.
    """

    original: libcst.Yield

    def _codegen_impl(self, *args: Any, **kwargs: Any) -> None:
        self.original._codegen(*args, **kwargs)

    def _visit_and_replace_children(self, visitor: libcst.CSTVisitorT) -> libcst.CSTNode:
        return type(self)(visit_required(self, "original", self.original, visitor))


@dataclasses.dataclass(slots=True, frozen=True)
class AnnotatedCall(libcst.BaseExpression):
    """
    A metadata node that carries extra information and wraps a `libcst.Call`
    node. This wrapper node contains extra information on the quantized
    operation that is 'called'. This can be helpful in further analysis.
    """

    original: libcst.Call
    operator: _quantops.operator.Operator

    def _codegen_impl(self, *args: Any, **kwargs: Any) -> None:
        self.original._codegen(*args, **kwargs)

    def _visit_and_replace_children(self, visitor: libcst.CSTVisitorT) -> libcst.CSTNode:
        return type(self)(
            visit_required(self, "original", self.original, visitor),
            self.operator,
        )


@dataclasses.dataclass(slots=True, frozen=True)
class VersionedName(libcst.Name):
    """
    Extension of a `libcst.Name` node that also has a version number. This
    version number can be used to deduplicate assignments and references within
    a local scope, without changing the actual name. The version number has no
    effect on code generation.
    """

    version: int = 0
