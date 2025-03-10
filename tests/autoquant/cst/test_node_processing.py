# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.


from collections.abc import Sequence

import libcst
import pytest

from fastforward.autoquant.cst import node_processing
from fastforward.autoquant.cst.nodes import GeneralAssignment
from fastforward.autoquant.cst.passes import WrapAssignments
from typing_extensions import override


def _assign_stmt(source: str) -> GeneralAssignment:
    """Helper to create assignment statements."""
    statement = libcst.parse_statement(source)
    assert isinstance(statement, libcst.SimpleStatementLine), "invalid parametrization"
    assignment = statement.body[0]
    assignment = assignment.visit(WrapAssignments())  # type: ignore[assignment]
    assert isinstance(assignment, GeneralAssignment), "invalid parametrization"
    return assignment


_expr = libcst.parse_expression
NormalizedAssignment = node_processing.NormalizedAssignment


def _normalized_assignment(targets: Sequence[str] | str, value: str) -> NormalizedAssignment:
    """Helper to create normalized assignments."""
    targets = (targets,) if isinstance(targets, str) else targets
    return NormalizedAssignment(
        targets=tuple(_expr(t) for t in targets),
        value=_expr(value),
    )


@pytest.mark.parametrize(
    "assignment,normalized",
    [
        # The structure of this parametrization is as follows:
        # Element 0: The assignment statement.
        # Element 1: A list of normalized assignments.
        (
            _assign_stmt("ant = bat"),
            [
                _normalized_assignment("ant", "bat"),
            ],
        ),
        (
            _assign_stmt("ant = bat = cat"),
            [
                _normalized_assignment("ant", "cat"),
                _normalized_assignment("bat", "cat"),
            ],
        ),
        (
            _assign_stmt("ant, bat = cat, dog"),
            [
                _normalized_assignment("ant", "cat"),
                _normalized_assignment("bat", "dog"),
            ],
        ),
        (
            _assign_stmt("[ant, bat] = [cat, dog]"),
            [
                _normalized_assignment("ant", "cat"),
                _normalized_assignment("bat", "dog"),
            ],
        ),
        (
            _assign_stmt("ant = bat, cat"),
            [
                _normalized_assignment("ant", "bat, cat"),
            ],
        ),
        (
            _assign_stmt("ant, bat = cat"),
            [
                _normalized_assignment(["ant", "bat"], "cat"),
            ],
        ),
        (
            _assign_stmt("ant, *bat = cat, dog, eel"),
            [
                _normalized_assignment("ant", "cat"),
                _normalized_assignment("bat", "[dog, eel]"),
            ],
        ),
        (
            _assign_stmt("*ant, bat = cat, dog, eel"),
            [
                _normalized_assignment("ant", "[cat, dog]"),
                _normalized_assignment("bat", "eel"),
            ],
        ),
        (
            _assign_stmt("ant, *bat, cat = dog, eel, fox, gar"),
            [
                _normalized_assignment("ant", "dog"),
                _normalized_assignment("bat", "[eel, fox]"),
                _normalized_assignment("cat", "gar"),
            ],
        ),
        (
            _assign_stmt("ant, *bat, cat = dog"),
            [
                _normalized_assignment(["ant", "bat", "cat"], "dog"),
            ],
        ),
        (
            _assign_stmt("ant, (bat, cat) = dog, (eel, fox)"),
            [
                _normalized_assignment("ant", "dog"),
                _normalized_assignment("bat", "eel"),
                _normalized_assignment("cat", "fox"),
            ],
        ),
        (
            _assign_stmt("(ant, *bat), cat = (dog, eel, fox), gar"),
            [
                _normalized_assignment("ant", "dog"),
                _normalized_assignment("bat", "[eel, fox]"),
                _normalized_assignment("cat", "gar"),
            ],
        ),
        (
            _assign_stmt("(ant, bat[0]), cat = dog"),
            [
                _normalized_assignment(["ant", "bat[0]", "cat"], "dog"),
            ],
        ),
        (
            _assign_stmt("ant, (bat, [cat, dog]) = (eel, fox)"),
            [
                _normalized_assignment(["ant"], "eel"),
                _normalized_assignment(["bat", "cat", "dog"], "fox"),
            ],
        ),
    ],
)
def test_normalize_assignments(
    assignment: GeneralAssignment,
    normalized: Sequence[NormalizedAssignment],
) -> None:
    # GIVEN a `GeneralAssignment` node

    # WHEN the given assignment is normalized
    observed_normalized = list(node_processing.normalize_assignments(assignment))
    normalized = list(normalized)

    # THEN: the cardinality of the expected and observed normalization must match
    assert len(observed_normalized) == len(normalized)

    for observed, expected in zip(observed_normalized, normalized):
        # THEN the number of targets for each normalization must match expectations
        assert len(observed.targets) == len(expected.targets)
        for target_observed, target_expected in zip(observed.targets, expected.targets):
            # THEN the order and target must match expectations
            assert _node_equals(target_observed, target_expected)
        # THEN the value of the normalization must match expectations.
        assert _node_equals(observed.value, expected.value)


def _node_equals(left: libcst.CSTNode, right: libcst.CSTNode) -> bool:
    """Helper to test two nodes for equality.

    This function removes some whitespace information to normalize nodes before
    comparison.
    """
    strip_comma = _StripCommaMetadata()
    left_stripped = left.visit(strip_comma)
    right_stripped = right.visit(strip_comma)
    assert isinstance(left_stripped, libcst.CSTNode)
    assert isinstance(right_stripped, libcst.CSTNode)
    return left_stripped.deep_equals(right_stripped)


class _StripCommaMetadata(libcst.CSTTransformer):
    """Strip comma metadata from `libcst.Element` nodes."""

    @override
    def leave_Element(
        self, original_node: libcst.Element, updated_node: libcst.Element
    ) -> libcst.Element:
        return updated_node.with_changes(comma=libcst.MaybeSentinel.DEFAULT)
