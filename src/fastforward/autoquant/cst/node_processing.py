# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.


import dataclasses

from collections.abc import Iterator, Sequence

import libcst

from . import nodes


def normalize_assignments(
    node: nodes.GeneralAssignment,
) -> Iterator["NormalizedAssignment"]:
    """Normalize complex assignment statements into individual assignments.

    This function takes in `GeneralAssignment` nodes representing assignment
    statements. It breaks down complex assignments into separate
    `NormalizedAssignment` instances. If the assignment cannot be fully broken
    down based on static information, it returns a single
    `NormalizedAssignment` with the original targets and value.

    No normalized assignments are returned, if the given assignment statement
    represents a declaration.

    Args:
        node: A `GeneralAssignment` representing an assignment statement.

    Returns:
        Iterator[NormalizedAssignment]: An iterator of `NormalizedAssignment`
            instances representing the individual assignments. If the assignment
            cannot be fully broken down, a single `NormalizedAssignment` is
            returned with the original targets and value.

    Examples:
        >>> list(normalize_assignments(cst_node_for("ant, bat = cat, dog")))
        [
            NormalizedAssignment(
                targets=(Name( value="ant",),),
                value=Name( value="cat")
            ),
            NormalizedAssignment(
                value=Name( value="bat")
                targets=(Name( value="dog",),),
            )
        ]

        >>> list(normalize_assignments(cst_node_for("ant = bat = cat")))
        [
            NormalizedAssignment(
                targets=(Name( value="ant",),),
                value=Name( value="cat")
            ),
            NormalizedAssignment(
                value=Name( value="bat")
                targets=(Name( value="cat",),),
            )
        ]

        >>> list(normalize_assignments(cst_node_for("ant, bat = cat")))
        [
            NormalizedAssignment(
                targets=(Name( value="ant",), (Name( value="bat",)),
                value=Name( value="cat")
            )
        ]
    """
    # Skip declarations
    if node.value is None:
        return

    # Split up the assignment with multiple targets and process for each. Note
    # that a case with multiple targets represents a statement like `a = b = c`
    # and not `(a, b) = c`
    for target in node.targets:
        yield from _normalize_assignment(target, node.value)


@dataclasses.dataclass
class NormalizedAssignment:
    """Representation of a normalized assignment.

    In this context, a normalized assignment is an assignment for which nested
    or joint assignment have been eliminated as much as possible. This eases
    further analysis that require matching between the targets and value of an
    assignment.
    """

    targets: Sequence[libcst.BaseExpression]
    value: libcst.BaseExpression


def _normalize_assignment(
    target: libcst.BaseExpression, value: libcst.BaseExpression
) -> Iterator["NormalizedAssignment"]:
    """Implementation of `normalize_assignment` for a single target assignment."""
    match target, value:
        case libcst.Name(), _:
            yield NormalizedAssignment(targets=(target,), value=value)
        case libcst.Tuple() | libcst.List(), libcst.Tuple() | libcst.List():
            if _indexof_star(target) is not None:
                yield from _normalize_seq_to_starred_seq_assignment(target, value)
            else:
                yield from _normalize_seq_to_seq_assignment(target, value)
        case libcst.Tuple() | libcst.List(), _:
            yield from _normalize_item_to_seq_assignment(target, value)
        case _:
            yield NormalizedAssignment(targets=(target,), value=value)


def _normalize_seq_to_seq_assignment(
    target: libcst.Tuple | libcst.List, value: libcst.Tuple | libcst.List
) -> Iterator["NormalizedAssignment"]:
    """Process a sequence to sequence assignment without a 'starred' element.

    If there is an equal number of elements in the target and value, yield an
    `NormalizedAssignment` for every ith elements in the target and value
    sequence. Otherwise, yield a single joint assignment.
    """
    target_elements = tuple(elem.value for elem in target.elements)
    if len(target.elements) != len(value.elements):
        yield NormalizedAssignment(targets=target_elements, value=value)
        return

    for target_elem, value_elem in zip(target_elements, value.elements):
        yield from _normalize_assignment(target_elem, value_elem.value)


def _normalize_seq_to_starred_seq_assignment(
    target: libcst.Tuple | libcst.List, value: libcst.Tuple | libcst.List
) -> Iterator[NormalizedAssignment]:
    """Process a sequence to sequence assignment with a 'starred' element.

    The 'starred' element will capture multiple values in a list and is used
    for flexible unpacking. The number of elements that are captured depends on
    the other targets and the number of values. Here, we assign a flexible
    number of elements from value to the starred element.
    """
    star_idx = _indexof_star(target)
    assert star_idx is not None

    if len(value.elements) < len(target.elements) - 1:
        # This may be an error case, unless value has a single sequence element.
        # Here we yield a single joint assignment.
        yield NormalizedAssignment(
            targets=tuple(elem.value for elem in target.elements), value=value
        )
        return

    ntarget = len(target.elements)
    nvalue = len(value.elements)
    tail_size = ntarget - star_idx - 1

    # Yield all normalized assignments up to the starred element
    for i in range(0, star_idx):
        yield from _normalize_assignment(target.elements[i].value, value.elements[i].value)

    # Yield the starred element, capturing all required values in a list
    yield NormalizedAssignment(
        targets=(target.elements[star_idx].value,),
        value=libcst.List(elements=value.elements[star_idx : nvalue - tail_size]),
    )

    # Yield all normalized assignments after the starred element
    for i in range(-tail_size, 0):
        yield from _normalize_assignment(target.elements[i].value, value.elements[i].value)


def _normalize_item_to_seq_assignment(
    target: libcst.Tuple | libcst.List, value: libcst.BaseExpression
) -> Iterator["NormalizedAssignment"]:
    """Process an item to sequence assignment without a 'starred' element.

    Nested structures are unpacked and the returned targets represent the
    assignment targets. For example, an assignment like `(a, (b, c)) = d` is
    unpacked to a `NormalizedAssignment` with `a`, `b`, and `c` as targets and
    `d` as value.
    """
    target_elements = _unpack_target(target)
    yield NormalizedAssignment(targets=target_elements, value=value)


def _unpack_target(target: libcst.BaseExpression) -> tuple[libcst.BaseExpression, ...]:
    """Unpack (nested) targets into a flat tuple."""
    unpacked: list[libcst.BaseExpression] = []
    match target:
        case libcst.Tuple() | libcst.List():
            for elem in target.elements:
                unpacked += _unpack_target(elem.value)
            return tuple(unpacked)
        case _:
            return (target,)


def _indexof_star(sequence_node: libcst.Tuple | libcst.List) -> int | None:
    """Returns the index of a 'starred' element in a tuple or list if it exists, None otherwise."""
    for i, elem in enumerate(sequence_node.elements):
        if isinstance(elem, libcst.StarredElement):
            return i
    return None
