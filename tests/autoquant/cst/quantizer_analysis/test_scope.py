# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


import libcst
import pytest

from fastforward._autoquant.cst.quantizer_analysis.scope import (
    QuantizationMetadata,
    Scope,
    _Assignment,
    _Assignments,
    _QuantizedAssignments,
)


@pytest.fixture
def node() -> libcst.CSTNode:
    return libcst.Module([])


def test_Assignments_record_assignment(node: libcst.CSTNode) -> None:
    # GIVEN: An empty _Assignments object
    assignments = _Assignments[bool]({})

    # WHEN: An assignment is recorded
    assignments.record_assignment("ant", node, True)

    # THEN: The variable and node are stored in the assignments
    assert "ant" in assignments
    assert ("ant", node) in assignments
    assert ("ant", node.with_changes()) not in assignments


@pytest.mark.parametrize("init_metadata", [True, False])
def test_Assignments_record_assignment_update(node: libcst.CSTNode, init_metadata: bool) -> None:
    # GIVEN: An _Assignments object with an existing assignment
    assignments = _Assignments[bool]({})
    assignments.record_assignment("ant", node, not init_metadata)

    # WHEN: The same assignment is recorded again with a different quantization status
    assignments.record_assignment("ant", node, init_metadata)

    # THEN: The quantization status of the existing assignment is updated
    assert assignments["ant", node].metadata == init_metadata


@pytest.mark.parametrize("init_metadata", [True, False])
def test_QuantizedAssignments_record_assignment_update(
    node: libcst.CSTNode, init_metadata: bool
) -> None:
    # GIVEN: An _Assignments object with an existing assignment
    assignments = _QuantizedAssignments({})
    assignments.record_assignment("ant", node, QuantizationMetadata(not init_metadata))

    # WHEN: The same assignment is recorded again with a different quantization status
    assignments.record_assignment("ant", node, QuantizationMetadata(init_metadata))

    # THEN: The quantization status of the existing assignment is updated
    assert assignments["ant", node].metadata.is_quantized


def test_Assignments_getitem_variable(node: libcst.CSTNode) -> None:
    # GIVEN: An _Assignments object with multiple assignments for different variables
    assignments = _Assignments[bool]({})
    assignments.record_assignment("ant", node, True)
    assignments.record_assignment("bat", node, True)

    # WHEN: The assignments for a specific variable are retrieved
    statuses = list(assignments["ant"])

    # THEN: The assignments for the specified variable are returned
    assert assignments[("ant", node)] in statuses
    assert len(statuses) == 1


def test_Assignments_getitem_producer(node: libcst.CSTNode) -> None:
    # GIVEN: An _Assignments object with an assignment
    assignments = _Assignments[bool]({})
    assignments.record_assignment("ant", node, True)

    # WHEN: The assignment for a specific variable and node is retrieved
    status = assignments[("ant", node)]

    # THEN: The assignment is returned with the correct producer and name
    assert isinstance(status, _Assignment)
    assert status.producer is node
    assert status.name == "ant"


def test_Assignments_getitem_not_found(node: libcst.CSTNode) -> None:
    # GIVEN: An empty _Assignments object
    assignments = _Assignments[bool]({})

    # WHEN: An assignment for a non-existent variable and node is retrieved
    # THEN: A KeyError is raised
    with pytest.raises(KeyError):
        assignments[("ant", node)]


def test_Assignments_contains_variable(node: libcst.CSTNode) -> None:
    # GIVEN: An _Assignments object with an assignment
    assignments = _Assignments[bool]({})
    assignments.record_assignment("ant", node, True)

    # WHEN: The presence of a variable in the assignments is checked
    # THEN: The variable is found in the assignments
    assert "ant" in assignments


def test_Assignments_contains_producer(node: libcst.CSTNode) -> None:
    # GIVEN: An _Assignments object with an assignment
    assignments = _Assignments[bool]({})
    assignments.record_assignment("ant", node, True)

    # WHEN: The presence of a variable and node in the assignments is checked
    # THEN: The variable and node are found in the assignments
    assert ("ant", node) in assignments


def test_Assignments_iter() -> None:
    # GIVEN: An _Assignments object with multiple assignments
    assignments = _Assignments[bool]({})

    node1: libcst.CSTNode = libcst.Module([])
    node2: libcst.CSTNode = libcst.IndentedBlock([])

    assignments.record_assignment("ant", node1, True)
    assignments.record_assignment("bat", node2, False)

    # WHEN: The assignments are iterated over
    statuses = list(assignments)

    # THEN: All assignments are returned
    assert len(statuses) == 2


def test_Assignments_variables(node: libcst.CSTNode) -> None:
    # GIVEN: An _Assignments object with an assignment
    assignments = _Assignments[bool]({})
    assignments.record_assignment("ant", node, True)

    # WHEN: The variables in the assignments are retrieved
    variables = list(assignments.variables)

    # THEN: The variables are returned
    assert len(variables) == 1
    assert variables[0] == "ant"


def test_Assignments_clone(node: libcst.CSTNode) -> None:
    # GIVEN: An _Assignments object with an assignment
    assignments = _Assignments[bool]({})
    assignments.record_assignment("ant", node, True)

    # WHEN: The assignments are cloned
    clone = assignments.clone()

    # THEN: The clone is a separate object with the same assignments
    assert assignments is not clone
    assert clone._assignments == assignments._assignments


def test_Assignments_merge() -> None:
    # GIVEN: Two _Assignments objects with different assignments
    assignments1 = _Assignments[bool]({})
    assignments2 = _Assignments[bool]({})

    node1: libcst.CSTNode = libcst.Module([])
    node2: libcst.CSTNode = libcst.IndentedBlock([])

    assignments1.record_assignment("ant", node1, True)
    assignments2.record_assignment("bat", node2, False)

    # WHEN: The assignments are merged
    merged = assignments1.merge(assignments2)

    # THEN: The merged assignments contain all assignments from both objects
    assert ("ant", node1) in merged
    assert ("bat", node2) in merged
    assert len(list(merged)) == 2


def test_Assignments_overwrite() -> None:
    # GIVEN: Two _Assignments objects with assignments for the same variable but different nodes
    assignments1 = _Assignments[bool]({})
    assignments2 = _Assignments[bool]({})

    node1: libcst.CSTNode = libcst.Module([])
    node2: libcst.CSTNode = libcst.IndentedBlock([])

    assignments1.record_assignment("ant", node1, True)
    assignments2.record_assignment("ant", node2, False)
    assert assignments1._assignments != assignments2._assignments

    # WHEN: The assignments in the first object are overwritten by the second object
    assignments1.overwrite(assignments2)

    # THEN: The assignments in the first object are updated to match the second object
    assert len(list(assignments1)) == 1
    assert ("ant", node2) in assignments1


def test_Scope_clone() -> None:
    # GIVEN: A Scope instance with some attributes set
    scope = Scope[bool]()
    scope.parent = Scope[bool]()
    scope.assignments = _Assignments({})
    scope.record_assignment("ant", libcst.Module([]), True)
    scope.is_looping_branch = True
    scope.repeated_evaluation = True

    # WHEN: Cloning the instance
    cloned_scope = scope.clone()

    # THEN: The cloned instance should have the same attributes
    assert cloned_scope.parent is scope.parent
    assert cloned_scope.assignments == scope.assignments
    assert cloned_scope.is_looping_branch is scope.is_looping_branch
    assert cloned_scope.repeated_evaluation is scope.repeated_evaluation


def test_Scope_record_assignment() -> None:
    # GIVEN: A Scope instance
    scope = Scope[bool]()

    # WHEN: Recording an assignment
    node = libcst.IndentedBlock([])
    scope.record_assignment("ant", node, True)

    # THEN: The assignment should be added to the scope's assignments
    assert list(scope.assignments["ant"]) == list([_Assignment("ant", node, True)])


def test_Scope_record_assignment_in_terminated_scope() -> None:
    # GIVEN: A terminated Scope instance
    scope = Scope[bool]()
    scope.terminate("return")

    # WHEN: Recording an assignment
    # THEN: A RuntimeError should be raised
    with pytest.raises(RuntimeError):
        scope.record_assignment("a", libcst.Module([]), True)


def test_Scope_terminate() -> None:
    # GIVEN: A Scope instance
    scope = Scope[bool]()

    # WHEN: Terminating the scope
    scope.terminate("return")

    # THEN: The scope's termination status should be set
    assert scope._termination_status == "return"


def test_Scope_terminate_with_break() -> None:
    # GIVEN: A Scope instance
    scope = Scope[bool]()
    scope.is_looping_branch = True
    scope.repeated_evaluation = True

    # WHEN: Terminating the scope with a break reason
    scope.terminate("break")

    # THEN: The scope's termination status should be set and the scope
    # assignments should be empty.
    assert scope._termination_status == "break"
    assert scope.assignments == _Assignments({})


@pytest.mark.parametrize("metadata", [True, False])
def test_Scope_getitem(metadata: bool) -> None:
    # GIVEN: A Scope instance with some assignments
    scope = Scope[bool]()
    node = libcst.IndentedBlock([])
    scope.record_assignment("a", node, metadata)

    # WHEN: Getting the assignment for a variable
    status = list(scope["a"])[0]

    # THEN: The metadata should be returned
    assert status.producer == node
    assert status.metadata == metadata


def test_Scope_merge() -> None:
    # GIVEN: Two Scope instances
    scope1 = Scope[bool]()
    scope2 = Scope[bool]()
    node1 = libcst.Module([])
    node2 = libcst.IndentedBlock([])
    scope1.record_assignment("ant", node1, True)
    scope2.record_assignment("bat", node2, False)

    # WHEN: Merging the scopes
    merged_scope = scope1.merge(scope2)

    # THEN: The merged scope should have the correct attributes
    assert list(merged_scope.assignments["ant"]) == [_Assignment("ant", node1, True)]
    assert list(merged_scope.assignments["bat"]) == [_Assignment("bat", node2, False)]


def test_Scope_merge_with_termination() -> None:
    # GIVEN: Two Scope instances with termination
    scope1 = Scope[bool]()
    scope2 = Scope[bool]()

    node1 = libcst.Module([])
    node2 = libcst.IndentedBlock([])
    node3 = libcst.Name("name")

    scope1.record_assignment("ant", node1, True)
    scope1.record_assignment("bat", node2, True)

    scope2.record_assignment("ant", node3, False)
    scope2.terminate("return")

    # WHEN: Merging the scopes
    merged_scope = scope1.merge(scope2)

    # THEN: The merged scope should have the correct attributes
    assert list(merged_scope["ant"]) == list(scope1["ant"])
    assert list(merged_scope["bat"]) == list(scope1["bat"])


def test_Scope_overwrite() -> None:
    # GIVEN: Two Scope instances
    scope1 = Scope[bool]()
    scope2 = Scope[bool]()

    node1 = libcst.Module([])
    node2 = libcst.IndentedBlock([])
    scope1.record_assignment("ant", node1, True)
    scope2.record_assignment("ant", node2, False)

    # WHEN: Overwriting one scope with another
    scope1.overwrite(scope2)
    # THEN: The overwritten scope should have the correct attributes
    assert list(scope1.assignments["ant"]) == [_Assignment("ant", node2, False)]
