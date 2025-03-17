# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


from typing import Any

import libcst
import pytest

from fastforward.autoquant.cfg import blocks, variable_tracking
from fastforward.autoquant.cst.passes import WrapAssignments
from fastforward.autoquant.pysource import SourceContext
from typing_extensions import override

from tests.autoquant.cfg.cfg_test import CFGTest


def _variable(name: str, version: int) -> variable_tracking.Variable:
    """Create a variable with `name` and `version`."""
    block = blocks.ExitBlock()
    return variable_tracking.Variable(name=name, version=version, declaration_block=block)


@pytest.fixture()
def variable_set() -> variable_tracking.VariableSet:
    """Create a variable set with two members: ant:2, bat:0."""
    var_set = variable_tracking.VariableSet()
    var_set.add(_variable(name="ant", version=2))
    var_set.add(_variable(name="bat", version=0))
    return var_set


@pytest.mark.parametrize("name,version", [("ant", 2), ("bat", 0)])
def test_variable_set_contains(
    variable_set: variable_tracking.VariableSet, name: str, version: int
) -> None:
    # GIVEN a variable set of two elements containing ant:2 and bat:0
    assert len(variable_set) == 2

    # THEN name:version must be a member of the variable set through all
    # methods for member testing.
    assert variable_set.contains(name)
    assert variable_set.contains(name, version)
    assert _variable(name, version) in variable_set
    assert variable_set.contains(_variable(name, version))


@pytest.mark.parametrize("name,version", [("ant", 0), ("bat", 3), ("cat", 0)])
def test_variable_set_not_contains(
    variable_set: variable_tracking.VariableSet, name: str, version: int
) -> None:
    # GIVEN a variable set of two elements containing ant:2 and bat:0
    assert len(variable_set) == 2

    # THEN name:version must not be a member of the variable set through all
    # methods for member testing.
    if name not in ("ant", "bat"):  # 'ant' and 'bat' are members, so skip this assertion
        assert not variable_set.contains(name)
    assert not variable_set.contains(name, version)
    assert _variable(name, version) not in variable_set
    assert not variable_set.contains(_variable(name, version))


def test_variable_set_remove(variable_set: variable_tracking.VariableSet) -> None:
    # GIVEN a variable set of two elements
    assert len(variable_set) == 2

    # WHEN a variable is removed from the variable set
    variable_set.remove(_variable("ant", 2))

    # THEN it must no longer be a member of the variable set
    assert _variable("ant", 2) not in variable_set

    # WHEN a variable is removed from the variable set
    variable_set.remove("bat")

    # THEN it must no longer be a member of the variable set
    assert not variable_set.contains("bat")

    # THEN the variable set must be empty
    assert len(variable_set) == 0


def test_variable_set_union() -> None:
    # GIVEN two variable sets {ant:0, bat:2} and {ant:0, cat:1}
    varset1 = variable_tracking.VariableSet()
    varset2 = variable_tracking.VariableSet()

    varset1.add(var1 := _variable("ant", 0))
    varset1.add(var2 := _variable("bat", 2))
    varset2.add(_variable("ant", 0))
    varset2.add(var3 := _variable("cat", 1))

    # WHEN a union between both variable sets is created
    union = varset1.union(varset2)

    # THEN the resulting set must be a proper set union of both
    assert len(union) == 3
    assert var1 in union
    assert var2 in union
    assert var3 in union


def test_variable_set_subtract() -> None:
    # GIVEN two variable sets {ant:0, bat:2} and {ant:0, cat:1}
    varset1 = variable_tracking.VariableSet()
    varset2 = variable_tracking.VariableSet()

    varset1.add(_variable("ant", 0))
    varset1.add(remaining_var := _variable("bat", 2))
    varset2.add(_variable("ant", 0))
    varset2.add(_variable("cat", 1))

    # WHEN a the second set is subtracted from the first
    subtraction = varset1.subtract(varset2)

    # THEN the resulting set must be a proper set subtraction
    assert len(subtraction) == 1
    assert remaining_var in subtraction


def test_variable_set_subtract_all_versions() -> None:
    # GIVEN two variable sets {ant:0, bat:2, ant:5} and {ant:0, cat:1}
    varset1 = variable_tracking.VariableSet()
    varset2 = variable_tracking.VariableSet()

    varset1.add(_variable("ant", 0))
    varset1.add(_variable("ant", 5))
    varset1.add(remaining_var := _variable("bat", 2))
    varset2.add(_variable("ant", 0))
    varset2.add(_variable("cat", 1))

    # WHEN a the second set is subtracted from the first, ignoring versions
    subtraction = varset1.subtract(varset2, all_versions=True)

    # THEN the resulting set must be a proper set subtraction
    assert len(subtraction) == 1
    assert remaining_var in subtraction


def test_variable_set_eq() -> None:
    # GIVEN three variable sets of which the first two are equal and the last
    # is different
    varset1 = variable_tracking.VariableSet()
    varset2 = variable_tracking.VariableSet()
    varset3 = variable_tracking.VariableSet()

    varset1.add(_variable("ant", 0))
    varset1.add(_variable("bat", 2))
    varset2.add(_variable("ant", 0))
    varset2.add(_variable("bat", 2))
    varset3.add(_variable("ant", 0))
    varset3.add(_variable("cat", 1))

    # THEN the resulting set equality must follow normal expectations
    assert varset1 == varset2
    assert varset1 != varset3
    assert varset2 != varset3


def test_ordered_set() -> None:
    # GIVEN an ordered set
    ordered_set: variable_tracking._OrderedSet[int] = variable_tracking._OrderedSet()

    # WHEN elements are added multiple times
    for insertion in (1, 2, 1, 2, 3, 2, 1):
        ordered_set.add(insertion)

    # THEN elements are popped in the reverse order of first insertion
    assert ordered_set.pop() == 3
    assert ordered_set.pop() == 2
    assert ordered_set.pop() == 1
    # THEN the set is empty if all elements are popped
    assert ordered_set.pop(-1) == -1

    # WHEN elements are added multiple times
    for insertion in (1, 2, 1, 2, 3, 2, 1):
        ordered_set.add(insertion)

    # WHEN an elements is removed
    ordered_set.remove(2)

    # THEN elements are popped in the reverse order of first insertion,
    # skipping the removed element
    assert ordered_set.pop() == 3
    assert ordered_set.pop() == 1
    # THEN the set is empty after all elements are popped
    assert ordered_set.pop(-1) == -1


class TestVariableReachability(CFGTest):
    @pytest.fixture(scope="class")
    def source_context(self) -> SourceContext:
        """Overwrite default `source_context`.

        Apply `WrapAssignments` to every case when creating CSTs.
        """
        return SourceContext(preprocessing_passes=(WrapAssignments(),))

    def test_variable_reachability(self, cfg: blocks.Block) -> None:
        """Test if reachability analysis on top of CFG is correct.

        This test function is mostly a runner for the test cases that are defined
        at the bottom of this file.
        """
        # GIVEN a CFG for a function that contains reachability assertions.

        # WHEN dataflow analysis is performed on top of the CFG.
        block_vars = variable_tracking.infer_block_dataflow(cfg)

        # THEN the obtained set of incoming (vars_in) and outgoing (vars_out)
        # variables must agree with the reachability assertions for each block.
        # Please see the comment block labeled "TEST CASES" below for more context.
        for block in cfg.blocks():
            if not isinstance(block, blocks.SimpleBlock):
                continue
            vars_in = block_vars[block].vars_in
            vars_out = block_vars[block].vars_out
            visitor = _ReachabilityAssertionVisitor(
                vars_in,
                vars_out,
            )
            for line in block.statements:
                _ = line.visit(visitor)

    # --------------------------------------------------------------------------
    #                                TEST CASES
    # --------------------------------------------------------------------------
    # The following functions are not executed, but will be parsed into a CFG.
    # The function names and what they do are irrelevant; only the control flow
    # matters.
    #
    # These functions contain `assert_reaches` and `assert_not_reaches` calls.
    # These calls serve as labels to a Visitor to trigger a check if a
    # particular version of a variable can reach or does not reach the point
    # where the call is made. They cannot be used to verify the creation of a
    # new variable version within a block.
    #
    # Note: The exact version numbers of variables are an implementation detail
    # and might change due to implementation updates. Therefore, the tests
    # below include comments indicating the expected version number for each
    # assignment. It's recommended to verify these annotations when a test
    # fails. To assist with this, `assert_provides` assertions are included in
    # the test cases. These assertions confirm that the block produces the
    # expected variable versions. If a test fails on one of these assertions,
    # updating the version numbers to match the actual values assigned by the
    # implementation should likely fix the issue.
    # --------------------------------------------------------------------------

    def case_if_statement1(ant: int, bat: int) -> None:  # type: ignore[misc]
        """All parameters of a function must reach the exit block for a function without a body."""
        assert_reaches(ant, 0)
        assert_reaches(bat, 0)

    def case_if_statement2(ant: int, bat: int) -> None:  # type: ignore[misc]
        if ant > bat:
            assert_reaches(ant, 0)
            assert_reaches(bat, 0)
            cat = ant  # cat:0
            assert_provides(cat, 0)
        else:
            assert_reaches(ant, 0)
            assert_reaches(bat, 0)
            dog = bat  # dog:0
            assert_provides(dog, 0)
        assert_reaches(ant, 0)
        assert_reaches(bat, 0)
        assert_reaches(cat, 0)  # possibly unbound, but can be used for CFG-based evaluation
        assert_reaches(dog, 0)  # possibly unbound, but can be used for CFG-based evaluation

    def case_if_statement3(ant: int, bat: int) -> None:  # type: ignore[misc]
        if ant > bat:
            assert_reaches(ant, 0)
            assert_reaches(bat, 0)
            ant = bat  # ant:1
            assert_provides(ant, 1)
        else:
            assert_reaches(ant, 0)
            assert_reaches(bat, 0)
            ant = ant * 2  # ant:2
            assert_provides(ant, 2)
        assert_reaches(ant, 1)
        assert_reaches(ant, 2)
        assert_reaches(bat, 0)
        assert_not_reaches(ant, 0)

    def case_if_statement4(ant: int, bat: int) -> None:  # type: ignore[misc]
        if ant > bat:
            assert_reaches(ant, 0)
            assert_reaches(bat, 0)
            ant = bat  # ant:1
            assert_provides(ant, 1)
        else:
            assert_reaches(ant, 0)
            assert_reaches(bat, 0)
            bat = ant  # bat:1
            assert_provides(bat, 1)
        assert_reaches(ant, 0)
        assert_reaches(bat, 0)
        assert_reaches(ant, 1)
        assert_reaches(bat, 1)

    def case_if_statement5(ant: int, bat: int, flag: bool) -> None:  # type: ignore[misc]
        if ant > bat:
            assert_reaches(ant, 0)
            assert_reaches(bat, 0)
            ant = bat  # ant:1
            assert_provides(ant, 1)
        elif bat > ant:
            assert_reaches(ant, 0)
            assert_reaches(bat, 0)
            ant, bat = bat, ant  # ant:3, bat:2
            assert_provides(ant, 3)
            assert_provides(bat, 2)
        else:
            assert_reaches(ant, 0)
            assert_reaches(bat, 0)
            bat = ant  # bat:3
            assert_provides(bat, 3)
        assert_reaches(ant, 0)
        assert_reaches(ant, 1)
        assert_reaches(ant, 3)
        assert_reaches(bat, 0)
        assert_reaches(bat, 2)
        assert_reaches(bat, 3)
        ant, bat = bat, ant  #  ant:2, bat:1
        assert_provides(ant, 2)
        assert_provides(bat, 1)
        if flag:  # Included to force the creation of new blocks.
            assert_reaches(ant, 2)
            assert_reaches(bat, 1)
            assert_not_reaches(ant, 0)
            assert_not_reaches(ant, 1)
            assert_not_reaches(ant, 3)
        assert_reaches(ant, 2)
        assert_reaches(bat, 1)
        assert_not_reaches(ant, 0)
        assert_not_reaches(ant, 1)
        assert_not_reaches(ant, 3)


class _ReachabilityAssertionVisitor(libcst.CSTVisitor):
    """CST node visitor that acts on reachability assertions.

    In particular, any call to `assert_reaches`, `assert_not_reaches` and
    `assert_provides` is validated against the set of incoming and outgoing
    variables for a block. If the assertions do not match the inferred variables
    an `AssertionError` is raised.

    Args:
        in_variables: The variable set that represents the incoming variables
            to a block.
        out_variables: The variable set that represents the outgoing variables
            to a block.
    """

    def __init__(
        self,
        in_variables: variable_tracking.VariableSet,
        out_variables: variable_tracking.VariableSet,
    ) -> None:
        super().__init__()
        self.in_variables = in_variables
        self.out_variables = out_variables

    @override
    def visit_Call(self, node: libcst.Call) -> bool:
        match node.func:
            case libcst.Name(value=assert_reaches.__name__):
                self._assert_reaches(node)
            case libcst.Name(value=assert_not_reaches.__name__):
                self._assert_not_reaches(node)
            case libcst.Name(value=assert_provides.__name__):
                self._assert_provides(node)
            case _:
                pass

        return True

    def _get_name_and_version(self, node: libcst.Call, caller: str) -> tuple[str, int]:
        """Get name and version for an assertion call."""
        match node:
            case libcst.Call(
                args=[
                    libcst.Arg(libcst.Name(value=name)),
                    libcst.Arg(libcst.Integer(value=version_str)),
                ]
            ):
                return name, int(version_str)
            case _:
                pass

        msg = (
            f"'{caller}' must be called with two positional-only arguments, "
            + "the first being a variable and the second an integer."
        )
        raise TypeError(msg)

    def _assert_reaches(self, node: libcst.Call) -> None:
        name, version = self._get_name_and_version(node, "assert_reaches")

        assert self.in_variables.contains(name, version), (
            f"Expected {name}:{version} to be reachable. Reachable variables: {self.in_variables}"
        )

    def _assert_provides(self, node: libcst.Call) -> None:
        name, version = self._get_name_and_version(node, "assert_provides")

        assert self.out_variables.contains(name, version), (
            f"Expected {name}:{version} to be provided. Provided variables: {self.out_variables}"
        )

    def _assert_not_reaches(self, node: libcst.Call) -> None:
        name, version = self._get_name_and_version(node, "assert_not_reaches")

        assert not self.in_variables.contains(name, version), (
            f"Expected {name}:{version} to be not reachable. Reachable variables: {self.in_variables}"
        )


def assert_reaches(_var: Any, _version: int, /) -> None:
    """Serves as a label to `_ReachabilityAssertionVisitor` to assert variable reachability.

    Signifies to `_ReachabilityAssertionVisitor` to test if `var` with
    `version` reaches the start of the block of the call site.

    This function is a no-op and the assert is performed on top of the CFG of
    the function of the call site.
    """


def assert_not_reaches(_var: Any, _version: int, /) -> None:
    """Serves as a label to `_ReachabilityAssertionVisitor` to assert variable reachability.

    Signifies to `_ReachabilityAssertionVisitor` to test if `var` with
    `version` does not reach the start of the block of the call site.

    This function is a no-op and the assert is performed on top of the CFG of
    the function of the call site.
    """


def assert_provides(_var: Any, _version: int, /) -> None:
    """Serves as a label to `_ReachabilityAssertionVisitor` to assert variable creation.

    Signifies to `_ReachabilityAssertionVisitor` to test if `var` is in
    the out set of the block of the call site.

    This function is a no-op and the assert is performed on top of the CFG of
    the function of the call site.
    """
