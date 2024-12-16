# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import dataclasses

from typing import (
    Protocol,
    Sequence,
    TypeAlias,
    TypeVar,
    Union,
    cast,
    runtime_checkable,
)

import libcst
import libcst.helpers
import libcst.matchers as m

import fastforward as ff

from fastforward._quantops.optable import OperatorTable

from .nodes import (
    AnnotatedCall,
    CandidateAssignment,
    CandidateReturn,
    CandidateYield,
    GeneralAssignment,
    ReplacementCandidate,
)


@runtime_checkable
class HasLeadingLines(Protocol):
    leading_lines: Sequence[libcst.EmptyLine]


_LineStatement: TypeAlias = libcst.BaseStatement
_BaseStatements: TypeAlias = libcst.SimpleStatementLine | libcst.BaseCompoundStatement
_SuiteT = TypeVar("_SuiteT", libcst.IndentedBlock, libcst.Module)


class SimpleStatementSuiteToIndentedBlock(libcst.CSTTransformer):
    """
    Convert SimpleStatementSuite to IndentedBlock.

    This converts single line if, for, and other statements to
    statements with an indented block.

    For example:

        if <test>: a=10; b=100

    is converted to

        if <test>:
            a=10
            b=100

    Once this transformer concludes, there are no `SimpleStatementSuite`s in
    the CST. This eases further analysis, but does result on more code rewrites
    than strictly required.
    """

    def leave_SimpleStatementSuite(
        self,
        original_node: libcst.SimpleStatementSuite,
        updated_node: libcst.SimpleStatementSuite,
    ) -> libcst.BaseSuite:
        del original_node
        new_body: list[libcst.SimpleStatementLine] = []
        semicolon = libcst.BaseSmallStatement.semicolon  # use default semicolon
        for statement in updated_node.body:
            line = libcst.SimpleStatementLine(body=[statement.with_changes(semicolon=semicolon)])
            new_body.append(line)
        return libcst.IndentedBlock(body=tuple(new_body))


class WrapAssignments(libcst.CSTTransformer):
    def leave_AnnAssign(
        self, _original_node: libcst.AnnAssign, updated_node: libcst.AnnAssign
    ) -> GeneralAssignment:
        return GeneralAssignment(
            original=_original_node,
            targets=(updated_node.target,),
            annotation=updated_node.annotation,
            value=updated_node.value,
        )

    def leave_Assign(
        self, _original_node: libcst.Assign, updated_node: libcst.Assign
    ) -> GeneralAssignment:
        return GeneralAssignment(
            original=_original_node,
            targets=tuple(x.target for x in updated_node.targets),
            annotation=None,
            value=updated_node.value,
        )

    def leave_AugAssign(
        self, _original_node: libcst.AugAssign, updated_node: libcst.AugAssign
    ) -> GeneralAssignment:
        return GeneralAssignment(
            original=_original_node,
            targets=(updated_node.target,),
            annotation=None,
            value=updated_node.value,
        )


class MarkReplacementCandidates(libcst.CSTTransformer):
    """
    Mark nodes in the CST for replacement by future passes. Marking is performed
    by wrapping the CST node in a ReplacementCandidate node.
    """

    def leave_Call(
        self, original_node: libcst.Call, updated_node: libcst.Call
    ) -> libcst.BaseExpression:
        if m.matches(original_node.func, m.Name("super")):
            return updated_node
        return ReplacementCandidate(updated_node)

    def leave_BinaryOperation(
        self,
        original_node: libcst.BinaryOperation,
        updated_node: libcst.BinaryOperation,
    ) -> libcst.BaseExpression:
        del original_node
        return ReplacementCandidate(updated_node)


@dataclasses.dataclass
class _NodeInsertion:
    target: libcst.CSTNode
    insertion: libcst.BaseSmallStatement
    position: _LineStatement


_Statement: TypeAlias = libcst.BaseStatement | libcst.BaseSmallStatement
_PositionMap: TypeAlias = dict[_Statement, list[_NodeInsertion]]


@dataclasses.dataclass
class _Insertions:
    """
    Insertion collection. Collects insertions during traversal of a CST and
    associates each with an insertion point.

    An insertion is a CST node with extra information that should be added to
    a compound statement (e.g., an IndentedBlock) at a certain position in the
    indented block. Insertions are represented by `_NodeInsertion`s.
    """

    # Map T -> I for CST node T and insertion I. T denotes the target for
    # insertion, e.g., an IndentedBlock in which a statement should be
    # inserted. I contains all information on the insertion, including the
    # target.
    insertions: dict[libcst.CSTNode, list[_NodeInsertion]] = dataclasses.field(default_factory=dict)

    # CST nodes are immutable, however, while traversing the CST, the nodes may
    # get updated by creating new instances. Insertion nodes reference nodes
    # in the CST. This map maps CST nodes to insertions that reference them and
    # can be used to update reference in case of a new CST node instance.
    # For this the `update_position` function can be used
    position_map: _PositionMap = dataclasses.field(default_factory=dict)

    def add_insertion(
        self,
        target: libcst.BaseSuite,
        insertion: libcst.BaseSmallStatement,
        position: _LineStatement,
    ) -> None:
        """
        Add an insertion to the collection

        Args:
            target: the suite in which to insert.
            insertion: the node to insert in target
            position: a node currently in target before which insertion will be
                inserted.

        Note:
            It is not possible to create an insertion that will be added as
            last statement in target. This is currently not a limitation as
            candidates for isolation are always inserted before the current
            statement they are in.
        """
        if target not in self.insertions:
            self.insertions[target] = []
        if position not in self.position_map:
            self.position_map[position] = []

        insertion_data = _NodeInsertion(target, insertion, position)
        self.insertions[target].append(insertion_data)
        self.position_map[position].append(insertion_data)

    def clean_target_insertions(self, target: libcst.CSTNode) -> None:
        """
        Remove a target from the collection. This will remove the insertions
        associated with target and any other related data.
        """
        if target in self.insertions:
            for insertion in self.insertions[target]:
                position_list = self.position_map[insertion.position]
                position_list.remove(insertion)
                if len(position_list) == 0:
                    del self.position_map[insertion.position]

            del self.insertions[target]

    def __getitem__(self, key: libcst.CSTNode) -> list[_NodeInsertion]:
        return self.insertions.get(key, [])

    def update_position(self, old_node: _LineStatement, new_node: _LineStatement) -> None:
        """
        Update references in the collections to old_node to new_node. This is
        used when old_node is replaced in the CST by new_node. When this update
        is omitted, the insertion points of insertions may no longer exist.
        """
        if old_node not in self.position_map:
            return
        for insertion in self.position_map[old_node]:
            insertion.position = new_node
        self.position_map[new_node] = self.position_map[old_node]
        del self.position_map[old_node]


class IsolateReplacementCandidates(libcst.CSTTransformer):
    """
    Replacement candidates may occur in compound statements or expressions. To
    ease further analysis and code rewrites, each candidate is isolated into a
    separate line on which the result is stored in a temporary variable. The
    expression in the compound statement or expression is replaced by the
    temporary variable

    For example:

        #       [------------------] expression 2
        #                     [---] expression 1
        result = torch.sigmoid(x*y) + torch.sigmoid(y)
        #                            [----------------] expression 3

    becomes

        _tmp_1 = x * y
        _tmp_2 = torch.sigmoid(_tmp_1)
        _tmp_3 = torch.sigmoid(y)
        result = _tmp_2 + _tmp_3
    """

    # _visit_stack is a stack of nodes that are currently visited, a node is
    # added to the stack after visit and popped from the stack before leave.
    # This means that during visit and leave calls, the parent of the current
    # node (i.e., the node for which the visit and leave methods are run) is as
    # the top of the stack.
    _visit_stack: list[libcst.CSTNode]

    # A collection of insertions. An insertion is a node that should be
    # inserted in a suite at a specific position. Since there may be many
    # insertions for a single suite, we collect all in a collection and
    # perform all insertions once.
    _insertions: _Insertions

    def __init__(self) -> None:
        self._visit_stack = []
        self._insertions = _Insertions()
        self._count = 1

    def parent(self) -> libcst.CSTNode | None:
        """
        Parent of the deepest node currently being explored. This is the node
        at the top of the visit stack.
        """
        if self._visit_stack:
            return self._visit_stack[-1]
        return None

    def on_visit(self, node: libcst.CSTNode) -> bool:
        visit_children = super().on_visit(node)
        self._visit_stack.append(node)
        return visit_children

    def on_leave(
        self, original_node: libcst.CSTNodeT, updated_node: libcst.CSTNodeT
    ) -> Union[libcst.CSTNodeT, libcst.RemovalSentinel, libcst.FlattenSentinel[libcst.CSTNodeT]]:
        self._visit_stack.pop()
        result_node = super().on_leave(original_node, updated_node)

        # Don't allow for Removal- or FlattenSentinel as these could remove
        # insertion points.
        if isinstance(result_node, (libcst.RemovalSentinel, libcst.FlattenSentinel)):
            raise RuntimeError(
                f"RemovalSentinel and FlattenSentinel are not allowed in conjunction "
                f"with {type(self).__name__}"
            )

        # Cleanup any insertions for orignal/updated node
        self._insertions.clean_target_insertions(original_node)
        if isinstance(original_node, _LineStatement) and isinstance(result_node, _LineStatement):
            self._insertions.update_position(original_node, result_node)

        return cast(libcst.CSTNodeT, result_node)

    def _resolve_insertions(
        self,
        original_node: _SuiteT,
        suite: _SuiteT,
    ) -> _SuiteT:
        """
        Resolve all insertion for original_node. Suite is the updated node in
        which the insertions are added
        """
        insertions = self._insertions[original_node]
        if len(insertions) == 0:
            return suite

        body = list(suite.body)
        for insertion in insertions:
            position = insertion.position

            # Lookup the location of the element that we want to insert before.
            # This should always succeed since body must contain
            # insertion.position since we do not allow for deletions and any
            # change to the node was also captured in the insertion.
            idx = body.index(cast(_BaseStatements, position))

            if isinstance(position, HasLeadingLines):
                leading_lines = position.leading_lines
                body[idx] = body[idx].with_changes(leading_lines=())
                self._insertions.update_position(position, body[idx])
            else:
                leading_lines = ()
            body.insert(
                idx,
                libcst.SimpleStatementLine([insertion.insertion], leading_lines=leading_lines),
            )

        return suite.with_changes(body=tuple(body))

    def leave_Assign(
        self, original_node: libcst.Assign, updated_node: libcst.Assign
    ) -> libcst.BaseSmallStatement:
        if isinstance(updated_node.value, ReplacementCandidate):
            return CandidateAssignment(updated_node)
        return updated_node

    def leave_Yield(
        self, original_node: libcst.Yield, updated_node: libcst.Yield
    ) -> libcst.BaseExpression:
        if isinstance(updated_node.value, ReplacementCandidate):
            return CandidateYield(updated_node)
        return updated_node

    def leave_Return(
        self, original_node: libcst.Return, updated_node: libcst.Return
    ) -> libcst.BaseSmallStatement:
        if isinstance(updated_node.value, ReplacementCandidate):
            return CandidateReturn(updated_node)
        return updated_node

    def leave_ReplacementCandidate(
        self, original_node: ReplacementCandidate, updated_node: ReplacementCandidate
    ) -> ReplacementCandidate | libcst.Name:
        del original_node

        parent = self.parent()
        if parent is None:
            # If there are no parents in the tree, we cannot move up the
            # replacement candidate
            return updated_node

        if m.matches(parent, m.Assign() | m.Return() | m.Expr()):
            # If the replacement candidate is already in an assign node or
            # IndentedBlock we don't have to move it
            return updated_node

        assign_name = libcst.Name(f"_tmp_{self._count}")
        self._count += 1
        assign_target = libcst.AssignTarget(assign_name)
        assignment = CandidateAssignment(libcst.Assign([assign_target], updated_node))

        # Walk up the visitor stack. When we find a SimpleStatementLine,
        # we can insert our new statement just before that in its parent.
        enumerator = range(len(self._visit_stack) - 1, -1, -1)
        for i, node in zip(enumerator, reversed(self._visit_stack)):
            if m.matches(node, m.SimpleStatementLine()):
                if i == 0:
                    # In this case, the simple statement we found is the root of the
                    # tree. In this case, do not create an insertion and fall into
                    # the else clause of this for loop (as is the last element
                    # in the visit stack)
                    continue
                insert_target = self._visit_stack[i - 1]
                assert isinstance(insert_target, libcst.BaseSuite)
                insert_location = cast(libcst.SimpleStatementLine, node)
                break
        else:
            # We were unable to find an higher node to re-insert candidate,
            # try to continue without moving it
            return updated_node

        self._insertions.add_insertion(insert_target, assignment, insert_location)

        return assign_name

    def leave_IndentedBlock(
        self, original_node: libcst.IndentedBlock, updated_node: libcst.IndentedBlock
    ) -> libcst.BaseSuite:
        return self._resolve_insertions(original_node, updated_node)

    def leave_Module(
        self, original_node: libcst.Module, updated_node: libcst.Module
    ) -> libcst.Module:
        return self._resolve_insertions(original_node, updated_node)


class CandidateRewriter(libcst.CSTTransformer):
    """
    Rewrite function calls or operators for alternatives in optable.
    This is used to, for example, rewrite `torch.sigmoid` to an alternative
    activation function.

    Specifically, this pass will replace CandidateXXX nodes and
    ReplacementCandidate node children with annotated CST nodes. I.e., these
    are 'normal' CST nodes with extra metadata for further analysis.

    Args:
        optable: Operator table that contains all operations to rewrite.
    """

    def __init__(self, optable: OperatorTable) -> None:
        self.optable = optable

    def leave_CandidateAssignment(
        self, original_node: CandidateAssignment, updated_node: CandidateAssignment
    ) -> libcst.Assign:
        assign = updated_node.original

        value: libcst.BaseExpression
        match assign.value:
            case ReplacementCandidate(libcst.Call()):
                value = self._rewrite_call(cast(libcst.Call, assign.value.original))
            case ReplacementCandidate(libcst.BinaryOperation()):
                value = assign.value.original
            case ReplacementCandidate(node):
                raise ValueError(
                    f"Found {type(node)} but is not supported. However, it was identified as a "
                    "candidate for autoquant replacement. Please file an issue if you see "
                    "this exception."
                )
            case node:
                raise ValueError(f"Found {type(node)} but is not supported")

        return updated_node.original.with_changes(value=value)

    def leave_CandidateReturn(
        self, original_node: CandidateReturn, updated_node: CandidateReturn
    ) -> libcst.Return | libcst.FlattenSentinel[libcst.CSTNode]:
        return_node = updated_node.original

        value: libcst.BaseExpression
        match return_node.value:
            case None:
                return return_node
            case ReplacementCandidate(libcst.Call()):
                value = self._rewrite_call(cast(libcst.Call, return_node.value.original))
            case ReplacementCandidate(libcst.BinaryOperation()):
                value = return_node.value.original
            case ReplacementCandidate(node):
                raise ValueError(
                    f"Found {type(node)} but is not supported. However, it was identified as a "
                    "candidate for autoquant replacement. Please file an issue if you see "
                    "this exception."
                )
            case node:
                raise ValueError(f"Found {type(node)} but is not supported")

        return return_node.with_changes(value=value)

    def leave_CandidateYield(
        self, original_node: CandidateYield, updated_node: CandidateYield
    ) -> libcst.Yield:
        # TODO: implement yield support
        raise NotImplementedError("Yield support is pending")

    def _rewrite_call(self, call: libcst.Call) -> AnnotatedCall:
        """
        Given existing Call node, create a new Call node that replaces the
        function call with the target associated to the current target in the
        operator table.
        """

        func_name = libcst.helpers.get_full_name_for_node(call)

        if func_name is None:
            raise ff.exceptions.AutoquantError(f"Cannot resolve {call} to function")

        quant_op = self.optable.get(func_name)
        replace_name = quant_op.dispatch_qualified_name()
        if replace_name is None:
            raise ff.exceptions.AutoquantError(f"Cannot resolve qualified name for {quant_op} ")

        func_cst = libcst.parse_statement(replace_name)
        if len(func_refs := m.findall(func_cst, m.Expr(m.Attribute() | m.Name()))) == 1:
            func_ref = cast(libcst.Expr, func_refs[0]).value
            call = call.with_changes(func=func_ref)
            return AnnotatedCall(original=call, operator=quant_op)
        raise ff.exceptions.AutoquantError(f"{replace_name} is not a valid function name")
