# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import dataclasses
import inspect
import re
import types

from collections.abc import Sequence
from typing import Any, Callable, Protocol, TypeAlias, TypeVar, cast, runtime_checkable

import libcst
import libcst.helpers
import libcst.matchers as m

from libcst.metadata.scope_provider import Scope
from typing_extensions import override

from fastforward._autoquant.cst.node_creation import (
    get_keyword_argument_node,
    get_quantized_function_counterpart,
)
from fastforward._autoquant.function_context import FunctionContext
from fastforward._autoquant.mypy.type_provider import MypyTypeProvider, TypeInfo
from fastforward._autoquant.pybuilder import QuantizerReferenceCollection
from fastforward._autoquant.pysource.scope import ScopeProvider
from fastforward._quantops import OperatorTable
from fastforward._quantops.optable import (
    OPS_LIBCST_TO_TORCH_MAPPING,
)

from .nodes import (
    AbstractClassReference,
    GeneralAssignment,
    QuantizedCall,
    ReplacementCandidate,
    UnresolvedQuantizedCall,
    is_simple_literal,
    node_asdict,
)


@runtime_checkable
class _HasLeadingLines(Protocol):
    leading_lines: Sequence[libcst.EmptyLine]


_LineStatement: TypeAlias = libcst.BaseStatement
_BaseStatements: TypeAlias = libcst.SimpleStatementLine | libcst.BaseCompoundStatement
_SuiteT = TypeVar("_SuiteT", libcst.IndentedBlock, libcst.Module)


class ConvertSemicolonJoinedStatements(libcst.CSTTransformer):
    """Convert semicolon-separated statement into newline-separated statements.

    This includes:
        - Convert SimpleStatementSuite to IndentedBlock.
        - Flatten SimpleStatementLine with a body of more than one element.

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

    @override
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

    @override
    def leave_SimpleStatementLine(
        self, original_node: libcst.SimpleStatementLine, updated_node: libcst.SimpleStatementLine
    ) -> libcst.BaseStatement | libcst.FlattenSentinel[libcst.BaseStatement]:
        if len(updated_node.body) == 1:
            return updated_node
        else:

            def _as_statement_line(node: libcst.BaseSmallStatement) -> libcst.SimpleStatementLine:
                node = node.with_changes(semicolon=libcst.MaybeSentinel.DEFAULT)
                return libcst.SimpleStatementLine(body=(node,))

            nodes = [_as_statement_line(statement) for statement in updated_node.body]
            return libcst.FlattenSentinel(nodes)


class WrapAssignments(libcst.CSTTransformer):
    """Wraps all types of assignments into a wrapped assignment.

    That is, assignments of the sort
    - x = 0 (libcst.Assign)
    - x: int = 0 (libcst.AnnAssign)
    - x : int (libcst.AnnAssign)
    - x = y = 0 (libcst.Assign)
    - x += 1 (libcst.AugAssign)

    We do not wrap the walrus operator, and the assignments resulting from various forms of
    import statements.
    """

    @override
    def leave_AnnAssign(
        self, original_node: libcst.AnnAssign, updated_node: libcst.AnnAssign
    ) -> GeneralAssignment:
        return GeneralAssignment(
            original=original_node,
            targets=(updated_node.target,),
            annotation=updated_node.annotation,
            value=updated_node.value,
        )

    @override
    def leave_Assign(
        self, original_node: libcst.Assign, updated_node: libcst.Assign
    ) -> GeneralAssignment:
        return GeneralAssignment(
            original=original_node,
            targets=tuple(x.target for x in updated_node.targets),
            annotation=None,
            value=updated_node.value,
        )

    @override
    def leave_AugAssign(
        self, original_node: libcst.AugAssign, updated_node: libcst.AugAssign
    ) -> GeneralAssignment:
        return GeneralAssignment(
            original=original_node,
            targets=(updated_node.target,),
            annotation=None,
            value=updated_node.value,
        )


class MarkReplacementCandidates(libcst.CSTTransformer):
    """Mark nodes in the CST for replacement by future passes.

    Marking is performed by wrapping the CST node in a ReplacementCandidate
    node.

    If an operation only contains (non collection) literals, do not mark it
    for replacement.
    """

    @override
    def leave_Call(
        self, original_node: libcst.Call, updated_node: libcst.Call
    ) -> libcst.BaseExpression:
        if m.matches(original_node.func, m.Name("super")):
            return updated_node
        return ReplacementCandidate(updated_node)

    @override
    def leave_BinaryOperation(
        self,
        original_node: libcst.BinaryOperation,
        updated_node: libcst.BinaryOperation,
    ) -> libcst.BaseExpression:
        del original_node
        if is_simple_literal(updated_node.left) and is_simple_literal(updated_node.right):
            return updated_node
        else:
            return ReplacementCandidate(updated_node)

    @override
    def leave_UnaryOperation(
        self,
        original_node: libcst.UnaryOperation,
        updated_node: libcst.UnaryOperation,
    ) -> libcst.BaseExpression:
        del original_node
        if is_simple_literal(updated_node.expression):
            return updated_node
        else:
            return ReplacementCandidate(updated_node)

    @override
    def visit_Annotation(self, node: libcst.Annotation) -> bool:
        # Inside an annotation we never want to replace anything
        return False


_ExpressionT = TypeVar("_ExpressionT", bound=libcst.BaseExpression)


class ExtendedMarkReplacementCandidates(MarkReplacementCandidates):
    METADATA_DEPENDENCIES = (MypyTypeProvider,)

    def _determine_replacement_candidate(
        self, type_info: Sequence[TypeInfo | None], updated_node: _ExpressionT
    ) -> ReplacementCandidate | _ExpressionT | None:
        """Determine if a node should be marked as a replacement candidate based on type info.

        This method analyzes the type information of expressions to decide if they should be
        marked for replacement with quantized operations. It specifically looks for tensor types
        that would benefit from quantization.

        Args:
            type_info: A sequence of TypeInfo objects or None values representing the types
                      of expressions involved in the operation.
            updated_node: The CST node being considered for replacement.

        Returns:
            - ReplacementCandidate wrapping the node if any involved type is a torch.Tensor
            - The original node unchanged if no tensor types are detected
            - None if type information is missing, indicating the caller should fall back to
              a non-type-informed implementation
        """
        if all(info is None for info in type_info):
            # If type information is missing, fall back to a non-type informed implementation.
            return None

        if any(_is_subtype(info, "torch.Tensor") for info in type_info):
            return ReplacementCandidate(updated_node)
        else:
            return updated_node

    @override
    def leave_Call(
        self, original_node: libcst.Call, updated_node: libcst.Call
    ) -> libcst.BaseExpression:
        type_info = []
        for arg in original_node.args:
            type_info.append(self.get_metadata(MypyTypeProvider, arg, None))

        result = self._determine_replacement_candidate(type_info, updated_node)
        return result or super().leave_Call(original_node, updated_node)

    @override
    def leave_UnaryOperation(
        self,
        original_node: libcst.UnaryOperation,
        updated_node: libcst.UnaryOperation,
    ) -> libcst.BaseExpression:
        expr_type_info = self.get_metadata(MypyTypeProvider, original_node.expression, None)
        out_type_info = self.get_metadata(MypyTypeProvider, original_node, None)
        type_info = [expr_type_info, out_type_info]

        result = self._determine_replacement_candidate(type_info, updated_node)
        return result or super().leave_UnaryOperation(original_node, updated_node)

    @override
    def leave_BinaryOperation(
        self,
        original_node: libcst.BinaryOperation,
        updated_node: libcst.BinaryOperation,
    ) -> libcst.BaseExpression:
        lhs_type_info = self.get_metadata(MypyTypeProvider, original_node.left, None)
        rhs_type_info = self.get_metadata(MypyTypeProvider, original_node.right, None)
        out_type_info = self.get_metadata(MypyTypeProvider, original_node, None)
        type_info = [lhs_type_info, rhs_type_info, out_type_info]

        result = self._determine_replacement_candidate(type_info, updated_node)
        return result or super().leave_BinaryOperation(original_node, updated_node)


def _is_subtype(type_info: TypeInfo | None, type_name: str) -> bool:
    return False if type_info is None else type_info.is_subtype(type_name)


@dataclasses.dataclass
class _NodeInsertion:
    target: libcst.CSTNode
    insertion: libcst.BaseSmallStatement
    position: _LineStatement


_Statement: TypeAlias = libcst.BaseStatement | libcst.BaseSmallStatement
_PositionMap: TypeAlias = dict[_Statement, list[_NodeInsertion]]


@dataclasses.dataclass
class _Insertions:
    """Insertion collection.

    Collects insertions during traversal of a CST and associates each with an
    insertion point.

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
        target: libcst.BaseSuite | libcst.Module,
        insertion: libcst.BaseSmallStatement,
        position: _LineStatement,
    ) -> None:
        """Add an insertion to the collection.

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
        """Remove a target from the collection.

        This will remove the insertions associated with target and any other
        related data.
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
        """Update references in the collections to old_node to new_node.

        This is used when old_node is replaced in the CST by new_node. When
        this update is omitted, the insertion points of insertions may no
        longer exist.
        """
        if old_node not in self.position_map:
            return
        for insertion in self.position_map[old_node]:
            insertion.position = new_node
        self.position_map[new_node] = self.position_map[old_node]
        del self.position_map[old_node]


class IsolateReplacementCandidates(libcst.CSTTransformer):
    """Replacement candidates may occur in compound statements or expressions.

    To ease further analysis and code rewrites, each candidate is isolated into
    a separate line on which the result is stored in a temporary variable. The
    expression in the compound statement or expression is replaced by the
    temporary variable.

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

    _SKIP_ISOLATION_NODES = (libcst.BaseComp, libcst.IfExp)

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
        super().__init__()
        self._visit_stack = []
        self._insertions = _Insertions()
        self._nested_in_no_isolation_subtree = 0
        self._count = 1

    def parent(self) -> libcst.CSTNode | None:
        """Parent of the deepest node currently being explored.

        This is the node at the top of the visit stack.
        """
        if self._visit_stack:
            return self._visit_stack[-1]
        return None

    @override
    def on_visit(self, node: libcst.CSTNode) -> bool:
        visit_children = super().on_visit(node)
        self._visit_stack.append(node)
        if isinstance(node, self._SKIP_ISOLATION_NODES):
            self._nested_in_no_isolation_subtree += 1
        return visit_children

    @override
    def on_leave(
        self, original_node: libcst.CSTNodeT, updated_node: libcst.CSTNodeT
    ) -> libcst.CSTNodeT | libcst.RemovalSentinel:
        top = self._visit_stack.pop()
        if isinstance(top, self._SKIP_ISOLATION_NODES):
            self._nested_in_no_isolation_subtree -= 1

        result_node = super().on_leave(original_node, updated_node)

        # Don't allow for Removal- or FlattenSentinel as these could remove
        # insertion points.
        if isinstance(result_node, (libcst.RemovalSentinel, libcst.FlattenSentinel)):
            raise RuntimeError(
                "RemovalSentinel and FlattenSentinel are not allowed in conjunction "
                + f"with {type(self).__name__}"
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
        """Resolve all insertion for original_node.

        Suite is the updated node in which the insertions are added.
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

            if isinstance(position, _HasLeadingLines):
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

    def leave_ReplacementCandidate(
        self, original_node: ReplacementCandidate, updated_node: ReplacementCandidate
    ) -> ReplacementCandidate | libcst.Name:
        """Leave function for `ReplacementCandidate`.

        Within comprehensions, we skip this function.
        """
        del original_node

        parent = self.parent()
        if parent is None:
            # If there are no parents in the tree, we cannot move up the
            # replacement candidate
            return updated_node

        match parent:
            case libcst.Assign() | libcst.AugAssign() | libcst.AnnAssign():
                # If the replacement candidate is already in an assign node we
                # don't have to move it
                return updated_node
            case libcst.Return() | libcst.Yield() | libcst.Raise():
                # If the replacement candidate is the expression in a return or yield
                # statement, we don't have to move it.
                return updated_node
            case libcst.Expr():
                # If the replacement candidate is part of an Expression (whose result is
                # unassigned / unused, e.g., in a `print` statement), we skip it.
                return updated_node
            case _:
                pass

        # Skip isolation if inside one or more subtrees that have a root that is marked
        # as "no isolated" (i.e., a member of `_SKIP_ISOLATION_NODES`).
        if self._is_nested_in_no_isolation_subtree():
            return updated_node

        assign_name = libcst.Name(f"_tmp_{self._count}")
        self._count += 1
        assign_target = libcst.AssignTarget(assign_name)
        assignment = libcst.Assign([assign_target], updated_node)

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
                assert isinstance(insert_target, (libcst.BaseSuite, libcst.Module))
                insert_location = cast(libcst.SimpleStatementLine, node)
                break
        else:
            # We were unable to find an higher node to re-insert candidate,
            # try to continue without moving it
            return updated_node

        self._insertions.add_insertion(insert_target, assignment, insert_location)

        return assign_name

    def _is_nested_in_no_isolation_subtree(self) -> bool:
        return self._nested_in_no_isolation_subtree > 0

    @override
    def leave_IndentedBlock(
        self, original_node: libcst.IndentedBlock, updated_node: libcst.IndentedBlock
    ) -> libcst.BaseSuite:
        return self._resolve_insertions(original_node, updated_node)

    @override
    def leave_Module(
        self, original_node: libcst.Module, updated_node: libcst.Module
    ) -> libcst.Module:
        return self._resolve_insertions(original_node, updated_node)


class QuantizedCounterpartReplacer(libcst.CSTTransformer):
    """Replaces function calls with their quantized counterparts in the CST.

    This transformer identifies function calls, unary operations, and binary operations
    that have quantized equivalents and replaces them with their quantized counterparts.
    The exact replacements are determined by the provided operator table (optable).

    For example:
    - torch.sigmoid(*args, **kwargs) becomes
      ff.nn.functional.sigmoid(*args, **kwargs, output_quantizer=self.quantizer_sigmoid_1)
    - Binary operations like a + b become
      ff.nn.functional.add(a, b, output_quantizer=self.quantizer_add_1)
    - Unary operations like -x become
      ff.nn.functional.neg(x, output_quantizer=self.quantizer_neg_1)

    The transformer handles both resolved calls (where a quantized counterpart exists
    in the operator table) and unresolved calls (which are wrapped with relevant metadata
    for later processing).

    Args:
        optable: The operator table containing mappings from original functions
            to their quantized counterparts.
        func_ctx: Function context providing information about the current function
            being transformed.
    """

    def __init__(
        self,
        optable: OperatorTable,
        func_ctx: FunctionContext,
        quantizer_refs: QuantizerReferenceCollection,
    ) -> None:
        super().__init__()
        self._optable = optable
        self._func_ctx = func_ctx
        self._quantizer_refs = quantizer_refs

    def leave_ReplacementCandidate(
        self,
        original_node: ReplacementCandidate,
        updated_node: ReplacementCandidate,
    ) -> libcst.BaseExpression:
        """Leave function for `ReplacementCandidate`."""
        del original_node

        original_args: Sequence[libcst.Arg]
        match orig := updated_node.original:
            case libcst.UnaryOperation():
                original_func, _, original_func_name = _get_name_and_torch_name_for_operation(orig)
                original_args = (libcst.Arg(orig.expression),)
            case libcst.BinaryOperation():
                original_func, _, original_func_name = _get_name_and_torch_name_for_operation(orig)
                original_args = (libcst.Arg(orig.left), libcst.Arg(orig.right))
            case libcst.Call():
                original_func_name = libcst.helpers.get_full_name_for_node(orig)
                original_args = orig.args
                original_func = _resolve_reference(original_func_name, self._func_ctx)
            case _:
                return updated_node

        if original_func_name is None:
            # We cannot determine the name of the function called, skip it.
            return updated_node

        return self._create_quantized_call(
            node=updated_node,
            fn_name=original_func_name,
            func_ref=original_func,
            orig_args=original_args,
        )

    def _create_quantized_call(
        self,
        node: ReplacementCandidate,
        fn_name: str,
        func_ref: Any | None,
        orig_args: Sequence[libcst.Arg],
    ) -> libcst.BaseExpression:
        call_node = self._create_resolved_quantized_call(fn_name, func_ref or fn_name, orig_args)
        if call_node is not None:
            return call_node
        if func_ref is None:
            return node
        return self._create_unresolved_quantized_call(node, fn_name, func_ref, orig_args)

    def _create_resolved_quantized_call(
        self,
        fn_name: str,
        func_key: Callable[..., Any] | str,
        original_args: Sequence[libcst.Arg],
    ) -> libcst.BaseExpression | None:
        """Create a resolved quantized call if a quantized counterpart exists.

        This method attempts to create a quantized version of a function call by looking up
        the function name in the operator table. If a quantized counterpart is found, it
        creates a new call with the original arguments plus additional output quantizer
        arguments.
        """
        try:
            func, operator = get_quantized_function_counterpart(
                optable=self._optable,
                func_key=func_key,
                args=original_args,
            )
        except KeyError:
            # We don't have a quantized replacement for this function, skip it.
            return None

        extra_args: list[libcst.Arg] = []

        for intermediate_quantizer in operator.intermediate_quantizers:
            quantizer_var = self._quantizer_refs.create_quantizer_expression(intermediate_quantizer)
            extra_args.append(get_keyword_argument_node(intermediate_quantizer, quantizer_var))

        if isinstance(func_key, str):
            output_quantizer_name = func_key.split(".")[-1]
        else:
            output_quantizer_name = func_key.__name__

        for i in range(operator.num_output_quantizers):
            quantizer_var = self._quantizer_refs.create_quantizer_expression(output_quantizer_name)

            arg_name = "output_quantizer" + (f"_{i}" if operator.num_output_quantizers > 1 else "")
            extra_args.append(get_keyword_argument_node(arg_name, quantizer_var))

        return QuantizedCall(
            func=func,
            args=(*original_args, *extra_args),
            original_name=fn_name,
            operator=operator,
        )

    def _create_unresolved_quantized_call(
        self,
        node: ReplacementCandidate,
        func_name: str,
        func_ref: Callable[..., Any],
        original_args: Sequence[libcst.Arg],
    ) -> libcst.BaseExpression:
        """Create an unresolved quantized call for functions without quantized counterparts.

        This method handles cases where a function call cannot be directly replaced with
        a quantized counterpart (i.e., when _create_resolved_quantized_call returns None).
        It wraps the original call in an UnresolvedQuantizedCall node that preserves
        the original call structure along with metadata for potential later processing.
        """
        if not isinstance(node.original, libcst.Call):
            # This happens when an operator isn't supported by `_create_resolved_quantized_call`.
            # We leave it unchanged but keep it wrapped in ReplacementCandidate so later passes
            # can identify it.
            return node

        call_params = node_asdict(node.original)

        call_params["args"] = original_args
        call_params["func"] = _generalize_class_refs(node.original.func, self._func_ctx)
        return UnresolvedQuantizedCall(**call_params, original_name=func_name, func_ref=func_ref)


def _get_root(expr: libcst.BaseExpression) -> libcst.BaseExpression:
    """Get the 'root' expression of a (possibly nested) `Attribute` node."""
    match expr:
        case libcst.Attribute():
            return _get_root(expr.value)
        case _:
            return expr


def _generalize_class_refs(
    expr: libcst.BaseExpression, func_ctx: FunctionContext
) -> libcst.BaseExpression:
    """Replace torch module class name references with `AbstractClassReference` nodes.

    Args:
        expr: The CST expression node to transform.
        func_ctx: Function context containing the torch module information.
    """
    if (torch_module := func_ctx.torch_module) is None:
        return expr

    func_root = _get_root(expr)
    if isinstance(func_root, libcst.Name) and func_root.value == torch_module.__name__:
        return expr.deep_replace(func_root, AbstractClassReference(**node_asdict(func_root)))
    else:
        return expr


class FoldSimpleTemporaries(libcst.CSTTransformer):
    r"""Transformer that folds simple temporary variable assignments.

    This transformer identifies temporary variables (with names matching the pattern 'tmp_\d+')
    that are assigned once and used once. It then replaces the usage of these variables
    with their assigned value and removes the original assignment statement.

    For example, code like:
        tmp_1 = x + y
        result = tmp_1 * z

    Would be transformed to:
        result = (x + y) * z

    This helps simplify code that has been previously transformed by passes like
    IsolateReplacementCandidates, which may have introduced many temporary variables.

    The transformer only folds temporaries that:
    1. Match the naming pattern 'tmp_\d+'
    2. Are assigned exactly once
    3. Are accessed exactly once as a simple Name node
    """

    METADATA_DEPENDENCIES = (ScopeProvider,)

    def __init__(self) -> None:
        super().__init__()
        self._tmp_vars_stack: list[dict[str, GeneralAssignment]] = []

    @property
    def _tmp_vars(self) -> dict[str, GeneralAssignment]:
        return self._tmp_vars_stack[-1]

    def _is_tmp_var(self, name: str) -> bool:
        return re.fullmatch(r"tmp_\d+", name) is not None

    def _target_name(self, node: libcst.BaseAssignTargetExpression) -> str | None:
        match node:
            case libcst.Name(name):
                return name
            case _:
                return None

    def _remove_temporaries(
        self,
        original_node: libcst.FunctionDef,
        updated_node: libcst.FunctionDef,
    ) -> libcst.FunctionDef:
        if (scope := self.get_metadata(ScopeProvider, original_node.body)) is None:
            return original_node
        assert isinstance(scope, Scope)

        removals: set[libcst.CSTNode] = set()
        replacements: dict[libcst.CSTNode, libcst.BaseExpression] = {}

        for tmp_var, assign_node in self._tmp_vars.items():
            if assign_node.value is None:
                continue

            # We can assume there is a single target due to `visit_GeneralAssignment`
            assert len(assign_node.targets) == 1
            target = assign_node.targets[0]
            assignments = scope[tmp_var]
            # Remove assignment from accesses
            accesses = [access for access in scope.accesses[tmp_var] if access.node is not target]
            if len(assignments) != 1 or len(accesses) != 1:
                # We don't remove variables that are assigned more than once or
                # accessed more than once.
                continue

            access_node = accesses[0].node
            if not isinstance(access_node, libcst.Name):
                # Skip over everything that is not a `Name`
                continue

            removals.add(assign_node)
            replacements[access_node] = assign_node.value

        result = updated_node.visit(_NodeReplacer(removals, replacements))
        assert isinstance(result, libcst.FunctionDef)
        return result

    def visit_GeneralAssignment(self, node: GeneralAssignment) -> bool:
        # Do not remove possible aliases
        if len(node.targets) != 1:
            return True

        target = node.targets[0]

        # If target is not a `Name`, no tracking is required
        if (target_name := self._target_name(target)) is None:
            return True

        if self._is_tmp_var(target_name):
            self._tmp_vars[target_name] = node

        return False

    def visit_FunctionDef(self, node: libcst.FunctionDef) -> bool:
        del node
        self._tmp_vars_stack.append({})
        return True

    def leave_FunctionDef(
        self, original_node: libcst.FunctionDef, updated_node: libcst.FunctionDef
    ) -> libcst.FunctionDef:
        result = self._remove_temporaries(original_node, updated_node)
        self._tmp_vars_stack.pop()
        return result


class _NodeReplacer(libcst.CSTTransformer):
    """A CST transformer that replaces or removes nodes in the syntax tree.

    This transformer performs two operations:
    1. Replaces specific nodes with new expressions (adding parentheses around them)
    2. Removes specific nodes from the tree

    It's used by FoldSimpleTemporaries to replace temporary variable references with
    their values and remove the original assignment statements.

    Args:
        removals: A set of CST nodes that should be removed from the tree
        replacements: A dictionary mapping CST nodes to their replacement expressions
    """

    def __init__(
        self,
        removals: set[libcst.CSTNode],
        replacements: dict[libcst.CSTNode, libcst.BaseExpression],
    ) -> None:
        super().__init__()
        self.removals = removals
        self.replacements = replacements

    def on_leave(
        self, original_node: libcst.CSTNodeT, updated_node: libcst.CSTNodeT
    ) -> libcst.CSTNodeT | libcst.RemovalSentinel | libcst.FlattenSentinel[libcst.CSTNodeT]:
        # Replace nodes
        if original_node in self.replacements:
            expr = self.replacements[original_node]
            return expr.with_changes(  # type: ignore[return-value]
                lpar=(libcst.LeftParen(),) + tuple(expr.lpar),
                rpar=tuple(expr.rpar) + (libcst.RightParen(),),
            )

        if original_node in self.removals:
            return libcst.RemoveFromParent()

        if isinstance(original_node, libcst.SimpleStatementLine):
            if len(original_node.body) == 1 and original_node.body[0] in self.removals:
                return libcst.RemoveFromParent()

        return super().on_leave(original_node, updated_node)


def _get_name_and_torch_name_for_operation(
    node: libcst.UnaryOperation | libcst.BinaryOperation,
) -> tuple[Callable[..., Any], str, str] | tuple[None, None, None]:
    """Maps a BinaryOperation node to its corresponding common name and torch name.

    Examples:
        - libcst.Add() -> "add", "torch.add"
        - libcst.Expr() -> None, None
    """
    op = OPS_LIBCST_TO_TORCH_MAPPING.get(type(node.operator))
    if op is None:
        return None, None, None

    return op, op.__name__, f"{op.__module__}.{op.__name__}"


def _resolve_reference(reference: str | None, func_ctx: FunctionContext) -> Any | None:
    """Resolve reference within function context.

    Resolves a string reference to an actual object by looking up the reference
    in the closure variables of the function context.

    Args:
        reference: A string representing a variable or attribute path (e.g. "torch.nn.Linear")
        func_ctx: The function context containing information about the wrapping function
                 and its environment

    Returns:
        The resolved object if found, None otherwise
    """
    if reference is None:
        return None
    if (wrapping_func := func_ctx.func) is None:
        return None

    closure_vars = inspect.getclosurevars(inspect.unwrap(wrapping_func))
    scope_vars = {**closure_vars.builtins, **closure_vars.globals, **closure_vars.nonlocals}

    if func_ctx.torch_module:
        if func_ctx.instance_var:
            scope_vars[func_ctx.instance_var] = func_ctx.torch_module
        if func_ctx.class_var:
            scope_vars[func_ctx.class_var] = func_ctx.torch_module

    root, *parts = reference.split(".")

    try:
        if (obj := scope_vars.get(root, None)) is None:
            return None
        for part in parts:
            obj = getattr(obj, part)
        if isinstance(obj, types.MethodType):
            obj = obj.__func__
        return obj
    except AttributeError:
        return None
