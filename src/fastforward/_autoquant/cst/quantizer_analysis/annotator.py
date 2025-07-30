# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Module for annotating CST nodes with quantization metadata.

This module provides functionality for analyzing the control flow and data flow
of a given piece of code, and annotating the corresponding CST nodes with
quantization metadata. The annotations are used to determine which nodes
require quantization, and to track the usage of quantized variables.

The module includes classes for representing scopes, assignments, and
quantization statuses, as well as functions for merging scopes, resolving scope
breaks, and iterating over quantized arguments.

The main class in this module is `_QuantizationAnnotator`, which is responsible
for traversing the CST and adding quantization annotations to nodes that
require quantization. The annotator uses a scope-based approach to keep track
of the current context and make decisions about which nodes to annotate.

The module also includes a `QuantizationAnnotationProvider` class, which is a
metadata provider that stores the quantization annotations for a given node and
can be used by any libcst visitor or transformer.
"""

import contextlib
import dataclasses

from typing import Iterator

import libcst

from libcst.metadata.expression_context_provider import (
    ExpressionContext,
    ExpressionContextProvider,
    ExpressionContextVisitor,
)
from typing_extensions import overload

from fastforward._autoquant.cst import node_processing, nodes

from .scope import QuantizationMetadata
from .scope import QuantizationScope as Scope


class _ExpressionContextVisitor(ExpressionContextVisitor):
    """Extends the ExpressionContextVisitor to include support for `GeneralAssignment`."""

    def visit_GeneralAssignment(self, node: nodes.GeneralAssignment) -> bool:
        for target in node.targets:
            target.visit(_ExpressionContextVisitor(self.provider, ExpressionContext.STORE))
        if node.value:
            node.value.visit(self)
        return False


class _ExpressionContextProvider(ExpressionContextProvider):
    """Subclass of `ExpressionContextProvider` that uses `_ExpressionContextVisitor`."""

    def visit_Module(self, node: libcst.Module) -> None:
        node.visit(_ExpressionContextVisitor(self, ExpressionContext.LOAD))


@dataclasses.dataclass(frozen=True)
class QuantizationAnnotation:
    """Annotation for CSTNodes that stores quantization metadata.

    Attributes:
        target: The target variable name that is being quantized.
        uses: The nodes that use the target.
    """

    target: str
    uses: list[libcst.CSTNode] = dataclasses.field(compare=False, hash=False, default_factory=list)


class QuantizationAnnotationProvider(libcst.BatchableMetadataProvider[set[QuantizationAnnotation]]):
    """Metadata provider for quantization annotations."""

    METADATA_DEPENDENCIES = (_ExpressionContextProvider,)

    def visit_Module(self, node: libcst.Module) -> None:
        annotator = _QuantizationAnnotator(self)
        node.visit(annotator)
        return None


class _QuantizationAnnotator(libcst.CSTVisitor):
    """A visitor that annotates CSTNodes with quantization metadata.

    This visitor is responsible for traversing the CST and adding quantization
    annotations to nodes that require quantization. It uses a scope-based
    approach to keep track of the current context and make decisions about
    which nodes to annotate.

    Assumptions and choices on the annotation and location of annotation is
    documented in the `evaluate_*` methods for specific nodes.

    This visitor utilizes `evaluate_Node` methods instead of the traditional
    `visit_Node` approach. The `evaluate_Node` methods abstract over the
    pattern of returning `False` from `visit_Node` and explicitly handling
    sub-node visits in the corresponding `leave_Node` method. By using
    `evaluate_Node`, we simplify the implementation and focus on the node
    evaluation logic, while the visitor takes care of the underlying traversal
    mechanics. This approach ensures that nodes are visited in lexical order by
    default, unless an explicit evaluation order is required, in which case an
    `evaluate_Node` method must be implemented to override the default
    behavior.

    Args:
        provider: The metadata provider that is used as the key to store the
            necessary metadata for quantization annotation.
    """

    def __init__(self, provider: QuantizationAnnotationProvider) -> None:
        self._active_scope: "Scope" = Scope()
        self._provider = provider

    @contextlib.contextmanager
    def enter_scope(self, *, is_loop: bool = False) -> Iterator["Scope"]:
        """Enters a new scope, allowing for nested scope management.

        This context manager creates a new scope, sets it as the active scope, and yields it.
        When the context is exited, the active scope is reset to the parent scope.

        Args:
            is_loop: Whether the new scope is a looping branch.

        Yields:
            The new scope.
        """
        new_scope = Scope(
            parent=self._active_scope,
            is_looping_branch=is_loop,
            repeated_evaluation=self._active_scope.repeated_evaluation,
        )
        self._active_scope = new_scope
        try:
            yield self._active_scope
        finally:
            if self._active_scope is not new_scope:
                msg = (
                    "The active scope was changed within a `enter_scope` block. "
                    + "Cannot guarantee correct scope analysis anymore."
                )
                raise RuntimeError(msg)
            if self._active_scope.parent is None:
                msg = "Trying to exit the main scope, this is not allowed."
                raise RuntimeError(msg)
            self._active_scope = self._active_scope.parent

    @contextlib.contextmanager
    def reevaluate_within_active_scope(self) -> Iterator[None]:
        """Temporarily enables repeated evaluation within the active scope.

        Repeated evaluation is used to perform correct scope analysis for
        for-loops. In for-loops, a variable defined in the loop's body is
        'reachable' by all statements in the loop body. Hence, we re-evaluate
        the loop twice to 'simulate' this behavior and ensure accurate scope
        analysis.
        """
        current_repeated_evaluation = self._active_scope.repeated_evaluation
        active_scope = self._active_scope
        active_scope.repeated_evaluation = True
        try:
            yield
        finally:
            active_scope.repeated_evaluation = current_repeated_evaluation

    def on_visit(self, node: libcst.CSTNode) -> bool:
        # Call `evulate_{Node}` if an implementation exists, otherwise default to `visit_{Node}`.
        eval_func = getattr(self, f"evaluate_{type(node).__name__}", None)
        if eval_func:
            eval_func(node)
            return False
        else:
            return super().on_visit(node)

    def evaluate_FunctionDef(self, node: libcst.FunctionDef) -> None:
        """Create new scope and evaluate params and body."""
        with self.enter_scope():
            node.params.visit(self)
            node.body.visit(self)

    def evaluate_IndentedBlock(self, node: libcst.IndentedBlock) -> None:
        """Evaluate `IndentedBlock`.

        Evaluate each statement in the body of `IndentedBlock` unless a break,
        continue, or return statement is encountered, which terminates the
        evaluation of the current block.

        In the case of a break and return statement, the active scope is marked
        as terminated. This allows further handling of the scope in a manner
        that follows Python's semantics.
        """
        for subnode in node.body:
            match subnode:
                case libcst.SimpleStatementLine([libcst.Return(value)]):
                    if value is not None:
                        value.visit(self)
                    self._active_scope.terminate(reason="return")
                    return
                case libcst.SimpleStatementLine([libcst.Continue()]):
                    return
                case libcst.SimpleStatementLine([libcst.Break()]):
                    self._active_scope.terminate(reason="break")
                    return
                case libcst.SimpleStatementLine([libcst.Raise(exc, cause)]):
                    if exc is not None:
                        exc.visit(self)
                    if cause is not None:
                        cause.visit(self)
                    self._active_scope.terminate(reason="raise")
                    return
                case libcst.SimpleStatementLine(body):
                    for statement in body:
                        statement.visit(self)
                case _:
                    subnode.visit(self)

    def evaluate_Param(self, node: libcst.Param) -> None:
        """Record an assignment to params' name in current scope."""
        self.record_assignment(node.name, node, is_quantized=False)

    def evaluate_If(self, node: libcst.If) -> None:
        """Evaluate If node."""
        # Evaluate test node in current scope
        node.test.visit(self)

        # Evaluate true branch in new scope
        with self.enter_scope() as true_scope:
            node.body.visit(self)
        if_scope = true_scope

        # If a orelse branch exists, evaluate it in its own scope and merge
        # with true scope.
        if node.orelse:
            with self.enter_scope() as false_scope:
                node.orelse.visit(self)

            if_scope = true_scope.merge(false_scope)

        # If the if node is exhaustive (i.e., there is an else branch), then
        # overwrite the current scope with the merged scopes of the branches.
        # Otherwise, merge with the active scope.
        if _if_node_is_exhaustive(node):
            self._active_scope.overwrite(if_scope)
        else:
            self._active_scope.merge(if_scope, inplace=True)

    def evaluate_GeneralAssignment(self, node: nodes.GeneralAssignment) -> None:
        """Evaluate assignments.

        Record all (supported) assignments in the current scope. See the docstring
        of `record_assignment` for more info on supported assignments.
        """
        assignments = list(node_processing.normalize_assignments(node))

        for assign in assignments:
            # First evaluate the value expressions before any assignment
            assign.value.visit(self)

        for assign in assignments:
            self._process_assignment(assign, node)

        for target in node.targets:
            target.visit(self)

    def _process_assignment(
        self, assign: node_processing.NormalizedAssignment, producer: libcst.CSTNode
    ) -> None:
        """Process a normalized assignment.

        This method checks if the assignment value is a quantized call. If it is,
        the method records the assignment with the quantized status. If not, it
        records the assignment as non-quantized.

        Args:
            assign: The normalized assignment.
            producer: The node that produced the assignment.

        Raises:
            NotImplementedError: If the quantized operator has multiple return values.
        """
        if not isinstance(assign.value, nodes.QuantizedCall):
            for target in assign.targets:
                self.record_assignment(target, producer, is_quantized=False)
        else:
            operator = assign.value.operator
            match assign.targets:
                case [target]:
                    self.record_assignment(
                        target, producer, is_quantized=operator.returns_quantized
                    )
                case [*_]:  # more than one target
                    msg = "Quantized operators and multiple return values are not yet supported"
                    raise NotImplementedError(msg)

    def record_assignment(
        self, target: libcst.CSTNode, producer: libcst.CSTNode, is_quantized: bool
    ) -> None:
        """Records an assignment in the current scope.

        It currently only supports assignments to names (i.e., variables), and does not
        handle assignments to other types of targets (e.g., attributes, subscripts).

        Args:
            target: The target of the assignment (e.g., a variable name).
            producer: The node that produced the assignment.
            is_quantized: Whether the assignment is quantized.

        Note:
            This method assumes that the target is a name (i.e., a variable). If the
            target is not a name, this method will not record the assignment.
        """
        match target:
            case libcst.Name(value=var):
                self._active_scope.record_assignment(
                    name=var, producer=producer, metadata=QuantizationMetadata(is_quantized)
                )
            case _:
                pass

    def evaluate_QuantizedCall(self, node: nodes.QuantizedCall) -> None:
        """Evaluate a `QuantizedCall`.

        If an argument to the quantized call is marked as quantized, add an
        annotation to the producer of the argument. Otherwise, evaluate as usual.
        """
        node.func.visit(self)
        for arg, quantized in _iter_quantized_arguments(node):
            arg.visit(self)
            if quantized:
                self._ensure_quantized_node(arg)

    def evaluate_For(self, node: libcst.For) -> None:
        """Evaluate a For node.

        This method evaluates the For node by first evaluating the iter expression,
        then entering a new scope and evaluating the target and body of the loop.
        It also re-evaluates the loop body with repeated evaluation enabled to
        simulate the behavior of variables defined in the loop body being reachable
        by all statements in the loop body.
        """
        _assert_subnode_is_none(node, "orelse")

        # evaluate iter before any assignment
        node.iter.visit(self)

        def _visit_loop() -> None:
            for target in node_processing.unpack_sequence_expression(node.target):
                self.record_assignment(target, producer=node.target, is_quantized=False)
            node.body.visit(self)

        # First visit the body, then re-visit it again to simulate the
        # reachbability of all assignment in the loop body within the
        # entire loop body. Since the target assignment is executed on each
        # iteration, we also re-assign this to ensure correct tracking.
        with self.enter_scope(is_loop=True) as for_scope:
            _visit_loop()
            if not for_scope.is_terminated:
                with self.reevaluate_within_active_scope():
                    _visit_loop()

        # Here we merge the for_scope with the active scope.
        # Possible improvement: if we can infer that the loop body is always
        # entered (e.g., `node.iter` represents range(...) or a non-empty
        # list), we can overwrite the active scope.
        self._active_scope.merge(for_scope, inplace=True)

    def evaluate_While(self, node: libcst.While) -> None:
        """Evaluate a While node.

        This method evaluates the While node by first evaluating the test expression,
        then entering a new scope and evaluating the body of the loop. It also
        re-evaluates the loop body with repeated evaluation enabled to simulate the
        behavior of variables defined in the loop body being reachable by all
        statements in the loop body.
        """
        _assert_subnode_is_none(node, "orelse")

        node.test.visit(self)

        # First visit the body, then re-visit it again to simulate the
        # reachbability of all assignment in the loop body within the entire
        # loop body. Since the test expression is executed on each iteration,
        # we also re-evaluate this to ensure correct tracking.
        with self.enter_scope(is_loop=True) as while_scope:
            node.body.visit(self)
            if not while_scope.is_terminated:
                with self.reevaluate_within_active_scope():
                    node.test.visit(self)
                    node.body.visit(self)

        # Here we merge the while_scope with the active scope.
        # Possible improvement: if we can infer that the loop body is always
        # entered (e.g., `node.test` always evaluates to True), we can
        # overwrite the active scope.
        self._active_scope.merge(while_scope, inplace=True)

    def evaluate_With(self, node: libcst.With) -> None:
        """Evaluate a With node.

        This method evaluates the With node by first evaluating the items in
        the with statement, then evaluating the body of the with statement.

        For each item in the with statement, if an asname is provided, it
        records an assignment to the asname in the current scope.
        """
        for item in node.items:
            if item.asname is not None:
                for target in node_processing.unpack_sequence_expression(item.asname.name):
                    self.record_assignment(target, item, False)
        node.body.visit(self)

    def evaluate_ListComp(self, node: libcst.ListComp) -> None:
        """Evaluate a List Comprehension node.

        This method evaluates the ListComp node by entering a new scope and
        evaluating the for_in expression and the elt expression.
        """
        with self.enter_scope():
            node.for_in.visit(self)
            node.elt.visit(self)

    def evaluate_DictComp(self, node: libcst.DictComp) -> None:
        """Evaluate a Dict Comprehension node.

        This method evaluates the `DictComp` node by entering a new scope and
        evaluating the `for_in` expression, `key` expression and the `value`
        expression.
        """
        with self.enter_scope():
            node.for_in.visit(self)
            node.key.visit(self)
            node.value.visit(self)

    def evaluate_SetComp(self, node: libcst.SetComp) -> None:
        """Evaluate a Set Comprehension node.

        This method evaluates the `SetComp` node by entering a new scope and
        evaluating the `for_in` expression and the `elt` expression.
        """
        with self.enter_scope():
            node.for_in.visit(self)
            node.elt.visit(self)

    def evaluate_GeneratorExp(self, node: libcst.GeneratorExp) -> None:
        """Evaluate a generator expression node.

        This method evaluates the GeneratorExp node by entering a new scope and
        evaluating the for_in expression and the elt expression.
        """
        with self.enter_scope():
            node.for_in.visit(self)
            node.elt.visit(self)

    def evaluate_CompFor(self, node: libcst.CompFor) -> None:
        """Evaluate a `CompFor` node.

        Records an assignment to the target(s) in the current scope.
        """
        node.iter.visit(self)
        for if_node in node.ifs:
            if_node.visit(self)

        for target in node_processing.unpack_sequence_expression(node.target):
            self.record_assignment(target, node.target, False)

        if node.inner_for_in is not None:
            node.inner_for_in.visit(self)

    def evaluate_Name(self, node: libcst.Name) -> None:
        """Evaluate Name node.

        If `node` appears in a `LOAD` context, update the active with a use,
        also ensure that all relevant existing annotations uses list is
        updated.
        """
        ctx = self._provider.get_metadata(_ExpressionContextProvider, node, None)
        target = node.value
        if ctx is ExpressionContext.LOAD:
            for quant_status in self._active_scope[target]:
                quant_status.uses.append(node)

                for annotation in self.get_annotations(quant_status.producer, target=target):
                    annotation.uses.append(node)

    def _ensure_quantized_node(self, node: libcst.CSTNode) -> None:
        """Ensures that a given node is quantized.

        Ensuring quantization here means quantization annotations imply
        quantization. The actual quantization transformation is handled in a
        downstream transformer.

        This method checks the type of the node and performs the necessary
        actions to ensure that it is quantized. Currently, it only supports
        quantizing Name nodes.
        """
        match node:
            case libcst.Name(value=var):
                self._ensure_quantized_var(var)
            case _:
                pass

    def _ensure_quantized_var(self, var: str) -> None:
        """Ensures that a variable, identified by name, is quantized.

        Ensuring quantization here means quantization annotations imply
        quantization. The actual quantization transformation is handled in a
        downstream transformer.
        """
        for assignment in self._active_scope[var]:
            is_quantized = assignment.metadata.is_quantized
            if not is_quantized:
                annotation = QuantizationAnnotation(target=var, uses=assignment.uses[:])
                producer = assignment.producer

                annotation_set = self.get_annotations(producer)
                assert isinstance(annotation_set, set)
                annotation_set.add(annotation)
                self._provider.set_metadata(assignment.producer, annotation_set)

                assignment.metadata.is_quantized = True

    @overload
    def get_annotations(
        self, node: libcst.CSTNode, target: str
    ) -> Iterator[QuantizationAnnotation]: ...
    @overload
    def get_annotations(
        self, node: libcst.CSTNode, target: None = None
    ) -> set[QuantizationAnnotation]: ...

    def get_annotations(
        self, node: libcst.CSTNode, target: str | None = None
    ) -> set[QuantizationAnnotation] | Iterator[QuantizationAnnotation]:
        """Retrieves the quantization annotations for a given node.

        If a target is specified, this method returns an iterator over the annotations
        that match the target. Otherwise, it returns a set of all annotations for the node.

        Args:
            node: The node for which to retrieve annotations.
            target: The target to filter annotations by, or None to retrieve all annotations.

        Returns:
            An iterator over the annotations that match the target, or a set of all annotations.
        """
        annotation_set = self._provider.get_metadata(QuantizationAnnotationProvider, node, set())
        if target is None:
            return annotation_set
        else:
            return (a for a in annotation_set if a.target == target)


def _if_node_is_exhaustive(node: libcst.If) -> bool:
    """Checks if an if node is exhaustive.

    An if node is considered exhaustive if it has an else branch or if its
    orelse branch is also an if node that is exhaustive.

    Args:
        node: The if node to check.

    Returns:
        bool: True if the if node is exhaustive, False otherwise.
    """
    match node.orelse:
        case libcst.Else():
            return True
        case libcst.If():
            return _if_node_is_exhaustive(node.orelse)
        case None:
            return False


def _iter_quantized_arguments(
    node: nodes.QuantizedCall,
) -> Iterator[tuple[libcst.BaseExpression, bool]]:
    """Iterate over the quantized arguments of a QuantizedCall node.

    This function yields tuples containing the argument expression and a boolean
    indicating whether the argument is quantized.

    Args:
        node: The QuantizedCall node to iterate over.

    Yields:
         A tuple containing the argument expression and a boolean indicating
         whether the argument is quantized.
    """
    args: list[libcst.BaseExpression] = []
    kwargs: dict[str, libcst.BaseExpression] = {}
    seen_kwarg = False
    for arg in node.args:
        if (kw := arg.keyword) is not None:
            kwargs[kw.value] = arg.value
            seen_kwarg = True
        elif seen_kwarg:
            raise SyntaxError("Positional argument follows keyword argument")
        else:
            args.append(arg.value)

    for param, value in node.operator.bind_partial(*args, **kwargs):
        yield value, param.quantized


def _assert_subnode_is_none(node: libcst.CSTNode, attr: str) -> None:
    """Asserts that a given subnode of a CSTNode is None.

    This function checks if a subnode of a CSTNode is None. If the subnode is not None,
    it raises a NotImplementedError with a message indicating that the use of the subnode
    is not supported.

    Args:
        node: The CSTNode to check.
        attr: The name of the subnode to check.

    Raises:
        NotImplementedError: If the subnode is not None.
    """
    if getattr(node, attr, None) is not None:
        msg = f"The use of '{attr}' on '{type(node).__name__}' is not supported"
        raise NotImplementedError(msg)
