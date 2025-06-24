# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from collections.abc import Sequence

import libcst


class NotImplementedMixin(libcst.CSTTransformer):
    """A mixin class for CSTTransformers that provides a warning mechanism.

    This transformer monitors nodes with a `leading_lines` attribute. It allows
    calling `warn_not_implemented` at any point during tree traversal. The
    warning message is then inserted into the closest node that wraps the
    current node and has a `leading_lines` attribute.

    Attributes:
        _visit_stack: A stack of visited CST nodes.
        _not_implemented_warnings: A dictionary mapping CST nodes to lists of warnings for unimplemented functionality.
    """

    def __init__(self) -> None:
        self._visit_stack: list[libcst.CSTNode] = []
        self._not_implemented_warnings: dict[libcst.CSTNode, list[str]] = {}

    def on_visit(self, node: libcst.CSTNode) -> bool:
        if hasattr(node, "leading_lines"):
            self._visit_stack.append(node)
            self._not_implemented_warnings[node] = []
        return super().on_visit(node)

    def on_leave(
        self, original_node: libcst.CSTNodeT, updated_node: libcst.CSTNodeT
    ) -> libcst.CSTNodeT | libcst.RemovalSentinel | libcst.FlattenSentinel[libcst.CSTNodeT]:
        """General on_leave impl for mixin.

        Check if the current node is the last one in the visit stack. If it is,
        it removes the node from the stack and checks if there are any
        remaining warnings. If there are, it adds them to the leading lines of
        the current node or the next node in the stack if the current node has
        been replaced.

        Args:
            original_node: The original node being left.
            updated_node: The updated node being left.

        Returns:
            The transformed node
        """
        transformed_node = super().on_leave(original_node=original_node, updated_node=updated_node)

        if self._visit_stack and original_node is self._visit_stack[-1]:
            self._visit_stack.pop()

            remaining_warnings = self._not_implemented_warnings[original_node]
            if not isinstance(transformed_node, type(original_node)):
                if self._visit_stack:
                    self._not_implemented_warnings[self._visit_stack[-1]] += remaining_warnings
                return transformed_node

            leading_lines: Sequence[libcst.EmptyLine] = transformed_node.leading_lines  # type: ignore[attr-defined]
            warning_lines: list[libcst.EmptyLine] = []
            for warning in reversed(remaining_warnings):
                warning_lines.append(
                    libcst.EmptyLine(comment=libcst.Comment(f"# WARNING: {warning}"))
                )

            transformed_node = transformed_node.with_changes(
                leading_lines=tuple(leading_lines) + tuple(warning_lines)
            )
            del self._not_implemented_warnings[original_node]

        return transformed_node

    def warn_not_implemented(self, msg: str) -> None:
        """Issues a warning for an unimplemented feature or functionality.

        Args:
            msg (str): The message to be associated with the warning.

        Raises:
            RuntimeError: If no elements are present in the visit stack, indicating that
                        there is no context to attach the warning to.

        Notes:
            The warning is attached to the last element in the visit stack. If the visit
            stack is empty, a RuntimeError is raised.
        """
        if not self._visit_stack:
            msg = "Cannot attach warning because current node has no container node as predecessor"
            raise RuntimeError(msg)

        self._not_implemented_warnings[self._visit_stack[-1]].append(msg)
