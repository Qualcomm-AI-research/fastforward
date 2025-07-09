# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import libcst

from fastforward._autoquant.cst.quantizer_analysis.unimplemented import NotImplementedMixin
from fastforward.testing.string import assert_strings_match_verbose, dedent_strip


class _TestVisitor(NotImplementedMixin):
    def visit_For(self, _node: libcst.For) -> None:
        self.warn_not_implemented("For Loop Warning")

    def leave_CompFor(
        self, _original_node: libcst.CompFor, updated_node: libcst.CompFor
    ) -> libcst.CompFor:
        node_code = libcst.Module([]).code_for_node(updated_node.iter)
        msg = f"CompFor Loop Warning [{node_code}]"
        self.warn_not_implemented(msg)
        return updated_node

    def visit_Arg(self, node: libcst.Arg) -> None:
        node_code = libcst.Module([]).code_for_node(node)
        self.warn_not_implemented(f"Arg warning [{node_code}]")


def test_NotImplementedMixin() -> None:
    # GIVEN source code
    (src,) = dedent_strip("""
    def my_function():
        sum = 0
        for i in range(3):
            sum += i
            
        asum = [i + j for i in range(3) for j in range(4)]
        
        print({i: float(i) for i in [1,12,32]})
    """)

    # WHEN a visitor that produces inline warnings using NotImplementedMixin
    cst = libcst.parse_module(src)
    cst = cst.visit(_TestVisitor())

    # THEN the new code (represented as CST) must match the expected code.
    (actual,) = dedent_strip(cst.code)
    (expected,) = dedent_strip("""
    def my_function():
        sum = 0
        # WARNING: Arg warning [3]
        # WARNING: For Loop Warning
        for i in range(3):
            sum += i

        # WARNING: CompFor Loop Warning [range(3)]
        # WARNING: CompFor Loop Warning [range(4)]
        # WARNING: Arg warning [4]
        # WARNING: Arg warning [3]
        asum = [i + j for i in range(3) for j in range(4)]

        # WARNING: CompFor Loop Warning [[1,12,32]]
        # WARNING: Arg warning [i]
        # WARNING: Arg warning [{i: float(i) for i in [1,12,32]}]
        print({i: float(i) for i in [1,12,32]})
    """)

    assert_strings_match_verbose(actual, expected)
