# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


import libcst
import pytest
import syrupy

from fastforward._autoquant.cst.passes import ExpandAugAssign
from fastforward.testing.autoquant import autoquantize_str


def _expand(src: str) -> str:
    module = libcst.parse_module(src)
    return module.visit(ExpandAugAssign()).code


def test_expand_aug_assign_rewrites_simple_target() -> None:
    # GIVEN a function using ``+=`` on a Name target
    src = "def f(a, b):\n    a += b\n"

    # WHEN ExpandAugAssign processes the source
    result = _expand(src)

    # THEN the augmented assignment is rewritten as a regular assignment with a binary op
    assert result == "def f(a, b):\n    a = a + b\n"


@pytest.mark.parametrize(
    ("aug_op", "binary_op"),
    [
        ("+=", "+"),
        ("-=", "-"),
        ("*=", "*"),
        ("/=", "/"),
        ("//=", "//"),
        ("%=", "%"),
        ("**=", "**"),
        ("<<=", "<<"),
        (">>=", ">>"),
        ("&=", "&"),
        ("|=", "|"),
        ("^=", "^"),
        ("@=", "@"),
    ],
)
def test_expand_aug_assign_supports_all_binary_aug_ops(aug_op: str, binary_op: str) -> None:
    # GIVEN a function with a specific augmentation operator
    src = f"def f(a, b):\n    a {aug_op} b\n"

    # WHEN ExpandAugAssign processes the source
    result = _expand(src)

    # THEN the operator is expanded to its binary counterpart
    assert result == f"def f(a, b):\n    a = a {binary_op} b\n"


def test_expand_aug_assign_supports_attribute_target() -> None:
    # GIVEN an AugAssign on an Attribute target
    src = "def f(self, b):\n    self.x += b\n"

    # WHEN ExpandAugAssign processes the source
    result = _expand(src)

    # THEN the target is preserved on both sides of the assignment
    assert result == "def f(self, b):\n    self.x = self.x + b\n"


def test_expand_aug_assign_supports_subscript_target() -> None:
    # GIVEN an AugAssign on a Subscript target
    src = "def f(a, b, i):\n    a[i] += b\n"

    # WHEN ExpandAugAssign processes the source
    result = _expand(src)

    # THEN the subscript expression is preserved on both sides of the assignment
    assert result == "def f(a, b, i):\n    a[i] = a[i] + b\n"


def test_autoquant_quantizes_inplace_operations(
    snapshot: syrupy.assertion.SnapshotAssertion,
) -> None:
    # GIVEN a forward method using several augmentation operators
    input = """
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x += y
        x *= y
        x -= y
        return x
    """

    # WHEN the function is autoquantized
    result = autoquantize_str(input)

    # THEN every augmented assignment is rewritten into its quantized binary
    # counterpart and matches the recorded snapshot
    assert snapshot == result
