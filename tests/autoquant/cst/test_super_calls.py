# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


import libcst
import pytest

from fastforward._autoquant.cst import super_calls


def _funcdef(source: str) -> libcst.FunctionDef:
    statement = libcst.parse_statement(source)
    assert isinstance(statement, libcst.FunctionDef), "invalid parametrization"
    return statement


@pytest.mark.parametrize(
    "source, owner_class_name, expected",
    [
        pytest.param(
            "def forward(self, x):\n    return super().forward(x)\n",
            "Owner",
            {"forward": None},
            id="bare_super",
        ),
        pytest.param(
            "def forward(self, x):\n    return super(Owner, self).forward(x)\n",
            "Owner",
            {"forward": None},
            id="explicit_super_to_owner",
        ),
        pytest.param(
            "def forward(self, x):\n"
            "    a = super().forward(x)\n"
            "    b = super().helper(x)\n"
            "    return a + b\n",
            "Owner",
            {"forward": None, "helper": None},
            id="multiple_methods",
        ),
        pytest.param(
            "def forward(self, x):\n    y = super().forward(x)\n    return super().forward(y)\n",
            "Owner",
            {"forward": None},
            id="repeated_calls_deduplicated",
        ),
        pytest.param(
            "def forward(self, x):\n    return super().helper(x)\n",
            "Owner",
            {"helper": None},
            id="delegates_to_other_method_name",
        ),
        pytest.param(
            "def forward(self, x):\n    return x\n",
            "Owner",
            {},
            id="no_super_call",
        ),
        pytest.param(
            "def forward(self, x):\n    return super(OtherAncestor, self).forward(x)\n",
            "Owner",
            {"forward": "OtherAncestor"},
            id="deliberate_mro_skip",
        ),
        pytest.param(
            "def forward(self, x):\n    delegate = super()\n    return delegate.forward(x)\n",
            "Owner",
            {},
            id="super_object_bound_to_name",
        ),
    ],
)
def test_super_delegated_methods(
    source: str, owner_class_name: str, expected: dict[str, str | None]
) -> None:
    # GIVEN a method body that may or may not call through `super()` to `owner_class_name`'s
    # immediate superclass, or to some other ancestor via a deliberate MRO skip

    # WHEN it is scanned for super() delegation
    delegated = super_calls.super_delegated_methods(_funcdef(source), owner_class_name)

    # THEN exactly the methods invoked through a `super()` call are reported, mapped to their
    # explicit anchor class name, or `None` if the call targets the immediate superclass
    assert delegated == expected


@pytest.mark.parametrize(
    "call_source, owner_class_name, self_name, expected",
    [
        pytest.param("super()", "Owner", None, True, id="bare_super"),
        pytest.param("super(Owner, self)", "Owner", None, True, id="explicit_matches_owner"),
        pytest.param("super(Owner, self)", "Owner", "self", True, id="explicit_matches_self_name"),
        pytest.param("super(Owner, obj)", "Owner", "self", False, id="explicit_self_name_mismatch"),
        pytest.param("super(OtherAncestor, self)", "Owner", None, False, id="explicit_mro_skip"),
        pytest.param("super(Owner)", "Owner", None, False, id="wrong_arg_count"),
    ],
)
def test_super_call_targets_owner(
    call_source: str, owner_class_name: str, self_name: str | None, expected: bool
) -> None:
    # GIVEN a `super(...)` call expression

    # WHEN checked against the owning class (and optionally the `self` parameter name)
    call = libcst.parse_expression(call_source)
    assert isinstance(call, libcst.Call), "invalid parametrization"
    targets_owner = super_calls.super_call_targets_owner(call, owner_class_name, self_name)

    # THEN whether it targets the immediate superclass of `owner_class_name` matches expectations
    assert targets_owner is expected
