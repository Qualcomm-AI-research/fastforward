# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Detection of `super()` usage in methods that are converted by autoquant.

A method that delegates to `super()` relies on an implementation that lives on
a superclass. Autoquant copies methods into a newly generated quantized class,
so such a delegation must reach the *quantized* counterpart of the superclass
implementation rather than the original one. This module locates the `super()`
callsites; `fastforward._autoquant.autoquant` uses the result to quantize the
relevant superclass and inherit from it.
"""

import libcst
import libcst.matchers as m

_SUPER_ATTRIBUTE_CALL = m.Call(
    func=m.Attribute(
        value=m.Call(func=m.Name("super")),
        attr=m.Name(),
    )
)

_INVALID = object()
"""Sentinel returned by `_super_call_anchor` for a malformed `super(...)` call."""


def super_call_targets_owner(
    super_call: libcst.Call,
    owner_class_name: str,
    self_name: str | None = None,
) -> bool:
    """Check whether `super_call` targets the immediate superclass of `owner_class_name`.

    Both bare `super()` and the explicit `super(owner_class_name, self)` form
    target, in the original source, the immediate superclass of the owning
    class. An explicit call anchored to some other ancestor (a deliberate MRO
    skip) does not.

    Args:
        super_call: CST of the `super(...)` call itself, i.e. the `super()`
            part of a `super().method()` expression.
        owner_class_name: Name of the class that defines the method
            containing `super_call`.
        self_name: Name of the enclosing method's `self` parameter. When
            given, the explicit two-argument form only matches if its second
            argument is this exact name. When omitted, any name is accepted
            for the second argument, since the caller has no way to know
            which parameter is `self`.

    Returns:
        Whether `super_call` targets the immediate superclass of
        `owner_class_name`.
    """
    if len(super_call.args) == 0:
        return True
    if len(super_call.args) != 2:
        return False

    first, second = super_call.args
    if not (isinstance(first.value, libcst.Name) and first.value.value == owner_class_name):
        return False
    if not isinstance(second.value, libcst.Name):
        return False
    if self_name is not None and second.value.value != self_name:
        return False
    return True


def _super_call_anchor(super_call: libcst.Call, owner_class_name: str) -> str | None | object:
    """Return the explicit anchor class name of `super_call`.

    Returns `None` if it targets the immediate superclass of `owner_class_name`
    (bare `super()`, or explicit `super(owner_class_name, self)`).

    Returns `_INVALID` for anything else that is not a recognized two-argument
    `super(SomeClass, self)` form.
    """
    if len(super_call.args) == 0:
        return None
    if len(super_call.args) != 2:
        return _INVALID

    first, second = super_call.args
    if not (isinstance(first.value, libcst.Name) and isinstance(second.value, libcst.Name)):
        return _INVALID
    if first.value.value == owner_class_name:
        return None
    return first.value.value


def super_delegated_methods(
    funcdef: libcst.CSTNode, owner_class_name: str
) -> dict[str, str | None]:
    """Find the method names that `funcdef` invokes through `super()`, and how they are anchored.

    Both bare `super()` and the explicit `super(owner_class_name, self)` form
    target, in the original source, the immediate superclass of the owning
    class. An explicit call anchored to some other ancestor (a deliberate MRO
    skip) targets that ancestor directly instead -- it is still reported, but
    with its explicit anchor name rather than `None`, so callers can decide
    how to handle the skip.

    Args:
        funcdef: CST of the method to inspect.
        owner_class_name: Name of the class that defines `funcdef`.

    Returns:
        A mapping from each method name invoked through a `super()` call to
        the explicit anchor class name used, or `None` if the call targets
        the immediate superclass of `owner_class_name` (bare `super()`, or
        explicit `super(owner_class_name, self)`). For example,
        `super().forward(x)` yields `{"forward": None}`, while
        `super(Ancestor, self).forward(x)` yields `{"forward": "Ancestor"}`.
        Empty if `funcdef` contains no such delegation.
    """
    result: dict[str, str | None] = {}
    for call in m.findall(funcdef, _SUPER_ATTRIBUTE_CALL):
        assert isinstance(call, libcst.Call)
        func = call.func
        assert isinstance(func, libcst.Attribute)
        assert isinstance(func.attr, libcst.Name)

        super_call = func.value
        assert isinstance(super_call, libcst.Call)

        anchor = _super_call_anchor(super_call, owner_class_name)
        if anchor is _INVALID:
            continue
        result[func.attr.value] = anchor  # type: ignore[assignment]
    return result
