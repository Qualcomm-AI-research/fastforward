# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from collections.abc import Sequence
from typing import Any, Callable, ParamSpec, TypeVar

import libcst
import pytest

from fastforward._autoquant.autoquant import default_optable, default_source_context
from fastforward._autoquant.cfg import blocks, construct
from fastforward._autoquant.pysource import SourceContext
from fastforward._import import fully_qualified_name
from fastforward._quantops import OperatorTable

_P = ParamSpec("_P")
_T = TypeVar("_T")


_CASE_MARKER = object()


class _MetaCaseCollector(type):
    """Metaclass for `FunctionCaseCollector`.

     A metaclass that collects functions marked as cases in a class and creates a
    `case` fixture for them.
    """

    def __new__(cls, name: str, bases: tuple[type, ...], dct: dict[str, Any], **kwds: Any) -> type:
        # Collect all functions marked as case using the case decorator or
        # functions whose name starts with "case_".
        cases = []
        for name, value in dct.items():
            has_marker = getattr(value, "_cfg_test_case_marker", None) is _CASE_MARKER
            has_case_prefix = name.startswith("case_")
            if has_marker or has_case_prefix:
                cases.append(value)

        # If one or more cases are collected, create a `case` fixture for these
        # functions.
        if cases:
            dct["case"] = _case_fixture(cases)

        return super().__new__(cls, name, bases, dct, **kwds)


def _case_fixture(cases: Sequence[Callable[..., Any]]) -> Callable[..., Any]:
    @pytest.fixture(scope="class", params=tuple(cases))
    def case(_: Any, request: pytest.FixtureRequest) -> Callable[..., Any]:
        return request.param  # type: ignore[no-any-return]

    return case


class FunctionCaseCollector(metaclass=_MetaCaseCollector):
    """Baseclass for test classes with functions as test data.

    This class is subclassed to automatically collect functions in a class that
    are marked as cases using the `case` decorator or by prefixing the function
    name with "case_". It then creates a `case` fixture that can be used in
    test functions to iterate over the collected cases.

    The `case` fixture is created at class creation time and is available as a
    class attribute. It can be used in test functions to iterate over the
    collected cases.

    Example:
    ```python
    class MyTest(FunctionCaseCollector):
        def case_foo(self):
            pass

        def case_bar(self):
            pass

        def test_something(self, case):
            # case will be one of case_foo or case_bar
            pass
    ```
    """

    @staticmethod
    def case(func: Callable[_P, _T]) -> Callable[_P, _T]:
        """Decorator for marking a function as case."""
        func._cfg_test_case_marker = _CASE_MARKER  # type: ignore[attr-defined]
        return func


class CFGTest(FunctionCaseCollector):
    """CFGTest case collector."""

    @pytest.fixture(scope="class")
    def optable(self) -> OperatorTable:
        """Fixture for `OperatorTable`.

        Can be overridden in a subclass to provide a non-default
        `OperatorTable`.
        """
        return default_optable()

    @pytest.fixture(scope="class")
    def source_context(self) -> SourceContext:
        """Fixture for `SourceContext`.

        Can be overridden in a subclass to provide a non-default
        `SourceContext`.
        """
        return default_source_context()

    @pytest.fixture(scope="class")
    def raw_cst(
        self,
        case: Callable[..., Any],
        source_context: SourceContext,
    ) -> libcst.FunctionDef:
        """Fixture for raw CST.

        Will convert all `case`s to a CST without any further processing other
        than that specified by the `source_context` fixture.

        It is recommended that further post-processing of the CST is performed
        in the `cst` fixture.
        """
        fcn = fully_qualified_name(type(self))
        src = source_context.get(fcn).member(case.__name__)
        cst = src.cst(NodeType=libcst.FunctionDef)
        assert isinstance(cst, libcst.FunctionDef)
        return cst

    @pytest.fixture(scope="class")
    def cst(self, raw_cst: libcst.FunctionDef) -> libcst.FunctionDef:
        """Fixture that takes `raw_cst` and further processes the CST.

        This can be used to perform passes on top of the raw CST before the
        test function. It is recommended to override this function instead of
        `raw_cst`.
        """
        return raw_cst

    @pytest.fixture(scope="class")
    def cfg(self, cst: libcst.FunctionDef) -> blocks.FunctionBlock:
        """Construct a CFG from the output of `cst`."""
        return construct(cst)
