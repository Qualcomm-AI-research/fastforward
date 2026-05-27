# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


import contextlib
import functools
import logging
import multiprocessing as mp
import operator
import pathlib
import re
import subprocess
import sys
import types

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, TypeAlias, cast
from unittest import mock
from unittest.mock import patch

import fastforward
import fastforward as ff
import libcst
import pytest
import syrupy
import torch

from fastforward._autoquant import pybuilder, pysource
from fastforward._autoquant.autoquant import (
    _find_known_quantized_modules,
    _propagate_quantizers,
    _QuantizedFunctionSpec,
    _QuantizerRefTrace,
    autoquant,
    autoquant_with_defaults,
    codeformat_with_defaults,
    default_source_context,
)
from fastforward._autoquant.cst import passes
from fastforward._autoquant.cst.nodes import QuantizerReference
from fastforward._autoquant.pysource import SourceContext
from fastforward._autoquant.pysource.scope import ImportSymbol
from fastforward._quantops import OperatorTable, optable
from fastforward.autoquant import autoquantize
from fastforward.testing.metrics import sqnr as metric_sqnr
from fastforward.testing.string import assert_strings_match_verbose
from torch import Tensor as TensorAlias  # required for tests, do not remove
from typing_extensions import override

from tests._core_package_version_utils import TORCH_VERSION

Tensor: TypeAlias = torch.Tensor  # Required for tests, do not remove


class _AssertNoAssignments(libcst.CSTVisitor):
    @override
    def visit_Assign(self, node: libcst.Assign) -> bool | None:
        assert False, "CST contains Assign node"

    @override
    def visit_AugAssign(self, node: libcst.AugAssign) -> bool | None:
        assert False, "CST contains AugAssign node"

    @override
    def visit_AnnAssign(self, node: libcst.AnnAssign) -> bool | None:
        assert False, "CST contains AnnAssign node"


@pytest.mark.slow
def test_default_source_context_wraps_assignment_nodes() -> None:
    # GIVEN the default source context
    source_context = default_source_context(use_type_inference=False)

    # WHEN a CST is obtained through default source context
    cst = source_context.get_cst("torch.nn.modules.conv", NodeType=libcst.Module)

    # THEN, the must be no Assign, AnAssign, or AugAssign in the
    # CST anymore.
    _ = cst.visit(_AssertNoAssignments())

    # GIVEN a source context that does not remove assignment nodes
    source_context = SourceContext()

    # WHEN a CST is obtained through this source context
    cst = source_context.get_cst("torch.nn.modules.conv", NodeType=libcst.Module)

    # THEN it is expected that the CST contains no assignment nodes
    with pytest.raises(AssertionError):
        _ = cst.visit(_AssertNoAssignments())


# ------------------------------------------------------------------------------


# Example module with an __init__ function and attributes
class ExampleModule1(torch.nn.Module):
    def __init__(self, z: torch.Tensor):
        super().__init__()
        self.z = z

    def forward(self, x: torch.Tensor) -> tuple[TensorAlias, torch.Tensor]:
        y = torch.sigmoid(x)
        return self.z, torch.relu(y)


# ------------------------------------------------------------------------------


# Example module without __init__ function
class ExampleModule2(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.nn.functional.conv2d(x, x)
        return torch.nn.functional.linear(y, y)


# ------------------------------------------------------------------------------


# Example module with binary operators
class ExampleModule3(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        s = x + y
        s = x | y
        s = x ^ y
        s = x / y
        s = x // y
        s = x << y
        s = x @ y
        s = x % y
        s = x * y
        s = x**y
        s = x >> y
        s = x - y
        return s


# ------------------------------------------------------------------------------


# Example module with unary operators
class ExampleModule4(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = +x
        s = -x
        s = ~x
        return s


# ------------------------------------------------------------------------------


# Example with local variable re-assignment with non-quantized functions
class ExampleModule5(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(x)
        x = self.do_something(x)
        x = self.do_something(x)
        x = torch.sigmoid(x)
        return x

    def do_something(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


# ------------------------------------------------------------------------------


# Example with sub- and sub-sub-modules that do not have manually quantized counterparts
class ExampleSubModule6(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.module_1 = torch.nn.Identity()
        self.module_2 = torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.module_1(x)
        x = self.module_2(x)
        return x


class ExampleModule6(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.module = ExampleSubModule6()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.module(x)
        return x


# ------------------------------------------------------------------------------


# Example with manually quantized counterparts
class ExampleModule7(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.module = torch.nn.Linear(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.module(x)
        return x


def _attribute_alias_target(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)


_attribute_alias_primary = _attribute_alias_target
_attribute_alias_secondary = _attribute_alias_target
_THIS_MODULE = sys.modules[__name__]


class ExampleModuleAttributeAlias(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, _THIS_MODULE._attribute_alias_secondary(x))


class ExampleModuleAttributeAliasPrimary(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, _THIS_MODULE._attribute_alias_primary(x))


def _public_api_impl(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)


public_api = _public_api_impl


class ExampleModulePublicApiAlias(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, _THIS_MODULE.public_api(x))


def _normalize_torch_injected_docstrings(code: str) -> str:
    """Normalize a torch version >= 2.9 injected method docstring.

    Torch >= 2.9 may inject a default docstring in generated Identity.forward:
      \"\"\"Runs the forward pass.\"\"\"
    Strip this exact block so snapshots stay stable across torch versions.
    """
    if TORCH_VERSION.release[:2] < (2, 9):
        return code

    return re.sub(
        r"\n([ \t]+)\"\"\"\n\1Runs the forward pass\.\n\1\"\"\"\n",
        "\n",
        code,
    )


# ------------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "input_module",
    [
        ExampleModule1(z=torch.tensor([0])),
        ExampleModule2(),
        ExampleModule3(),
        ExampleModule4(),
        ExampleModule5(),
        ExampleModule6(),
        ExampleModule7(),
    ],
    ids=[f"case-{i}" for i in range(1, 8)],
)
def test_autoquant_introduces_quantization_method(
    input_module: torch.nn.Module,
    snapshot: syrupy.assertion.SnapshotAssertion,
) -> None:
    """Verifies autoquantization introduces the magic method and quantizers."""
    # GIVEN a torch module with a forward pass and quantizable function calls

    # GIVEN the default operator table
    operator_table = optable.OperatorTable.from_yaml(
        alias_extensions=optable.STR_ALIASES_EXTENSIONS
    )

    # GIVEN a SourceContext with a minimal set of preprocessing passes
    source_context = pysource.SourceContext(
        preprocessing_passes=[
            passes.MarkReplacementCandidates(),
            passes.WrapAssignments(),
        ]
    )

    # WHEN we autoquantize the example module
    autoquant_code = autoquant(
        module=input_module, source_context=source_context, operator_table=operator_table
    )
    actual_code = codeformat_with_defaults(code=autoquant_code)
    actual_code = _normalize_torch_injected_docstrings(actual_code)

    # THEN the generated code matches an earlier snapshot
    assert snapshot == actual_code.strip()


# --------------------------------------------------------------------------------


# Example with literal integer
class ExampleModule8(torch.nn.Module):
    num_features: int = 3

    def forward(self, x: Tensor) -> Tensor:
        h = x.reshape((-1 + 2 // 3, self.num_features))
        h = h.reshape((999 - 12, self.num_features))
        h = h.reshape((-1, self.num_features))
        return h


# --------------------------------------------------------------------------------


# Example with loops
class ExampleModule9(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        for _ in range(3):
            x = torch.sigmoid(x)
        x = torch.relu(x)
        test = True
        while test:
            for _ in range(3):
                x = _my_func(x)
            test = False
        return torch.relu(x)


# --------------------------------------------------------------------------------


# Example with quantized loop variables
class ExampleModule10(torch.nn.Module):
    def forward(self, x: Tensor) -> Any:
        for a in x:
            _ = torch.relu(a)
        for a in x:
            if 0 > 0:
                break
            _ = torch.relu(a)
        for a in x:
            if 0 > 0:
                continue
            return
        return x


# --------------------------------------------------------------------------------


# Example with `with` statement
class ExampleModule11(torch.nn.Module):
    def forward(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        with _my_context(x) as xx, _my_context(y) as yy, _my_context(x):
            if True:
                pass
            out1 = torch.relu(xx)
            out2 = torch.sigmoid(yy)

        return out1, out2


# --------------------------------------------------------------------------------


# Example with docstring and empty assignment
class FloatModule12(torch.nn.Module):
    def forward(self, x: torch.Tensor, meta: OperatorTable) -> torch.Tensor:
        """I am an important docstring.

        How multiline of me!
        """
        del meta  # meta is included to test for Symbol imports in params
        y: Tensor
        y = torch.zeros([0])
        return x + y


# --------------------------------------------------------------------------------


# Example with comprehensions
class ExampleModule13(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[Any]]:
        return [torch.relu(z) for y in x for z in y], [x + x for x in x if x > 0]


# --------------------------------------------------------------------------------


# Example with branches
class ExampleModule14(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> Iterable[Tensor]:
        if 0 > 0:
            return {torch.relu(y) for y in x}
        elif 0 == 0:
            return (torch.relu(y) for y in x)
        else:
            return {torch.relu(y): torch.sigmoid(z) for y, z in x}


# --------------------------------------------------------------------------------


# When a decorator returns a wrapping function without using `functools.wraps` the name
# of the wrapping function is not updated. This case tests that the correct function
# (`custom_helper` and not `cusotm_decorator`) is quantized and the quantized function
# also has the decorator.


def custom_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return wrapper


@custom_decorator
def custom_helper(x: torch.Tensor) -> torch.Tensor:
    return x * 2


class ExampleModule15(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> Any:
        return custom_helper(x)


# --------------------------------------------------------------------------------

# When a helper function calls another helper function more than once, a unique set of
# quantizers is required for each call of the inner function.


def inner_helper(x: torch.Tensor) -> torch.Tensor:
    return x * 2


def outer_helper(x: torch.Tensor) -> torch.Tensor:
    return inner_helper(x) + inner_helper(x)


class ExampleModule16(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> Tensor:
        return outer_helper(x)


# --------------------------------------------------------------------------------

# When two functions are used that have the same `__name__` (here `sqnr`), multiple
# quantized functions must be created where each (except the first) has a count suffix.


def sqnr(a: torch.Tensor, b: torch.Tensor, flag: bool = True, eps: float = 1e-15) -> torch.Tensor:
    del flag, eps
    return a - b


class ExampleModule17(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> Tensor:
        return sqnr(x, x) + metric_sqnr(x, x)


# --------------------------------------------------------------------------------

# Testcase with a decorator on a function that calls other functions that are transformed
# to by autoquant to quantized versions. The autoquantized code must call the transformed
# functions and not the original.


def helper_inside_forward_with_decorator(x: torch.Tensor) -> torch.Tensor:
    return x * 2


def decorator_on_forward(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return wrapper


class ExampleModule18(torch.nn.Module):
    @decorator_on_forward
    def forward(self, x: torch.Tensor) -> Tensor:
        return helper_inside_forward_with_decorator(x)


# --------------------------------------------------------------------------------


class ExampleModule19(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor | float:
        out = torch.tensor(0)
        try:
            return x * x
        except AttributeError:
            out = x * 2
        finally:
            out = 2 * out
        return out


# --------------------------------------------------------------------------------

# When a caller function invokes two DIFFERENT helpers that each perform the same
# operation type (e.g. multiply), both helpers produce a quantizer with the same
# base name. The caller must receive DISTINCT kw-only quantizer parameters — one
# set per helper call — with no name collision. The naming convention prefixes each
# propagated parameter with the callee's function name to guarantee uniqueness.
#
# Example: both _two_helper_mul_a and _two_helper_mul_b need "quantizer_mul".
# The caller _two_helper_caller receives:
#   quantizer__two_helper_mul_a_mul  (propagated from _two_helper_mul_a)
#   quantizer__two_helper_mul_b_mul  (propagated from _two_helper_mul_b)


def _two_helper_mul_a(x: torch.Tensor) -> torch.Tensor:
    return x * 2


def _two_helper_mul_b(x: torch.Tensor) -> torch.Tensor:
    return x * 3


def _two_helper_caller(x: torch.Tensor) -> torch.Tensor:
    return _two_helper_mul_a(x) + _two_helper_mul_b(x)


class ExampleModuleTwoHelpersSameQuantizerName(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _two_helper_caller(x)


# --------------------------------------------------------------------------------

# When a helper passes itself as a Name reference to another function (the
# "dispatch trampoline" pattern), autoquant must rename that reference after
# the helper is renamed.  torch.nn.functional.group_norm exhibits this via
# handle_torch_function(group_norm, ...).  We replicate the pattern here with
# a synthetic helper to avoid depending on torch-internal source that varies
# across versions.


def _dispatch_trampoline(fn: Any, *args: Any) -> torch.Tensor:
    return fn(*args)  # type: ignore[no-any-return]


def _self_ref_op(x: torch.Tensor) -> torch.Tensor:
    return _dispatch_trampoline(_self_ref_op, x + x)


class ExampleModule20(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _self_ref_op(x)


# --------------------------------------------------------------------------------

# When a helper calls itself recursively, _propagate_quantizers truncates the
# cycle at first re-occurrence so the self-call requires exactly the function's
# own local quantizer set — no extra parameters are added. The generated
# recursive call therefore uses identity forwarding: quantizer_X=quantizer_X.
# Without this forwarding, every recursive invocation would raise TypeError.


def _recursive_mul(x: torch.Tensor, n: int) -> torch.Tensor:
    if n <= 0:
        return x * 2  # base case — local mul op → quantizer_mul
    return _recursive_mul(x * 2, n - 1)  # recursive step, same op


class ExampleModule21(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _recursive_mul(x, 3)


# --------------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "input_module",
    [
        ExampleModule8(),
        ExampleModule9(),
        ExampleModule10(),
        ExampleModule11(),
        FloatModule12(),
        ExampleModule13(),
        ExampleModule14(),
        ExampleModule15(),
        ExampleModule16(),
        ExampleModule17(),
        ExampleModule18(),
        ExampleModule19(),
        ExampleModule20(),
        ExampleModule21(),
    ],
    ids=[f"case-{i}" for i in range(8, 22)],
)
def test_autoquant_end_to_end(
    input_module: torch.nn.Module,
    snapshot: syrupy.assertion.SnapshotAssertion,
) -> None:
    """Verifies autoquantization introduces the magic method and quantizers."""
    # GIVEN a torch module with a forward pass and quantizable function calls

    # GIVEN the default operator table
    operator_table = optable.OperatorTable.from_yaml(
        alias_extensions=optable.STR_ALIASES_EXTENSIONS
    )

    # GIVEN a default SourceContext
    source_context = default_source_context(use_type_inference=False)

    # WHEN we autoquantize the example module
    autoquantized = autoquant(
        module=input_module, source_context=source_context, operator_table=operator_table
    )
    actual_output = codeformat_with_defaults(code=autoquantized).strip()

    # THEN the generated code matches an earlier snapshot
    assert snapshot == actual_output


class ExampleExpression(torch.nn.Module):
    def forward(self) -> None:
        print("This is not a quantized function.")


@pytest.mark.slow
def test_expressions_not_quantized(snapshot: syrupy.assertion.SnapshotAssertion) -> None:
    """Tests that expressions are not quantized (fixes #80)."""
    actual = autoquant_with_defaults(ExampleExpression(), use_type_inference=False)
    actual = codeformat_with_defaults(code=actual).strip()
    assert snapshot == actual


@pytest.mark.slow
def test_codeformat() -> None:
    """Tests that code is formatted correctly."""
    input = "1 + 2 ==3"
    expected = "1 + 2 == 3"
    actual = codeformat_with_defaults(code=input).strip()
    assert_strings_match_verbose(expected, actual)


class ExampleModule1B(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.nn.functional.conv2d(x, x)
        return torch.nn.functional.linear(y, y)


class _IssueAHelperRefModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(x, x, x)
        return y  # type: ignore[no-any-return]


def _issue_b_external_dispatch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y


def _issue_b_dispatch_op(
    x: torch.Tensor, y: torch.Tensor, *, output_quantizer: fastforward.nn.Quantizer
) -> torch.Tensor:
    del output_quantizer
    return x + y


class _IssueBDispatchImportModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _issue_b_external_dispatch(x, x)


def test_pattern_based_replacement(snapshot: syrupy.assertion.SnapshotAssertion) -> None:
    module = ExampleModule1B()
    rule = ff.autoquant.PatternRule.from_str(
        pattern="torch.nn.functional.{func}({a}, {b})",
        replacement="my_own.{func}({a}, {b})",
    )
    quantized = autoquant_with_defaults(
        module, use_type_inference=False, replacement_patterns=[rule]
    )
    quantized = codeformat_with_defaults(quantized)
    assert snapshot == quantized


@pytest.mark.slow
@pytest.mark.parametrize(
    ("input_module",),
    [
        pytest.param(ExampleModuleAttributeAlias(), id="module-alias-secondary"),
        pytest.param(ExampleModuleAttributeAliasPrimary(), id="module-alias-primary"),
        pytest.param(ExampleModulePublicApiAlias(), id="module-alias-public-api"),
    ],
)
def test_autoquant_prefers_attribute_name_for_module_alias_resolution(
    input_module: torch.nn.Module,
    snapshot: syrupy.assertion.SnapshotAssertion,
) -> None:
    actual = autoquant_with_defaults(input_module, use_type_inference=False)
    actual = codeformat_with_defaults(actual)
    assert snapshot == actual


def test_helper_public_api_refs_follow_renamed_helpers() -> None:
    code = autoquant_with_defaults(_IssueAHelperRefModule(), use_type_inference=False)
    code = codeformat_with_defaults(code)

    assert "quantized_handle_torch_function(\n            multi_head_attention_forward," not in code
    assert "quantized_handle_torch_function(\n            relu," not in code
    assert "quantized_handle_torch_function(\n            group_norm," not in code
    assert (
        "quantized_handle_torch_function(\n            quantized_multi_head_attention_forward,"
        in code
    )
    assert "quantized_handle_torch_function(\n            quantized_softmax," in code


def _extract_mha_quantizer_params(
    code: str, function_name: str = "quantized_multi_head_attention_forward"
) -> set[str]:
    signature_anchor = f"def {function_name}("
    start = code.index(signature_anchor)
    end = code.index(") ->", start)
    signature = code[start:end]

    quantizers: set[str] = set()
    for line in signature.splitlines():
        stripped = line.strip()
        if stripped.startswith("quantizer_") and ": fastforward.nn.Quantizer" in stripped:
            quantizers.add(stripped.split(":", maxsplit=1)[0])

    return quantizers


@pytest.mark.slow
def test_autoquant_caller_has_distinct_quantizers_for_two_helpers_sharing_op_name() -> None:
    # GIVEN a module where forward calls two different helpers that each perform
    # the same operation type (multiply). Both helpers receive a quantizer with the
    # same base name ("mul"). The caller that invokes both must receive DISTINCT
    # kw-only quantizer parameters — one set per helper — with no name collision.
    # Naming convention: each propagated param is prefixed with the callee's name.

    # WHEN autoquant processes the module
    code = autoquant_with_defaults(
        ExampleModuleTwoHelpersSameQuantizerName(), use_type_inference=False
    )
    code = codeformat_with_defaults(code)

    # THEN each helper is quantized and has a quantizer param in its own signature
    helper_a_quantizers = _extract_mha_quantizer_params(code, "quantized__two_helper_mul_a")
    helper_b_quantizers = _extract_mha_quantizer_params(code, "quantized__two_helper_mul_b")
    assert helper_a_quantizers, "quantized__two_helper_mul_a must have quantizer params"
    assert helper_b_quantizers, "quantized__two_helper_mul_b must have quantizer params"

    # THEN the caller that invokes both helpers has distinct quantizer params for each.
    # If disambiguation failed, the two sets would collapse into one (name collision).
    caller_quantizers = _extract_mha_quantizer_params(code, "quantized__two_helper_caller")
    assert len(caller_quantizers) >= len(helper_a_quantizers) + len(helper_b_quantizers), (
        f"quantized__two_helper_caller must have at least "
        f"{len(helper_a_quantizers)} + {len(helper_b_quantizers)} distinct quantizer params "
        f"(one set per helper call), got: {sorted(caller_quantizers)}"
    )

    # THEN each set is prefixed with its helper's function name, making them disjoint
    a_params_in_caller = {q for q in caller_quantizers if "_two_helper_mul_a_" in q}
    b_params_in_caller = {q for q in caller_quantizers if "_two_helper_mul_b_" in q}
    assert a_params_in_caller, (
        "caller quantizer params must include params prefixed with _two_helper_mul_a"
    )
    assert b_params_in_caller, (
        "caller quantizer params must include params prefixed with _two_helper_mul_b"
    )
    assert a_params_in_caller.isdisjoint(b_params_in_caller), (
        f"quantizer names for both helpers must be disjoint (no collision), "
        f"but found overlap: {a_params_in_caller & b_params_in_caller}"
    )


def test_autoquant_emits_import_for_injected_dispatch_namespace() -> None:
    table = OperatorTable.from_yaml(alias_extensions=optable.STR_ALIASES_EXTENSIONS)
    table.add(
        "issue_b_external_dispatch(x: Tensor, y: Tensor) -> Tensor",
        _issue_b_external_dispatch,
        dispatch_op=_issue_b_dispatch_op,
    )

    original_dispatch_qualified_name = type(
        next(iter(table.get(_issue_b_external_dispatch)))
    ).dispatch_qualified_name

    def _patched_dispatch_qualified_name(self: Any) -> str | None:
        if self.identifier == "issue_b_external_dispatch":
            return "sam3_quantized.interpolate"
        return original_dispatch_qualified_name(self)

    with mock.patch(
        "fastforward._quantops.operator.Operator.dispatch_qualified_name",
        autospec=True,
        side_effect=_patched_dispatch_qualified_name,
    ):
        code = autoquant_with_defaults(
            _IssueBDispatchImportModule(),
            operator_table=table,
            use_type_inference=False,
        )

    code = codeformat_with_defaults(code)

    assert "import sam3_quantized" in code
    assert "sam3_quantized.interpolate" in code


def test_autoquant_cross_file_dispatch_namespace_contract(tmp_path: pathlib.Path) -> None:
    table = OperatorTable.from_yaml(alias_extensions=optable.STR_ALIASES_EXTENSIONS)
    table.add(
        "issue_b_external_dispatch(x: Tensor, y: Tensor) -> Tensor",
        _issue_b_external_dispatch,
        dispatch_op=_issue_b_dispatch_op,
    )

    original_dispatch_qualified_name = type(
        next(iter(table.get(_issue_b_external_dispatch)))
    ).dispatch_qualified_name

    def _patched_dispatch_qualified_name(self: Any) -> str | None:
        if self.identifier == "issue_b_external_dispatch":
            return "sam3_quantized.interpolate"
        return original_dispatch_qualified_name(self)

    with mock.patch(
        "fastforward._quantops.operator.Operator.dispatch_qualified_name",
        autospec=True,
        side_effect=_patched_dispatch_qualified_name,
    ):
        code = autoquant_with_defaults(
            _IssueBDispatchImportModule(),
            operator_table=table,
            use_type_inference=False,
        )

    code = codeformat_with_defaults(code)
    generated = tmp_path / "generated_autoquant_issue_c.py"
    generated.write_text(code, encoding="utf-8")

    stubs = tmp_path / "sam3_quantized.py"
    stubs.write_text(
        "def interpolate(*args, **kwargs):\n    return args[0]\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        ["ruff", "check", str(generated), "--select", "F821"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_module_builder_imports_fallback_for_invalid_symbols(
    snapshot: syrupy.assertion.SnapshotAssertion,
) -> None:
    class _StubStatementBuilder:
        """Minimal statement-builder test double for ModuleBuilder.

        ModuleBuilder.add_function() expects an object with a
        ``required_imports`` attribute and a ``build(...)`` method returning
        a ``libcst.SimpleStatementLine``. This stub keeps the test focused on
        import statement generation (including invalid-symbol fallbacks)
        without depending on full autoquant builder internals.
        """

        def __init__(self, required_imports: tuple[ImportSymbol, ...]) -> None:
            self.required_imports = required_imports

        def build(self, quantizer_refs: object) -> libcst.SimpleStatementLine:
            del quantizer_refs
            statement = libcst.parse_statement("pass")
            assert isinstance(statement, libcst.SimpleStatementLine)
            return statement

    module_builder = pybuilder.ModuleBuilder(origin=None)
    module_builder.add_function(
        cast(
            Any,
            _StubStatementBuilder(
                required_imports=(
                    ImportSymbol(name="torch"),
                    ImportSymbol(
                        name="Siglip2EncoderLayer",
                        module="transformers_modules.Eagle-Block2A-2B-v2.modeling_siglip2",
                    ),
                    ImportSymbol(
                        name="invalid-attr",
                        module="torch.nn.functional",
                        asname="local_name",
                    ),
                )
            ),
        )
    )

    cst_module = libcst.Module([])
    rendered_imports = "\n".join(
        cst_module.code_for_node(stmt) for stmt in module_builder.import_statements()
    )

    assert snapshot == rendered_imports


@pytest.mark.slow
def test_autoquant_emits_code_to_stdout(
    capsys: pytest.CaptureFixture[str], snapshot: syrupy.assertion.SnapshotAssertion
) -> None:
    # GIVEN a torch module to autoquantize
    input = ExampleModule1B()

    # WHEN we autoquantize the example code and write it with the default code writer
    autoquantized = autoquantize(
        module=input,
        code_writer=pybuilder.StdoutWriter(module_name="dummy_name"),
        auto_import=False,
        use_type_inference=False,
    )
    actual_output = autoquantized.code

    # WHEN we inspect the code written to stdout
    captured = capsys.readouterr()

    # THEN the only code written to stdout is the snapshot
    assert snapshot(name="captured") == captured

    # THEN the output matches the snapshot
    assert snapshot(name="actual_output") == actual_output


@pytest.mark.slow
def test_autoquant_writes_to_file(
    tmp_path: pathlib.Path, snapshot: syrupy.assertion.SnapshotAssertion
) -> None:
    # GIVEN a torch module to autoquantize
    input = ExampleModule1B()
    # GIVEN a target output file to write the code to
    out_dir = tmp_path / "some_nested_dir"

    # WHEN we autoquantize the example code and write it to a folder
    autoquantize(
        module=input,
        output_path=out_dir,
        auto_import=False,
        use_type_inference=False,
    )

    # THEN the output directory was created
    assert out_dir.is_dir()
    # THEN `__init__.py` was created in the output directory
    out_file = out_dir / "__init__.py"
    assert out_file.is_file()

    actual_output = out_file.read_text(encoding="utf-8")

    # THEN the file's contents matches the snapshot
    assert snapshot == actual_output

    # WHEN we try to overwrite the existing `__init__.py` file
    with pytest.raises(FileExistsError, match="already exists. Use `force_overwrite=True`"):
        # THEN we get a sensible error
        autoquantize(
            module=input,
            output_path=out_dir,
            auto_import=False,
            use_type_inference=False,
        )

    # WHEN we try to overwrite the `__init__.py` file with force-overwrite
    # THEN no error is thrown
    autoquantize(
        module=input,
        output_path=out_dir,
        force_overwrite=True,
        auto_import=False,
        use_type_inference=False,
    )

    # WHEN we autoquantize the example code and specify to write to a `.py`-file instead
    module_name = "some_file.py"
    output_path = tmp_path / module_name
    autoquantize(
        module=input,
        output_path=output_path,
        auto_import=False,
        use_type_inference=False,
    )
    # THEN we wrote to the `some_file.py`, and not to `__init__.py`
    assert output_path.is_file()


@pytest.mark.slow
@patch.dict("sys.modules")
def test_auto_import(tmp_path: pathlib.Path) -> None:
    # GIVEN a torch module
    input = ExampleModule1B()
    module_name = "ff_quant"
    output_path = tmp_path / module_name

    # WHEN we take note of the original system modules
    sys_modules = dict(sys.modules)

    # WHEN we autoquantize the example code
    # WHEN we add some additional imports (until this feature is added)
    autoquantized_code = autoquantize(
        module=input,
        output_path=output_path,
        auto_import=True,
        use_type_inference=False,
    )

    # THEN the system modules are extended by exactly one module
    new_modules = list(set(sys.modules.values()) - set(sys_modules.values()))
    assert len(new_modules) == 1
    # THEN the new module is ff_quant
    assert new_modules[0] == autoquantized_code.pymodule

    # THEN the module name is `__init__.py` default module name
    assert autoquantized_code.pymodule_name == module_name

    # THEN the generated module is a Python module
    pymodule = autoquantized_code.pymodule
    assert isinstance(pymodule, types.ModuleType)

    # THEN autoquant extends the system modules and we can import ff_quant
    import ff_quant as ff_quant  # type:ignore[import-not-found] # noqa: I001

    # THEN  `import` from the generated module works, too
    from ff_quant import (  # noqa: I001
        QuantizedExampleModule1B as QuantizedModule,
    )

    # THEN the quantized module is available in the generated Python module
    assert hasattr(pymodule, f"Quantized{ExampleModule1B.__name__}")
    assert hasattr(ff_quant, f"Quantized{ExampleModule1B.__name__}")

    # THEN the quantized module can be accessed via dot notation
    modules = [
        pymodule.QuantizedExampleModule1B,
        ff_quant.QuantizedExampleModule1B,
        QuantizedModule,
    ]

    for module in modules:
        # THEN the quantized module has expected static properties
        assert module is not None
        assert module.__name__ == f"Quantized{ExampleModule1B.__name__}"
        assert fastforward.nn.QuantizedModule in module.__bases__

        # THEN the quantized module can be instantiated
        instance = module()

        # THEN the instance has expected properties
        assert list(instance.modules())[-1].__class__ == fastforward.nn.QuantizerStub


def test_find_known_quantized_modules_subclasses_of_subclasses() -> None:
    # _find_known_quantized_modules should find all quantized modules, including
    # modules that don't directly subclass QuantizedModule. An example of this is
    # QuantizedRelu for which ReLU should be discovered.
    assert torch.nn.ReLU in _find_known_quantized_modules()


def _my_func(x: Any, *args: Any, **kwargs: Any) -> Any:
    """Used in autoquant examples."""
    del args, kwargs
    return x


@contextlib.contextmanager
def _my_context(x: Any) -> Iterator[Any]:
    yield x


# Test autoquant with custom Operator table
#
class SimpleLinearModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4, 4))
        self.bias = torch.nn.Parameter(torch.randn(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)


class CustomOpLinearModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight)


class MultiOpModule(torch.nn.Module):
    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x = torch.relu(x)
        x = torch.nn.functional.linear(x, weight)
        return torch.sigmoid(x)


def _custom_linear_dispatch(
    input: torch.Tensor,
    weight: torch.Tensor,
    *,
    quantizer_a: fastforward.nn.Quantizer,
    quantizer_b: fastforward.nn.Quantizer,
    output_quantizer: fastforward.nn.Quantizer,
) -> torch.Tensor:
    # The implementation is irrelevant for this test
    del weight, quantizer_a, quantizer_b, output_quantizer
    return input


@pytest.mark.slow
@pytest.mark.parametrize(
    "module",
    [
        SimpleLinearModule(),
        CustomOpLinearModule(),
        MultiOpModule(),
    ],
    ids=["SimpleLinearModule", "CustomOpLinearModule", "MultiOpModule"],
)
def test_autoquant_with_overloaded_operator_table(
    module: torch.nn.Module, snapshot: syrupy.assertion.SnapshotAssertion
) -> None:
    # GIVEN a simple module that uses torch.nn.functional.linear

    # GIVEN a custom operator table loaded from YAML and an overloaded linear operator
    custom_optable = OperatorTable.from_yaml(alias_extensions=optable.STR_ALIASES_EXTENSIONS)
    custom_schema = "linear(input: Quantized, weight: Quantized) -> Quantized"
    custom_optable.add(
        custom_schema,
        torch.nn.functional.linear,
        dispatch_op=_custom_linear_dispatch,
        intermediate_quantizers=("quantizer_a", "quantizer_b"),
    )

    # GIVEN a default source context
    source_context = default_source_context(use_type_inference=False)

    # WHEN we autoquantize the module with the custom operator table
    autoquant_code = autoquant(
        module=module,
        source_context=source_context,
        operator_table=custom_optable,
    )

    # THEN the generated code must match the snapshot
    actual_code = codeformat_with_defaults(code=autoquant_code)
    assert snapshot == actual_code


# fmt: off
class ExampleModuleMultiline(torch.nn.Module):
    def forward(self, x: torch.Tensor, n: int) -> torch.Tensor:
        y = (
            x
            .view(-1, 3)
            .permute(1, 0)
            .reshape(n, -1)
        )
        return torch.relu(y)
# fmt: on


@pytest.mark.slow
def test_autoquant_multiline_call(snapshot: syrupy.assertion.SnapshotAssertion) -> None:
    quantized = autoquant_with_defaults(ExampleModuleMultiline(), use_type_inference=False)
    formatted = codeformat_with_defaults(quantized)
    assert snapshot == formatted


class ExampleModuleBuiltinCallableAttr(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        op = operator.add
        call = getattr(op, "__call__")
        return cast(torch.Tensor, call(x, y))


@pytest.mark.slow
def test_autoquant_attribute_call_on_builtin_callable_does_not_crash() -> None:
    """Regression for alias scan on objects without ``__dict__``.

    The dependency scanner should not call ``vars(...)`` on builtin callable
    objects reached via attribute calls (e.g. ``op.__call__(...)``).
    """
    _ = autoquant_with_defaults(ExampleModuleBuiltinCallableAttr(), use_type_inference=False)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("target_qualified_name", "expected_function_name"),
    [
        pytest.param(
            __name__,
            "wrapper",
            id="inspect_ismodule_branch",
        ),
        pytest.param(
            f"{__name__}.ExampleModule15",
            "forward",
            id="issubclass_torch_module_branch",
        ),
    ],
)
def test_autoquant_logs_and_continues_on_source_context_error(
    target_qualified_name: str,
    expected_function_name: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # GIVEN a module and a source context that fails for one target qualified name
    class _FailingSourceContext(SourceContext):
        """Wrap a base source context and raise for a configured qualified name."""

        def __init__(self, base: SourceContext, target_qualified_name: str) -> None:
            super().__init__()
            self._base = base
            self._target_qualified_name = target_qualified_name

        @override
        def get(self, qualified_name: str) -> pysource.PySource:
            if qualified_name == self._target_qualified_name:
                raise Exception("boom")
            return self._base.get(qualified_name)

    module = ExampleModule15()
    base_source_context = default_source_context(use_type_inference=False)
    source_context = _FailingSourceContext(base_source_context, target_qualified_name)

    # WHEN autoquant processes the module
    with caplog.at_level(logging.WARNING, logger="fastforward._autoquant.autoquant"):
        generated = autoquant(
            module=module,
            source_context=source_context,
            operator_table=optable.OperatorTable.from_yaml(
                alias_extensions=optable.STR_ALIASES_EXTENSIONS
            ),
        )

    # THEN conversion stays non-fatal and the failure is logged with context
    assert isinstance(generated, str)
    assert any(
        f"Failed to quantize '{expected_function_name}'" in record.message
        for record in caplog.records
    )


@dataclass(frozen=True)
class _FakeCall:
    """Minimal call object for propagation tests.

    `_propagate_quantizers` only requires a `func_ref` attribute on call keys.
    """

    func_ref: Callable[..., Any]
    callsite_id: int


def _build_cyclic_specs() -> dict[Callable[..., Any], _QuantizedFunctionSpec]:
    def f0() -> None:
        return None

    def f1() -> None:
        return None

    def f2() -> None:
        return None

    specs = {
        f0: _QuantizedFunctionSpec(f0),
        f1: _QuantizedFunctionSpec(f1),
        f2: _QuantizedFunctionSpec(f2),
    }

    specs[f0].local_quantizers.append(
        _QuantizerRefTrace(src=(f0,), ref=QuantizerReference(value="q0", refid=0))
    )
    specs[f2].local_quantizers.append(
        _QuantizerRefTrace(src=(f2,), ref=QuantizerReference(value="q2", refid=1))
    )

    # Call graph:
    #   f0 -> f1, f0 -> f0
    #   f1 -> f2, f1 -> f1
    #   f2 -> f1
    specs[f0].calls[cast(Any, _FakeCall(func_ref=f1, callsite_id=0))] = []
    specs[f0].calls[cast(Any, _FakeCall(func_ref=f0, callsite_id=1))] = []
    specs[f1].calls[cast(Any, _FakeCall(func_ref=f2, callsite_id=2))] = []
    specs[f1].calls[cast(Any, _FakeCall(func_ref=f1, callsite_id=3))] = []
    specs[f2].calls[cast(Any, _FakeCall(func_ref=f1, callsite_id=4))] = []

    return specs


def _worker_propagate_cyclic_specs() -> None:
    _propagate_quantizers(_build_cyclic_specs())


@pytest.mark.slow
def test_propagate_quantizers_converges_on_cyclic_trace_growth() -> None:
    """Propagation should converge even when call graph contains cycles."""
    process = mp.Process(target=_worker_propagate_cyclic_specs, daemon=True)
    process.start()
    process.join(timeout=2.0)

    try:
        assert not process.is_alive(), "Propagation did not converge within timeout"
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
