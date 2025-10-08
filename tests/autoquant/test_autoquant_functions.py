# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import fastforward as ff
import pytest
import syrupy


def _autoquantize_str(input: str, as_module: bool = False, use_type_inference: bool = False) -> str:
    autoquant_result = ff.testing.autoquant.autoquantize_str(
        input, as_module=as_module, as_cst=False, use_type_inference=use_type_inference
    )
    assert isinstance(autoquant_result, str)
    return autoquant_result


@pytest.mark.slow
def test_raise(snapshot: syrupy.assertion.SnapshotAssertion) -> None:
    # Autoquant for `raise` behaves similarly to `return`. I.e., no new
    # quantizer should be introduced after 'resetting the quantization status'.
    # In the case below, the assignment to `x` in the if branch is not 'live'
    # after the raise statement. Hence, no quantization calls must be
    # introduced due to the multiplication in the final return statement.
    input = """
    def my_function(self, x: torch.Tensor) -> torch.Tensor:
        if True:
            x = some_function(x) # reset quantization status
            raise ValueError("msg")
        return x * x
    """
    result = _autoquantize_str(input)
    assert snapshot == result


@pytest.mark.slow
def test_autoquant_ignore_annotations(snapshot: syrupy.assertion.SnapshotAssertion) -> None:
    # Autoquant must ignore all annotations and leave them unchanged.
    input = """
    def my_func_with_annotations(self, x: int | float, y: float | int) -> str | bool:
        z: str | bool = y | x
        return z
    """
    result = _autoquantize_str(input, use_type_inference=True)
    assert snapshot == result


@pytest.mark.slow
def test_autoquant_skip_isolation_for_if_expr(snapshot: syrupy.assertion.SnapshotAssertion) -> None:
    # Autoquant must not isolate expressions in the `body` or `orelse` of an
    # if-expression since isolation would lead to unnecessary compute (i.e.,
    # both branches are evaluated).
    input = """
    def my_func_with_annotations(self, x: Tensor, y: Tensor, flag: bool) -> Tensor:
        out = x + y if flag else y - x
        return out
    """
    result = _autoquantize_str(input)
    assert snapshot == result
