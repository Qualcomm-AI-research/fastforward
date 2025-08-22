# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Callable

import fastforward as ff
import pytest


def autoquant_case(
    input: str,
    expected: str,
    as_module: bool = False,
    mark_slow: bool = True,
    use_type_inference: bool = False,
) -> Callable[[], None]:
    def test_function() -> None:
        ff.testing.autoquant.assert_autoquantize_result(
            input, expected, as_module, use_type_inference=use_type_inference
        )

    if mark_slow:
        test_function = pytest.mark.slow(test_function)

    return test_function


test_raise = autoquant_case(
    # Autoquant for `raise` behaves similarly to `return`. I.e., no new
    # quantizer should be introduced after 'resetting the quantization status'.
    # In the case below, the assignment to `x` in the if branch is not 'live'
    # after the raise statement. Hence, no quantization calls must be
    # introduced due to the multiplication in the final return statement.
    input="""
    def my_function(self, x: torch.Tensor) -> torch.Tensor:
        if True:
            x = some_function(x) # reset quantization status
            raise ValueError("msg")
        return x * x
    """,
    expected="""
    def my_function(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quantizer_x(x)
        if True:
            x = some_function(x) # reset quantization status
            raise ValueError("msg")
        return fastforward.nn.functional.mul(x, x, output_quantizer=self.quantizer_mul)
    """,
)


test_autoquant_ignore_annotations = autoquant_case(
    # Autoquant must ignore all annotations and leave them unchanged.
    input="""
    def my_func_with_annotations(self, x: int | float, y: float | int) -> str | bool:
        z: str | bool = y | x
        return z
    """,
    expected="""
    def my_func_with_annotations(self, x: int | float, y: float | int) -> str | bool:
        z: str | bool = y | x
        return z
    """,
    use_type_inference=True,
)

test_autoquant_skip_isolation_for_if_expr = autoquant_case(
    # Autoquant must not isolate expressions in the `body` or `orelse` of an
    # if-expression since isolation would lead to unnecessary compute (i.e.,
    # both branches are evaluated).
    input="""
    def my_func_with_annotations(self, x: Tensor, y: Tensor, flag: bool) -> Tensor:
        out = x + y if flag else y - x
        return out
    """,
    expected="""
    def my_func_with_annotations(self, x: Tensor, y: Tensor, flag: bool) -> Tensor:
        x = self.quantizer_x(x)
        y = self.quantizer_y(y)
        out = (
            fastforward.nn.functional.add(x, y, output_quantizer=self.quantizer_add)
            if flag
            else fastforward.nn.functional.sub(y, x, output_quantizer=self.quantizer_sub)
        )
        return out
    """,
)
