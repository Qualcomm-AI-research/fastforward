# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Callable

import fastforward as ff
import pytest


def autoquant_case(
    input: str, expected: str, as_module: bool = False, mark_slow: bool = True
) -> Callable[[], None]:
    def test_function() -> None:
        ff.testing.autoquant.assert_autoquantize_result(input, expected, as_module)

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
        x = self.quantizer_x(x)
        y = self.quantizer_y(y)
        z: str | bool = fastforward.nn.functional.bitwise_or(
            y, x, output_quantizer=self.quantizer_bitwise_or
        )
        return z
    """,
)
