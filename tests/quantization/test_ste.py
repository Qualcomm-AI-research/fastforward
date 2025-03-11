# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import pytest
import torch

from fastforward.quantization import ste


def test_ste_wrapper(_seed_prngs: int) -> None:
    x = torch.randn((2, 2), requires_grad=True)

    def multiply(data: torch.Tensor, scale: float) -> torch.Tensor:
        return data * scale

    ste_multiply = ste.ste(multiply)

    output = ste_multiply(x, 2.25)
    expected_grad = torch.randn(x.shape)
    output.backward(expected_grad)

    torch.testing.assert_close(output, x * 2.25)
    torch.testing.assert_close(x.grad, expected_grad)


def test_ste_ensure_shapes(_seed_prngs: int) -> None:
    x = torch.randn(3)

    # Define a function that changes the shape of the input
    def func(x: torch.Tensor) -> torch.Tensor:
        return x[:2]

    func_ste = ste.ste(func)

    # Check if applying the wrapped function raises a RuntimeError
    with pytest.raises(RuntimeError):
        func_ste(x)
