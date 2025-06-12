# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import functools

from typing import Any, Callable, Concatenate, ParamSpec, Protocol

import torch

_P = ParamSpec("_P")


class STEAutogradFunc(torch.autograd.Function):
    """A custom autograd function for Straight-Through Estimator (STE).

    This function allows the forward pass to be non-differentiable while
    providing a gradient for the backward pass.

    Attributes:
        apply: The function to apply STE.
    """

    apply: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    @staticmethod
    def forward(_ctx: Any, _input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Forward pass for the STE function.

        Args:
            _ctx: Context object to store information for backward computation.
            _input: The input tensor.
            output: The output tensor.

        Returns:
            The output tensor.
        """
        return output

    @staticmethod
    def backward(_ctx: Any, output_grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Backward pass for the STE function.

        Args:
            _ctx: Context object.
            output_grad: Gradient of the output.

        Returns:
            Gradient of the input and None for the output.
        """
        return output_grad, None


class STEWrappedFunction(Protocol[_P]):
    """Protocol for a wrapped function using STE.

    This protocol defines the call signature for functions wrapped with STE.
    """

    def __call__(self, __input: torch.Tensor, *args: _P.args, **kwargs: _P.kwargs) -> torch.Tensor:
        """Call the wrapped function.

        Args:
            __input: The input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The output tensor.
        """
        raise NotImplementedError()


def ste(func: Callable[Concatenate[torch.Tensor, _P], torch.Tensor]) -> STEWrappedFunction[_P]:
    """Decorator to apply STE to a function.

    This decorator wraps a function to use STE, ensuring the input and output shapes match.

    Args:
        func: The function to wrap.

    Returns:
        STEWrappedFunction[P]: The wrapped function with STE applied.
    """

    @functools.wraps(func)
    def wrapper(__input: torch.Tensor, *args: _P.args, **kwargs: _P.kwargs) -> torch.Tensor:
        with torch.no_grad():
            output = func(__input, *args, **kwargs)
        if __input.shape != output.shape:
            raise RuntimeError("The input and output shape of an STE function must match.")
        return STEAutogradFunc.apply(__input, output)

    return wrapper


round_ste = ste(torch.round)
