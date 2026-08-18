# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import functools

from typing import Any, Callable, TypeVar

import torch

from typing_extensions import ParamSpec

from fastforward.flags import get_compiled_quant_funcs

_RetT = TypeVar("_RetT")  # _T
_InputT = ParamSpec("_InputT")  # _P


def conditional_compile(func: Callable[_InputT, _RetT]) -> Callable[_InputT, _RetT]:
    """Compile this operator if compiled_quant_funcs flag is enabled."""
    compiled_func: Callable[_InputT, _RetT] | None = None

    @functools.wraps(func)
    def wrapper(*args: _InputT.args, **kwargs: _InputT.kwargs) -> _RetT:
        nonlocal compiled_func
        if "num_bits" in kwargs:
            kwargs["num_bits"] = float(kwargs["num_bits"])  # type: ignore[arg-type]

        if get_compiled_quant_funcs():
            if compiled_func is None:
                # `recompile_limit` is accepted by `torch.compile` but missing from both of
                # its `@overload` signatures, so the call does not type check.
                compiled_func = torch.compile(func, recompile_limit=1024)  # type: ignore[call-overload]
            return compiled_func(*args, **kwargs)
        return func(*args, **kwargs)

    return wrapper


def conditional_compile_fullgraph(func: Callable[_InputT, _RetT]) -> Callable[_InputT, _RetT]:
    """Compile this operator with `fullgraph=True` if `compiled_quant_funcs` is enabled.

    `torch.compile(func)` (used by `conditional_compile`) allows Dynamo to fall
    back to eager on parts of `func` it cannot trace, leaving some of that
    region's dispatch overhead in place. `fullgraph=True` instead requires the
    whole function to trace, which is what lets the Triton kernels this is used
    for fuse with whatever their caller does with the result — the reference
    broadcast ops do not consistently support `fullgraph=True` (their fake
    implementations are not always called the way it requires), so they keep
    using `conditional_compile` instead.
    """
    compiled_func: Callable[_InputT, _RetT] | None = None

    @functools.wraps(func)
    def wrapper(*args: _InputT.args, **kwargs: _InputT.kwargs) -> _RetT:
        nonlocal compiled_func
        if "num_bits" in kwargs:
            kwargs["num_bits"] = float(kwargs["num_bits"])  # type: ignore[arg-type]

        if get_compiled_quant_funcs():
            if compiled_func is None:
                compiled_func = torch.compile(func, fullgraph=True)
            return compiled_func(*args, **kwargs)
        return func(*args, **kwargs)

    return wrapper


def custom_quant_op(name: str) -> Callable[[Callable[..., Any]], Any]:
    """Wraps a function into custom operator.

    See torch.library.custom_op documentation.
    """

    def decorator(func: Callable[..., Any]) -> Any:
        return torch.library.custom_op("fastforward::" + name, mutates_args=())(func)  # type:ignore[attr-defined, unused-ignore]

    return decorator


def triton_quant_op(name: str) -> Callable[[Callable[..., Any]], Any]:
    """Wraps a function backed by Triton kernel(s) into a custom operator.

    Unlike `custom_op`, the operator's implementation stays visible to
    `torch.compile`/`torch.export`, so calls to it can be fused with
    surrounding operations instead of being treated as an opaque boundary. Any
    Triton kernel invoked inside the wrapped function must itself be wrapped
    with `wrap_triton` for this visibility to apply.

    See torch.library.triton_op documentation.
    """

    def decorator(func: Callable[..., Any]) -> Any:
        return torch.library.triton_op("fastforward::" + name, func, mutates_args=())  # type:ignore[attr-defined, unused-ignore]

    return decorator


wrap_triton = torch.library.wrap_triton


def register_quant_fake(name: str) -> Callable[[Callable[..., Any]], Any]:
    """Register a FakeTensor implementation ("fake impl") for this operator.

    Also sometimes known as a "meta kernel", "abstract impl".

    See torch.library.register_fake documentation.
    """

    def decorator(func: Callable[..., Any]) -> Any:
        return torch.library.register_fake("fastforward::" + name)(func)  # type:ignore[attr-defined, unused-ignore]

    return decorator
