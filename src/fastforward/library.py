# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import contextlib
import functools

from typing import Any, Callable, TypeVar

import torch

from typing_extensions import ParamSpec

from fastforward import ops
from fastforward.flags import get_compiled_quant_funcs, get_export_mode

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


_REGISTERED_TORCH_OPS: dict[str, Any] = {}


def _is_profiler_active() -> bool:
    """Return True if a torch profiler is active in the current context.

    Return `torch.autograd.profiler._is_profiler_enabled` or False if that property
    does not exist in current pytorch version.
    """
    try:
        return torch.autograd.profiler._is_profiler_enabled
    except Exception:  # noqa: BLE001
        return False


def _is_exporting() -> bool:
    """Return True if `torch.export` is tracing.

    `torch.compiler.is_exporting()` cannot be used: up to torch 2.11, dynamo
    constant-folds it to True under any `torch.compile`, not only during export.
    The flag it reads is only set by `torch.export` (strict and non-strict), and
    dynamo evaluates the attribute access correctly in all versions.
    """
    return torch.compiler._is_exporting_flag


def _compiled(func: Callable[..., Any], fullgraph: bool) -> Callable[..., Any]:
    return conditional_compile_fullgraph(func) if fullgraph else conditional_compile(func)


def custom_quant_op(
    name: str,
    *,
    conditional_compile: bool = True,
    fullgraph: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Wraps a function into a custom fastforward and a custom torch operator.

    The function is registered both as:
        - A fastforward op at `fastforward.ops.<name>`.
        - A torch op at `fastforward::<name>` (`torch.ops.fastforward.<name>`)
          (see torch.library.custom_op), which `torch.compile` treats as an
          opaque graph node.

    Calling the fastforward op will fire the opaque torch op (in Export mode) or
    the (maybe-compiled) plain python function (in Compile and Eager mode).

    Library code should always call the returned callable or the
    `fastforward.ops.<name>`, and *never* `torch.ops.fastforward.<name>` directly,
    which is always opaque.

    The callable (and the fastforward op) dispatches at call time:

    - When exporting (`export_mode` enabled or `torch.export` tracing), the
      registered `torch.ops.fastforward.<op>` runs, so it appears as an opaque
      node in the exported graph, never tracing into the python implementation,
      and using the registered fake-impl for shapes and dtypes.
      In this case quant-ops are opaque kernels that cannot be fused with
      neighbour ops: AOTAutograd and Inductor treat the node as
      an extern kernel call.
      The quant-op can still be compiled if conditional_compile=True and
      when the compiled_quant_funcs flag is enabled (like in eager mode).

    - Under `torch.compile`, the plain python function runs so its ops stay
      visible to the compiler (and can be fused with neighbour ops by Inductor),
      reducing memory movement and obtaining great performances in most cases.

    - Otherwise, the (possibly-compiled) plain python quantization function is
      run in eager mode. When conditional_compile=True and compiled_quant_funcs
      flag is enabled, the called function will be the torch-compiled version.

    The ops are not differentiable when called directly: the plain paths run under
    `torch.no_grad()`, so their outputs do not require grad. Gradients are provided
    by the `torch.autograd.Function` classes that wrap them.

    In eager mode, if an active torch profiler is detected, the call is recorded as
    a `fastforward::<name>` range in `torch.profiler`.

    Args:
        name: Name of the operator, without the `fastforward::` namespace. Must be a
            valid ASCII identifier.
        conditional_compile: If True, the plain function and the registered op
            are compiled when `compiled_quant_funcs` is enabled. If False, they
            are never compiled.
        fullgraph: If True, compile with `fullgraph=True`. Requires
            `conditional_compile=True`.

    Raises:
        ValueError: If the name is invalid, the signature of the function cannot
            be converted to a torch schema, or an op with the same name is already
            registered. A rejected op leaves all registered ops untouched. To replace
            an op, first remove it with `deregister_quant_op`.
    """
    if not (name.isascii() and name.isidentifier()):
        msg = f"Invalid quant op name '{name}': torch only accepts ASCII identifiers."
        raise ValueError(msg)

    if fullgraph and not conditional_compile:
        msg = "`fullgraph=True` requires `conditional_compile=True`."
        raise ValueError(msg)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        maybe_compiled = _compiled(func, fullgraph) if conditional_compile else func
        profiler_tag = "fastforward::" + name

        @functools.wraps(func)
        def dispatch(*args: Any, **kwargs: Any) -> Any:
            if _is_exporting():
                # Export tracing: call the registered opaque `torch.ops.fastforward.<op>`
                # (possibly compiled) so that the exported graph records it as a single
                # node. No `no_grad` here: it would wrap the node in a
                # `wrap_with_set_grad_enabled` submodule in the exported graph.
                return _REGISTERED_TORCH_OPS[name](*args, **kwargs)

            elif get_export_mode():
                # Export mode, called eagerly: same opaque op, but under `no_grad` to
                # match the other paths (the registered op has no autograd formula).
                with torch.no_grad():
                    return _REGISTERED_TORCH_OPS[name](*args, **kwargs)

            elif torch.compiler.is_compiling():
                # Compiling: we call the plain python function (never compiled)
                # so that Inductor can look through and fuse when compiling
                # the model. `compiled_quant_funcs` is intentionally ignored here.
                with torch.no_grad():
                    return func(*args, **kwargs)

            else:
                # Eager: not exporting nor compiling, the plain python function
                # (possibly-compiled) is called eagerly. The profiler range
                # gives the op a single entry named like the registered op.
                maybe_record_function: contextlib.AbstractContextManager[Any] = (
                    contextlib.nullcontext()
                )
                if _is_profiler_active():
                    maybe_record_function = torch.profiler.record_function(profiler_tag)
                with torch.no_grad(), maybe_record_function:
                    return maybe_compiled(*args, **kwargs)

        # Everything that can reject the op runs before the torch op library is modified,
        # as torch silently replaces an op registered under the same name (which then
        # loses its fake impl): the schema is inferred (fails on a bad signature) and
        # `fastforward.ops` rejects an already registered name.
        torch.library.infer_schema(maybe_compiled, mutates_args=())
        ops._register_op(name, dispatch)
        _REGISTERED_TORCH_OPS[name] = torch.library.custom_op(
            "fastforward::" + name, mutates_args=()
        )(maybe_compiled)
        return dispatch

    return decorator


def deregister_quant_op(name: str, *, missing_ok: bool = False) -> None:
    """Remove an op registered with `custom_quant_op`.

    Removes `fastforward.ops.<name>`, the torch op `torch.ops.fastforward.<name>`
    and the fake impl registered for it with `register_quant_fake`. To replace an
    op, deregister it and register the new op (and its fake impl) afterwards.

    The callable returned by `custom_quant_op` for the removed op must not be used
    anymore.

    Args:
        name: Name of the op, as passed to `custom_quant_op`.
        missing_ok: If True, do nothing when no op is registered under `name`
            instead of raising.

    Raises:
        ValueError: If no op with this name is registered and `missing_ok` is False.
    """
    ops._deregister_op(name, missing_ok=missing_ok)
    torch_op = _REGISTERED_TORCH_OPS.pop(name, None)
    if torch_op is not None:
        # Destroying the library also removes the fake impl registered for the op.
        torch_op._lib._destroy()


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
        return torch.library.triton_op("fastforward::" + name, func, mutates_args=())

    return decorator


wrap_triton = torch.library.wrap_triton


def register_quant_fake(name: str) -> Callable[[Callable[..., Any]], Any]:
    """Register a FakeTensor implementation ("fake impl") for this operator.

    Also sometimes known as a "meta kernel", "abstract impl".

    The op must already be registered with `custom_quant_op`. The fake impl is
    removed together with the op by `deregister_quant_op`.

    See torch.library.register_fake documentation.
    """

    def decorator(func: Callable[..., Any]) -> Any:
        return torch.library.register_fake("fastforward::" + name)(func)

    return decorator
