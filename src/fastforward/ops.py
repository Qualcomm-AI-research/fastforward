# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""FastForward quantization ops.

Every op registered with `fastforward.library.custom_quant_op` is available here as
`fastforward.ops.<name>`, next to the registered torch op `torch.ops.fastforward.<name>`.
Prefer `fastforward.ops.<name>`: it dispatches on the `export_mode` flag, `torch.export` and
`torch.compile` tracing, and the `compiled_quant_funcs` flag, while `torch.ops.fastforward.<name>`
is always an opaque call to the compiler.
"""

from typing import Any, Callable

_OPS: dict[str, Callable[..., Any]] = {}


def _register_op(name: str, op: Callable[..., Any]) -> None:
    """Make `op` available as `fastforward.ops.<name>`.

    Ops are never replaced implicitly: to replace an op, deregister it first.

    Args:
        name: Name of the op.
        op: The callable to register.

    Raises:
        ValueError: If an op with the same name is already registered.
    """
    if name in _OPS:
        msg = f"An op named '{name}' is already registered in `fastforward.ops`."
        raise ValueError(msg)
    _OPS[name] = op


def _deregister_op(name: str, *, missing_ok: bool = False) -> None:
    """Remove `fastforward.ops.<name>`.

    This only removes the entry in `fastforward.ops`. Ops registered with
    `fastforward.library.custom_quant_op` should be removed with
    `fastforward.library.deregister_quant_op`, which also removes the torch op.

    Args:
        name: Name of the op.
        missing_ok: If True, do nothing when no op is registered under `name`
            instead of raising.

    Raises:
        ValueError: If no op with this name is registered and `missing_ok` is False.
    """
    if name not in _OPS:
        if missing_ok:
            return
        msg = f"No op named '{name}' is registered in `fastforward.ops`."
        raise ValueError(msg)
    del _OPS[name]


def __getattr__(name: str) -> Callable[..., Any]:
    try:
        return _OPS[name]
    except KeyError:
        msg = f"module 'fastforward.ops' has no op '{name}'"
        raise AttributeError(msg) from None


def __dir__() -> list[str]:
    return sorted(_OPS)
