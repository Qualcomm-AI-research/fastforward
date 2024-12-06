# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import dataclasses
import functools

from typing import Callable, ContextManager, ParamSpec, TypeAlias, TypeVar

_T = TypeVar("_T")
_P = ParamSpec("_P")


_FlagSetter: TypeAlias = Callable[[bool], ContextManager[None]]
_FlagGetter: TypeAlias = Callable[[], bool]
_FlagContext: TypeAlias = Callable[[bool], ContextManager[None]]


@dataclasses.dataclass(frozen=True)
class _FlagMethods:
    setter: _FlagSetter
    getter: _FlagGetter
    context: _FlagContext


_FLAGS: dict[str, bool] = {}


def _context_flag(flag_name: str, init_value: bool) -> _FlagMethods:
    if flag_name in _FLAGS:
        raise ValueError(f"Flag '{flag_name}' already exists")
    _FLAGS[flag_name] = init_value

    def _setter(value: bool) -> None:
        _FLAGS[flag_name] = value

    def getter() -> bool:
        return _FLAGS[flag_name]

    class _ResetToTrue:
        def __enter__(self) -> None:
            pass

        def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore[no-untyped-def]
            _setter(True)

    class _ResetToFalse:
        def __enter__(self) -> None:
            pass

        def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore[no-untyped-def]
            _setter(False)

    # context could have been implemented as a contextmanager, however, as of
    # PyTorch 2.4, Dynamo does not support contextmanager and has limited
    # support for user defined classes that implement __enter__ and __exit__.
    # In particular, a user defined class that has a custom __init__ results in
    # Dynamo errors. For this reason, we create two separate ContextManager:
    # _ResetToTrue and _ResetToFalse that will reset the flag value at the end
    # of a with block. Once more support in dynamo lands this function can
    # rewritten to a contextmanager.
    def context(value: bool) -> ContextManager[None]:
        current_value = getter()
        _setter(value)
        if current_value:
            return _ResetToTrue()
        else:
            return _ResetToFalse()

    def setter(value: bool) -> ContextManager[None]:
        return context(value)

    setter.__name__ = setter.__qualname__ = f"set_{flag_name}"
    getter.__name__ = getter.__qualname__ = f"get_{flag_name}"
    context.__name__ = context.__qualname__ = flag_name
    _ResetToTrue.__name__ = _ResetToTrue.__qualname__ = f"{flag_name}_ResetToTrue"
    _ResetToFalse.__name__ = _ResetToFalse.__qualname__ = f"{flag_name}_ResetToFalse"

    return _FlagMethods(setter, getter, context)  # noqa: F821


def context(flag: _FlagContext, value: bool) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """
    Decorator to execute a function in a given context.

    Here context refers to a value of a flag. Using this decorator, the flag value is
    set to `value` before the decorated function is invoked and reset to it's state
    before invocations after the function completes.

    Args:
        flag: The flag to set
        value: The value to set the value to
    """

    def decorator(func: Callable[_P, _T]) -> Callable[_P, _T]:
        @functools.wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            with flag(value):
                return func(*args, **kwargs)

        return wrapper

    return decorator


_strict_quant = _context_flag("strict_quantization", True)
set_strict_quantization = _strict_quant.setter
get_strict_quantization = _strict_quant.getter
strict_quantization = _strict_quant.context
del _strict_quant

_export_mode = _context_flag("export_mode", False)
set_export_mode = _export_mode.setter
get_export_mode = _export_mode.getter
export_mode = _export_mode.context
del _export_mode

_compiled_quant_funcs = _context_flag("compiled_quant_funcs", False)
set_compiled_quant_funcs = _compiled_quant_funcs.setter
get_compiled_quant_funcs = _compiled_quant_funcs.getter
compiled_quant_funcs = _compiled_quant_funcs.context
del _compiled_quant_funcs

# remove _context_flag as we want to collect created flags in this file and not
# add them left and right.
del _context_flag
del _FlagMethods
