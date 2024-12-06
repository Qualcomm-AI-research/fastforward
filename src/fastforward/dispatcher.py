# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import bisect
import dataclasses
import enum
import inspect
import typing

from collections import defaultdict
from typing import Any, Callable, Generic, Optional, ParamSpec, Protocol

import torch

P = ParamSpec("P")
KernelType: typing.TypeAlias = Callable[P, torch.Tensor]


class _Predicate(Generic[P], Protocol):
    """
    Protocol for predicate functions used in dispatching.

    A predicate function takes arbitrary arguments and returns a boolean indicating
    whether the predicate is satisfied.
    """

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> bool:
        raise NotImplementedError

    def __invert__(self) -> "_PredicateNot[P]":
        """Return a predicate that is the logical negation of this predicate."""
        return _PredicateNot(self)

    def __and__(self, other: "_Predicate[P]") -> "_PredicateAnd[P]":
        """Return a predicate that is the logical conjunction of this predicate and another."""
        return _PredicateAnd(self, other)

    def __or__(self, other: "_Predicate[P]") -> "_PredicateOr[P]":
        """Return a predicate that is the logical disjunction of this predicate and another."""
        return _PredicateOr(self, other)


class _PredicateNot(_Predicate[P]):
    """
    Predicate that represents the logical negation of another predicate.
    """

    def __init__(self, p: _Predicate[P]) -> None:
        self._p = p

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> bool:
        return not self._p(*args, **kwargs)


class _PredicateAnd(_Predicate[P]):
    """
    Predicate that represents the logical conjunction of multiple predicates.
    """

    def __init__(self, *p: _Predicate[P]) -> None:
        self._p = p

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> bool:
        return all(p(*args, **kwargs) for p in self._p)


class _PredicateOr(_Predicate[P]):
    """
    Predicate that represents the logical disjunction of multiple predicates.
    """

    def __init__(self, *p: _Predicate[P]) -> None:
        self._p = p

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> bool:
        return any(p(*args, **kwargs) for p in self._p)


class Predicate(_Predicate[P]):
    """
    Wrapper for a predicate function.

    Args:
        fn: A callable that takes arbitrary arguments and returns a boolean.
    """

    def __init__(self, fn: Callable[P, bool]) -> None:
        self._fn = fn

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> bool:
        """
        Call wrapped predicate and return result.
        """
        return self._fn(*args, **kwargs)

    def __repr__(self) -> str:
        return f"{self._fn.__name__}: {inspect.signature(self._fn, eval_str=True)}"


class DispatcherPriority(enum.IntEnum):
    """
    Dispatch priority for registered implementations.

    Operator implementations predicates are evaluated in the order
    DEFAULT, FALLBACK, NOT_IMPLEMENTED_FALLBACK.
    """

    DEFAULT = 0
    FALLBACK = 1
    NOT_IMPLEMENTED_FALLBACK = 2


@dataclasses.dataclass
class DispatcherItem(Generic[P]):
    """
    Data class representing an item in the dispatcher.

    Args:
        predicate: A predicate that determines when the kernel should be used.
        fn: The kernel function to be dispatched.
        priority: The priority of the dispatcher item.
    """

    predicate: Predicate[P]
    fn: KernelType[P]
    priority: DispatcherPriority = DispatcherPriority.DEFAULT


_DISPATCHER: dict[str, list[DispatcherItem[Any]]] = defaultdict(list)


class DispatcherRegistrationHook:
    """
    Context manager for registering and unregistering dispatcher items.

    Args:
        op_name: The name of the operation.
        dispatcher_item: The dispatcher item to register.
    """

    def __init__(self, op_name: str, dispatcher_item: DispatcherItem[Any]):
        self._op_name = op_name
        self._dispatcher_item = dispatcher_item

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type, exc_value, traceback):  # type: ignore[no-untyped-def]
        try:
            _DISPATCHER[self._op_name].remove(self._dispatcher_item)
        except ValueError:
            pass


def _register_decorator(
    op_name: str,
    predicate: Optional[Predicate[P]],
    priority: DispatcherPriority = DispatcherPriority.DEFAULT,
) -> Callable[[KernelType[P]], KernelType[P]]:
    """
    Decorator for registering a kernel function with the dispatcher.

    Args:
        op_name: The name of the operation.
        predicate: The predicate that determines when the kernel should be used.
        priority: The priority of the dispatcher item.

    Returns:
        A decorator that registers the kernel function.
    """

    def decorator(kernel: KernelType[P]) -> KernelType[P]:
        register(op_name, predicate, kernel, priority)
        return kernel

    return decorator


def _true_predicate_func(signature_func: Callable[P, Any]) -> Callable[P, bool]:
    """
    Create a predicate that always returns true with the same typed signature as signature_func.

    Args:
        signature_func: The function whose signature is used for the predicate.

    Returns:
        A predicate function that always returns true.
    """

    def true_predicate(*args: P.args, **kwargs: P.kwargs) -> bool:
        return True

    return true_predicate


def _register_functional(
    op_name: str,
    predicate: Optional[Predicate[P]],
    kernel: KernelType[P],
    priority: DispatcherPriority = DispatcherPriority.DEFAULT,
) -> DispatcherRegistrationHook:
    """
    Register a kernel function with the dispatcher.

    Args:
        op_name: The name of the operation.
        predicate: The predicate that determines when the kernel should be used.
        kernel: The kernel function to be dispatched.
        priority: The priority of the dispatcher item.

    Returns:
        A `DispatcherRegistrationHook` for managing the registration.
    """
    predicate = predicate or Predicate(_true_predicate_func(kernel))
    dispatcher_item = DispatcherItem(predicate, kernel, priority)
    op_impls = _DISPATCHER[op_name]
    insert_idx = bisect.bisect_left(op_impls, priority, key=lambda item: item.priority)
    op_impls.insert(insert_idx, dispatcher_item)
    return DispatcherRegistrationHook(op_name, dispatcher_item)


@typing.overload  # Usage as a function or context manager
def register(
    op_name: str,
    predicate: Optional[Predicate[P]],
    kernel: KernelType[P],
    priority: DispatcherPriority = DispatcherPriority.DEFAULT,
) -> DispatcherRegistrationHook: ...


@typing.overload  # Usage as a decorator
def register(
    op_name: str,
    predicate: Optional[Predicate[P]],
    kernel: None = None,
    priority: DispatcherPriority = DispatcherPriority.DEFAULT,
) -> Callable[[KernelType[P]], KernelType[P]]: ...


def register(
    op_name: str,
    predicate: Optional[Predicate[P]],
    kernel: Optional[KernelType[P]] = None,
    priority: DispatcherPriority = DispatcherPriority.DEFAULT,
) -> Callable[[KernelType[P]], KernelType[P]] | DispatcherRegistrationHook:
    """
    Register a new implementation for an operator.

    This which will be used when the predicate evaluates to True. Might raise
    an exception if there is already a kernel with the same priority.

    register can be used standalone:

       > register(op_name, predicate, kernel)

    as a standalone context manager:

       > @register(op_name, predicate)
       > def my_kernel(...):
       >    ...

    or using a with statement:

        > with register(op_name, predicate, kernel):
        >     ...

    In the latter case, the registered implementation is removed at the end of the
    with block.
    """
    if kernel is None:
        return _register_decorator(op_name, predicate, priority)
    else:
        return _register_functional(op_name, predicate, kernel, priority)


def dispatch(op_name: str, *args: P.args, **kwargs: P.kwargs) -> Optional[KernelType[P]]:
    """
    Returns the latest registered kernel whose predicate evaluates to True.

    Args:
        op_name: The name of the operation.
        *args: Positional arguments to pass to the predicate.
        **kwargs: Keyword arguments to pass to the predicate.

    Returns:
        The kernel function if a matching predicate is found, otherwise None.
    """
    dispatcher_items = _DISPATCHER.get(op_name, [])
    for item in dispatcher_items:
        if item.predicate(*args, **kwargs):
            return item.fn
    return None
