# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import weakref

from typing import Any, Callable, Generic, Mapping, MutableMapping, Protocol, TypeVar

from typing_extensions import Self

T = TypeVar("T")


class OverrideFn(Protocol[T]):
    """Protocol for override functions.

    Override functions can be registered to override a quantizer. These acts as
    dynamic decorators that can be applied an removed at runtime.
    """

    def __call__(
        self,
        __context: Any,
        __overridden_fn: Callable[..., T],
        __args: tuple[Any, ...],
        __kwargs: dict[str, Any],
    ) -> T:
        """Generic call signature.

        Args:
            __context: The quantizer that is overriden by this override function
            __overriden_fn: The function that is overriden, this may the
                forward method of a quantizer, or a reference to another override.
                This should be called instead of the forward method on __context
                in order for an override to cooperate with other overrides.
            __args: The arguments passed to the quantizer.
            __kwargs: The keyword arguments passed to the quantizer.
        """
        raise NotImplementedError


class OverrideHandle:
    """Handle object which that can be used to remove a function override.

    Args:
        override_map: Mapping that stores
            overrides indexed by override id.
    """

    handle_id: int
    global_handles: int = 0

    def __init__(self, override_map: MutableMapping[int, OverrideFn[Any]]) -> None:
        self.override_map = weakref.ref(override_map)
        self.handle_id = OverrideHandle.global_handles
        OverrideHandle.global_handles += 1

    def remove(self) -> OverrideFn[Any] | None:
        """Remove override associated with this handle.

        Returns:
            (Callable) the override function associated with this handle
        """
        override_map = self.override_map()
        if override_map is not None and self.handle_id in override_map:
            return override_map.pop(self.handle_id)
        return None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, type, value, traceback) -> None:  # type: ignore[no-untyped-def]
        self.remove()


class _WrappedOverriddenFn(Generic[T]):
    def __init__(
        self,
        context: Any,
        overridden_fn: Callable[..., T],
        override_map: Mapping[int, OverrideFn[T]],
    ) -> None:
        self.context = context
        self.overridden_fn = overridden_fn
        _, override_stack = zip(*sorted(override_map.items()))
        self.override_stack: list[OverrideFn[T]] = list(override_stack)

    def __call__(self, *args: Any, **kwargs: Any) -> T:
        if not self.override_stack:
            return self.overridden_fn(*args, **kwargs)
        else:
            override_fn = self.override_stack.pop()
            return override_fn(self.context, self, args, kwargs)


def apply_overrides(
    context: Any,
    overridden_fn: Callable[..., T],
    override_map: Mapping[int, OverrideFn[T]],
) -> _WrappedOverriddenFn[T] | Callable[..., T]:
    """Apply overrides in `override_map` to `overridden_fn`.

    This returns a callable that, when applicable, calls functions overrides.
    These act similarly to decorators.

    The overrides are applied in descending order of the key in `override_map`

    Args:
        context: An object related to overridden fn that may store
            contextual data that is passed into the function overrides.
        overridden_fn: The function to override.
        override_map: A mapping from int to overides. The overrides are applied
            in order of the key.

    Returns:
        (Callable[..., T]) wrapped overridden_fn, or overridden_fn if override_map
        is empty.
    """
    if not override_map:
        return overridden_fn
    wrapped_fn = _WrappedOverriddenFn(
        context=context,
        overridden_fn=overridden_fn,
        override_map=override_map,
    )
    return wrapped_fn
