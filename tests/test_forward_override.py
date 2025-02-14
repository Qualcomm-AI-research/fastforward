# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import collections
import unittest.mock

from typing import Any, MutableMapping

from fastforward import forward_override as override


def test_function_override() -> None:
    def to_override_function(a: int, b: int, **kwargs: Any) -> int:
        return a + b

    def override1(ctx: Any, fn: Any, args: Any, kwargs: Any) -> Any:
        return fn(*args, **kwargs) + ctx["c"]

    def override2(ctx: Any, fn: Any, args: Any, kwargs: Any) -> Any:
        return fn(*args, **kwargs) * 2

    def override_noop(ctx: Any, fn: Any, args: Any, kwargs: Any) -> str:
        return "nothing"

    mock_override1 = unittest.mock.Mock(override1, wraps=override1)
    mock_override2 = unittest.mock.Mock(override2, wraps=override2)
    mock_override_noop = unittest.mock.Mock(override_noop, wraps=override_noop)

    # Test if overrides are called and 'correct' overridden output is returned
    context_object = {"c": 10}
    override_map = {
        2: mock_override2,
        1: mock_override1,
    }
    wrapped_fn = override.apply_overrides(context_object, to_override_function, override_map)

    assert to_override_function(2, 3) == 5
    assert wrapped_fn(2, 3, extra_kwarg=123) == 30
    mock_override1.assert_called_once_with(
        context_object, unittest.mock.ANY, (2, 3), {"extra_kwarg": 123}
    )
    mock_override2.assert_called_once_with(
        context_object, unittest.mock.ANY, (2, 3), {"extra_kwarg": 123}
    )

    mock_override1.reset_mock()
    mock_override2.reset_mock()

    # Test if overrides are called, except override1 as there is a higher
    # precedence override (override_noop) that doesn't call up the override
    # stack.
    override_map = {
        3: mock_override2,
        2: mock_override_noop,
        1: mock_override1,
    }
    wrapped_fn = override.apply_overrides(context_object, to_override_function, override_map)

    assert wrapped_fn(2, 3) == "nothingnothing"
    mock_override1.assert_not_called()
    mock_override2.assert_called_once_with(context_object, unittest.mock.ANY, (2, 3), {})
    mock_override_noop.assert_called_once_with(context_object, unittest.mock.ANY, (2, 3), {})

    # Test if there is a fast path when there are no overrides.
    wrapped_fn = override.apply_overrides(context_object, to_override_function, {})
    assert wrapped_fn == to_override_function


def test_override_handle() -> None:
    def to_override_function(a: Any, b: Any) -> Any:
        return a + b

    def override1(ctx: Any, fn: Any, args: Any, kwargs: Any) -> Any:
        return fn(*args, **kwargs) + ctx["c"]

    mock_override1 = unittest.mock.Mock(override1, wraps=override1)

    override_map: MutableMapping[int, override.OverrideFn[int]] = collections.OrderedDict()
    handle = override.OverrideHandle(override_map=override_map)
    override_map[handle.handle_id] = mock_override1
    context_object = {"c": 5}

    # Test if override works, then remove through handle and test if override is not used
    wrapped_fn = override.apply_overrides(context_object, to_override_function, override_map)
    assert wrapped_fn(2, 3) == 10
    mock_override1.assert_called_once_with(context_object, unittest.mock.ANY, (2, 3), {})

    mock_override1.reset_mock()
    handle.remove()
    assert wrapped_fn(2, 3) == 5
    mock_override1.assert_not_called()
