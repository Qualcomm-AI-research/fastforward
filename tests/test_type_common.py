# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#
import sys

import pytest

from fastforward.type_common import MethodType, method_type


class TestClass:
    def instance_method(self) -> None:
        pass

    @classmethod
    def class_method(cls) -> None:
        pass

    @staticmethod
    def static_method() -> None:
        pass


def module_function() -> None:
    pass


def test_instance_method() -> None:
    result = method_type(TestClass, "instance_method")
    assert result == MethodType.METHOD


def test_class_method() -> None:
    result = method_type(TestClass, "class_method")
    assert result == MethodType.CLASS_METHOD


def test_static_method() -> None:
    result = method_type(TestClass, "static_method")
    assert result == MethodType.STATIC_METHOD


def test_no_method_nonexistent() -> None:
    result = method_type(TestClass, "nonexistent_method")
    assert result == MethodType.NO_METHOD


def test_module_with_function() -> None:
    current_module = sys.modules[__name__]
    result = method_type(current_module, "module_function")
    assert result == MethodType.NO_METHOD


def test_module_with_nonexistent_function() -> None:
    current_module = sys.modules[__name__]
    result = method_type(current_module, "nonexistent_function")
    assert result == MethodType.NO_METHOD


def test_invalid_cls_or_module_type() -> None:
    with pytest.raises(ValueError, match="'cls_or_module' must be a module or class"):
        method_type("not_a_class", "method_name")  # type: ignore[arg-type]


def test_invalid_cls_or_module_instance() -> None:
    instance = TestClass()
    with pytest.raises(ValueError, match="'cls_or_module' must be a module or class"):
        method_type(instance, "instance_method")  # type: ignore[arg-type]


def test_method_with_none_value() -> None:
    # Test case where __dict__ contains None for the method name
    mock_class = type("MockClass", (), {"test_attr": None})
    result = method_type(mock_class, "test_attr")
    assert result == MethodType.NO_METHOD
