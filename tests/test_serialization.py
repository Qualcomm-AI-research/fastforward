# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Any

from fastforward.serialization import _has_custom_method, yamlable
from typing_extensions import Self


def test_has_custom_method() -> None:
    """Test detection of default methods."""

    # GIVEN: A class with default `__new__` and custom __init__ methods
    class DefaultClass:
        def __init__(self) -> None:
            pass

    # WHEN: Checking if the class has custom methods
    # THEN: Should return False for default methods
    assert not _has_custom_method(DefaultClass, "__new__")
    assert _has_custom_method(DefaultClass, "__init__")


def test_has_custom_new_method() -> None:
    """Test detection of custom __new__ method."""

    # GIVEN: Classes with default and custom __new__ methods
    class DefaultNew:
        pass

    class CustomNew:
        def __new__(cls) -> Self:
            return super().__new__(cls)

    # WHEN: Checking for custom __new__ method
    # THEN: Should return False for default and True for custom
    assert not _has_custom_method(DefaultNew, "__new__")
    assert _has_custom_method(CustomNew, "__new__")


def test_has_custom_init_method() -> None:
    """Test detection of custom __init__ method."""

    # GIVEN: Classes with default and custom __init__ methods
    class DefaultInit:
        pass

    class CustomInit:
        def __init__(self) -> None:
            pass

    # WHEN: Checking for custom __init__ method
    # THEN: Should return False for default and True for custom
    assert not _has_custom_method(DefaultInit, "__init__")
    assert _has_custom_method(CustomInit, "__init__")


def test_class_without_custom_methods() -> None:
    """Test class without custom __new__ or __init__."""

    # GIVEN: A yamlable class without custom methods
    @yamlable
    class SimpleClass:
        pass

    # WHEN: Creating an instance of the class
    obj = SimpleClass()

    # THEN: Should not have wrapped methods since they're default
    assert not hasattr(SimpleClass, "__getnewargs_ex__")
    # THEN: Should not have serialization methods since no custom methods were wrapped
    assert not hasattr(obj, "__getinitargs_ex__")


def test_class_with_custom_new() -> None:
    """Test class with custom __new__ method."""

    @yamlable
    class CustomNewClass:
        value: Any

        def __new__(cls, value: Any) -> Self:
            instance = super().__new__(cls)
            instance.value = value
            return instance

    # Create instance
    obj = CustomNewClass(42)

    # Should have __getnewargs_ex__ method
    assert hasattr(obj, "__getnewargs_ex__")
    args, kwargs = obj.__getnewargs_ex__()
    assert args == (42,)
    assert kwargs == {}


def test_class_with_custom_init() -> None:
    """Test class with custom __init__ method."""

    # GIVEN: A yamlable class with custom __init__ method
    @yamlable
    class CustomInitClass:
        def __init__(self, value: Any, name: Any = "default"):
            self.value = value
            self.name = name

    # WHEN: Creating an instance with arguments
    obj = CustomInitClass(42, name="test")

    # THEN: Should have __getinitargs_ex__ method
    assert hasattr(obj, "__getinitargs_ex__")
    args, kwargs = obj.__getinitargs_ex__()
    assert args == (42,)
    assert kwargs == {"name": "test"}


def test_derived_classes_are_yamlable() -> None:
    """Test that derived classes are also yamlable."""

    # GIVEN: Yamlable base and derived classes
    @yamlable
    class BaseClass:
        def __init__(self, base_value: Any) -> None:
            self.base_value = base_value

    @yamlable
    class DerivedClass(BaseClass):
        def __init__(self, base_value: Any, derived_value: Any) -> None:
            super().__init__(base_value)
            self.derived_value = derived_value

    # WHEN: Creating a derived instance
    obj = DerivedClass(10, 20)

    # THEN: Should have serialization method
    assert hasattr(obj, "__getinitargs_ex__")
    args, kwargs = obj.__getinitargs_ex__()
    assert args == (10, 20)
    assert kwargs == {}
