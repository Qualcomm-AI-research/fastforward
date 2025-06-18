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

    class DefaultNew:
        pass

    class CustomNew:
        def __new__(cls) -> Self:
            return super().__new__(cls)

    assert not _has_custom_method(DefaultNew, "__new__")
    assert _has_custom_method(CustomNew, "__new__")


def test_has_custom_init_method() -> None:
    """Test detection of custom __init__ method."""

    class DefaultInit:
        pass

    class CustomInit:
        def __init__(self) -> None:
            pass

    assert not _has_custom_method(DefaultInit, "__init__")
    assert _has_custom_method(CustomInit, "__init__")


def test_class_without_custom_methods() -> None:
    """Test class without custom __new__ or __init__."""

    @yamlable
    class SimpleClass:
        pass

    # Should not have wrapped methods since they're default
    assert not hasattr(SimpleClass, "__getnewargs_ex__")

    # Create instance
    obj = SimpleClass()

    # Should not have serialization methods since no custom methods were wrapped
    assert not hasattr(obj, "__getnewargs_ex__")
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

    @yamlable
    class CustomInitClass:
        def __init__(self, value: Any, name: Any = "default"):
            self.value = value
            self.name = name

    # Create instance
    obj = CustomInitClass(42, name="test")

    # Should have __getinitargs_ex__ method
    assert hasattr(obj, "__getinitargs_ex__")
    args, kwargs = obj.__getinitargs_ex__()
    assert args == (42,)
    assert kwargs == {"name": "test"}


def test_derived_classes_are_yamlable() -> None:
    """Test that derived classes are also yamlable."""

    @yamlable
    class BaseClass:
        def __init__(self, base_value: Any) -> None:
            self.base_value = base_value

    @yamlable
    class DerivedClass(BaseClass):
        def __init__(self, base_value: Any, derived_value: Any) -> None:
            super().__init__(base_value)
            self.derived_value = derived_value

    # Create derived instance
    obj = DerivedClass(10, 20)

    # Should have serialization method
    assert hasattr(obj, "__getinitargs_ex__")
    args, kwargs = obj.__getinitargs_ex__()
    assert args == (10, 20)
    assert kwargs == {}


def test_custom_new_multiple_objects() -> None:
    """Test class with custom __new__ works even if two objects are created."""

    @yamlable
    class MultipleNewClass:
        value: Any

        def __new__(cls, value: Any) -> Self:
            instance = super().__new__(cls)
            instance.value = value
            return instance

    # Create first instance
    obj1 = MultipleNewClass(100)
    assert hasattr(obj1, "__getnewargs_ex__")
    args1, kwargs1 = obj1.__getnewargs_ex__()
    assert args1 == (100,)

    # Create second instance
    obj2 = MultipleNewClass(200)
    assert hasattr(obj2, "__getnewargs_ex__")
    args2, kwargs2 = obj2.__getnewargs_ex__()
    assert args2 == (200,)

    # Verify they have different values
    assert obj1.value == 100
    assert obj2.value == 200


def test_base_and_derived_yamlable_init_args() -> None:
    """Test when base class and derived classes are yamlable, initargs are stored properly."""

    @yamlable
    class YamlableBase:
        def __init__(self, base_arg: Any) -> None:
            self.base_arg = base_arg

    @yamlable
    class YamlableDerived(YamlableBase):
        def __init__(self, base_arg: Any, derived_arg: Any) -> None:
            super().__init__(base_arg)
            self.derived_arg = derived_arg

    # Create derived instance
    obj = YamlableDerived("base_value", "derived_value")

    # Should have initargs from the derived class
    assert hasattr(obj, "__getinitargs_ex__")
    args, kwargs = obj.__getinitargs_ex__()
    assert args == ("base_value", "derived_value")
    assert kwargs == {}

    # Verify attributes are set correctly
    assert obj.base_arg == "base_value"
    assert obj.derived_arg == "derived_value"


def test_base_and_derived_yamlable_new_args() -> None:
    """Test when base class and derived classes are yamlable with __new__, newargs are stored properly."""

    @yamlable
    class YamlableBaseNew:
        base_arg: Any

        def __new__(cls, base_arg: Any) -> Self:
            instance = super().__new__(cls)
            instance.base_arg = base_arg
            return instance

    @yamlable
    class YamlableDerivedNew(YamlableBaseNew):
        derived_arg: Any

        def __new__(cls, base_arg: Any, derived_arg: Any) -> Self:
            instance = super().__new__(cls, base_arg)
            instance.derived_arg = derived_arg
            return instance

    # Create derived instance
    obj = YamlableDerivedNew("base_value", "derived_value")

    # Should have newargs from the derived class
    assert hasattr(obj, "__getnewargs_ex__")
    args, kwargs = obj.__getnewargs_ex__()
    assert args == ("base_value", "derived_value")
    assert kwargs == {}

    # Verify attributes are set correctly
    assert obj.base_arg == "base_value"
    assert obj.derived_arg == "derived_value"


def test_class_with_both_custom_new_and_init() -> None:
    """Test class with both custom __new__ and __init__ methods."""

    @yamlable
    class BothCustomClass:
        new_arg: Any

        def __new__(cls, new_arg: Any, *, init_arg: Any) -> Self:
            del init_arg  # ignore __init__'s argument
            instance = super().__new__(cls)
            instance.new_arg = new_arg
            return instance

        def __init__(self, new_arg: Any, init_arg: Any = "default") -> None:
            del new_arg  # ignore __new__'s argument
            self.init_arg = init_arg

    # Create instance
    obj = BothCustomClass("new_value", init_arg="init_value")

    # Should have both serialization methods
    assert hasattr(obj, "__getnewargs_ex__")
    assert hasattr(obj, "__getinitargs_ex__")

    new_args, new_kwargs = obj.__getnewargs_ex__()
    assert new_args == ("new_value",)
    assert new_kwargs == {"init_arg": "init_value"}

    init_args, init_kwargs = obj.__getinitargs_ex__()
    assert init_args == ("new_value",)
    assert init_kwargs == {"init_arg": "init_value"}


def test_mutable_object_modification_in_new() -> None:
    """Test class that accepts mutable objects and modifies them in __new__.

    This test ensures that each object stores its own version of the __new__ arguments,
    even when mutable objects are passed and modified during construction.
    """

    @yamlable
    class MutableModifierClass:
        data: Any

        def __new__(cls, data_list: Any, multiplier: Any = 1) -> Self:
            instance = super().__new__(cls)
            # Modify the mutable list during construction
            instance.data = data_list
            instance.data.append(multiplier)
            return instance

    # Create first instance with a list
    list1 = [1, 2, 3]
    obj1 = MutableModifierClass(list1, multiplier=10)

    # Create second instance with a different list
    list2 = [4, 5, 6]
    obj2 = MutableModifierClass(list2, multiplier=20)

    # Verify both objects have __getnewargs_ex__ method
    assert hasattr(obj1, "__getnewargs_ex__")
    assert hasattr(obj2, "__getnewargs_ex__")

    # Get stored arguments for first object
    args1, kwargs1 = obj1.__getnewargs_ex__()
    assert args1 == ([1, 2, 3],)
    assert kwargs1 == {"multiplier": 10}

    # Get stored arguments for second object
    args2, kwargs2 = obj2.__getnewargs_ex__()
    assert args2 == ([4, 5, 6],)
    assert kwargs2 == {"multiplier": 20}

    # Verify that each object has its own modified data
    assert obj1.data == [1, 2, 3, 10]
    assert obj2.data == [4, 5, 6, 20]
