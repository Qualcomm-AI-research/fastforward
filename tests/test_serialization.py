# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Any
from unittest.mock import patch

import pytest
import yaml

from fastforward.serialization import _has_custom_method, yamlable
from typing_extensions import Self


def test_has_custom_method() -> None:
    """Test detection of default methods."""

    # GIVEN: A class with default `__new__` and custom __init__ methods
    class Default:
        pass

    # GIVEN: A class with custom __init_subclass__ methods
    class Custom:
        def __new__(cls) -> Self:
            return super().__new__(cls)

        def __init__(self) -> None:
            pass

        def __init_subclass__(self) -> None:
            pass

    # WHEN: Checking if the class has custom methods
    # THEN: Should return False for methods from Default class
    assert not _has_custom_method(Default, "__new__")
    assert not _has_custom_method(Default, "__init__")
    assert not _has_custom_method(Default, "__init_subclass__")
    # WHEN: Checking if the class has custom methods
    # THEN: Should return True for methods from Custom class
    assert _has_custom_method(Custom, "__new__")
    assert _has_custom_method(Custom, "__init__")
    assert _has_custom_method(Custom, "__init_subclass__")


def test_class_without_custom_new_inint() -> None:
    """Test class without custom __new__ and __init__."""

    # GIVEN: A yamlable class without custom methods
    @yamlable
    class Default:
        pass

    # WHEN: Serializing instance of that class to yaml
    # THEN: Exception should be raised
    with pytest.raises(RuntimeError, match="neither `__initargs_ex__` nor `__newargs_ex__`"):
        yaml.dump(Default())


def test_class_with_custom_new_init() -> None:
    """Test class with custom __init__ and __new__ methods."""

    # GIVEN: A yamlable class with custom new and __init__ methods
    @yamlable
    class Custom:
        def __new__(cls, *_args: Any, **_kwargs: Any) -> Self:
            return super().__new__(cls)

        def __init__(self, value: Any, name: Any = "default") -> None:
            self.value = value
            self.name = name

    # WHEN: Creating a derived instance
    obj = Custom(10)

    # THEN: Should store arguments of new and init methods
    assert getattr(obj, "__newargs_ex__", (None, None)) == ((10,), {})
    assert getattr(obj, "__initargs_ex__", (None, None)) == ((10,), {})


def test_derived_classes_are_yamlable() -> None:
    """Test that derived class are also yamlable."""

    # GIVEN: Yamlable base and derived classes
    @yamlable
    class Base:
        def __init__(self, base: Any) -> None:
            super().__init__()
            self.base = base

    class Derived(Base):
        def __init__(self, base: Any, derived: Any) -> None:
            super().__init__(base)
            self.derived_value = derived

    # WHEN: Creating a derived instance
    obj = Derived(10, 20)

    # THEN: Should store arguments of derived init method
    assert getattr(obj, "__initargs_ex__", (None, None)) == ((10, 20), {})


def test_init_subclass_inheritance() -> None:
    """Test that user provided implementation of __init_subclass__ is not overridden."""

    # GIVEN: A yamlable base class with custom __init_subclass__ implementation
    @yamlable
    class Custom:
        def __init_subclass__(subcls) -> None:
            super().__init_subclass__()

    # WHEN: A derived class is defined
    with patch.object(
        Custom, "__init_subclass__", wraps=Custom.__init_subclass__
    ) as init_subclass_mock:

        class Derived(Custom):
            def __init__(self) -> None:
                super().__init__()

        # THEN: The user provided implementation should be called
        init_subclass_mock.assert_called_once()
        # if `__init_subclass__` is wrapped by yamable and mock, the base class is sent to
        # `__init_subclass__` instead of derived. Thus, we can't check whether argiments of derived
        # class init method are stored.


def test_init_subclass_multiple_inheritance() -> None:
    """Test that user provided implementation of __init_subclass__ is not overridden."""

    # GIVEN: A yamable base class with default __init_subclass__ implementation
    @yamlable
    class Default:
        pass

    # GIVEN: A base class with custom __init_subclass__ implementation
    class Custom:
        def __init_subclass__(subcls) -> None:
            super().__init_subclass__()

    # WHEN: A derived class is defined
    with patch.object(
        Custom, "__init_subclass__", wraps=Custom.__init_subclass__
    ) as init_subclass_mock:

        class Derived(Default, Custom):
            def __init__(self) -> None:
                super().__init__()

        # THEN: The implementation provided by yamable decorator should not break MRO chain
        init_subclass_mock.assert_called_once()
        # THEN: Args of Derived.__init__ should be stored
        assert getattr(Derived(), "__initargs_ex__", (None, None)) == ((), {})
