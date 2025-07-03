# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import copy
import functools

from typing import Any, Callable, Concatenate, ParamSpec, TypeVar, cast

import torch
import yaml

from fastforward._import import QualifiedNameReference, fully_qualified_name

_T = TypeVar("_T")
_P = ParamSpec("_P")
_R = TypeVar("_R", covariant=True)
_DerivedT = TypeVar("_DerivedT", bound=_T)  # type: ignore[valid-type]


def _has_custom_method(cls: type, method_name: str) -> bool:
    """Check if a class has a custom method implementation.

    Args:
        cls: The class to check.
        method_name: The name of the method to check (e.g., '__new__', '__init__').

    Returns:
        True if the class has a custom method implementation, False if it uses the default.
    """
    return method_name in cls.__dict__


def _wrap(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """Wrap a class method to store its arguments on the instance.

    Args:
        func: The class method to wrap

    Returns:
        Wrapped function that stores arguments in an attribute
    """

    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        # Store arguments (excluding self/cls)
        fn_args = copy.deepcopy(tuple((args[1:], kwargs)))
        fn_name = func.__name__.replace("__", "")
        result = func(*args, **kwargs)
        setattr(args[0], f"__{fn_name}args_ex__", fn_args)
        return result

    return wrapper


def _wrap_to_save_method_args(cls: type[_T], *methods: str) -> None:
    """Wrap specified methods of a class to save their arguments.

    Args:
        cls: The class whose methods should be wrapped.
        *methods: Variable number of method names to wrap (e.g., '__init__', '__new__').
    """
    for method in methods:
        if _has_custom_method(cls, method):
            setattr(cls, method, _wrap(getattr(cls, method)))


def _ff_obj_yaml_representer(dumper: yaml.Dumper, data: Any) -> yaml.MappingNode:
    """Represent an object as YAML for serialization.

    This function creates a YAML representation of an object that includes
    the class name and the arguments needed to reconstruct the object.

    Args:
        dumper: The YAML dumper instance.
        data: The object to be serialized to YAML.

    Returns:
        A YAML mapping node representing the object.

    Raises:
        RuntimeError: If the object cannot be serialized because it lacks
                     the necessary argument information (__initargs_ex__ or __newargs_ex__).
    """
    type_ = fully_qualified_name(data.__class__)

    if not hasattr(data, "__initargs_ex__") and not hasattr(data, "__newargs_ex__"):
        msg = (
            "There are neither `__initargs_ex__` nor `__newargs_ex__` attributes in the object."
            f"Maybe the object with {type(data)} can be serialized to yaml without @yamlable decorator."
        )
        raise RuntimeError(msg)
    state = {
        "name": f"{type_}",
    }
    if hasattr(data, "__newargs_ex__"):
        state["newargs"] = data.__newargs_ex__
    if hasattr(data, "__initargs_ex__"):
        state["initargs"] = data.__initargs_ex__
    # in python 3.11+ object provides a default implementation of `__getstate__`.
    # state is stored only if there is a way to restore it.
    if hasattr(data, "__setstate__") and hasattr(data, "__getstate__"):
        state["state"] = data.__getstate__()
    return dumper.represent_mapping(
        "!ff.obj",
        state,
    )


def _ff_obj_yaml_constructor(
    loader: yaml.Loader | yaml.FullLoader | yaml.UnsafeLoader, _tag: str, node: yaml.Node
) -> Any:
    """Construct an object from YAML during deserialization.

    This function reconstructs an object from its YAML representation by
    extracting the class name and constructor arguments, then creating
    the object using the appropriate __new__ and __init__ methods.

    Args:
        loader: The YAML loader instance.
        _tag: The YAML tag (unused but required by YAML constructor interface).
        node: The YAML mapping node containing the object data.

    Returns:
        The reconstructed object instance.

    Raises:
        RuntimeError: If the object cannot be deserialized.
    """
    values = loader.construct_mapping(cast(yaml.MappingNode, node), deep=True)
    class_name = values.pop("name")
    newargs, newkwargs = values.pop("newargs", ((), {}))
    initargs, initkwargs = values.pop("initargs", ((), {}))
    cls = QualifiedNameReference(class_name).import_()
    if not isinstance(cls, type):
        return cls
    if _has_custom_method(cls, "__new__"):
        obj = cls.__new__(cls, *newargs, **newkwargs)
    if _has_custom_method(cls, "__init__"):
        obj = cls(*initargs, **initkwargs)
    if obj is None:
        raise RuntimeError("cant deserialise an object")
    if "state" in values:
        obj.__setstate__(values["state"])
    return obj


yaml.add_multi_representer(
    torch.dtype, lambda dumper, data: dumper.represent_mapping("!ff.obj", {"name": f"{data}"})
)


def yamlable(
    cls: type[_T],
    /,
    tag: str = "!ff.obj",
    yaml_constructor: Callable[
        [yaml.Loader | yaml.FullLoader | yaml.UnsafeLoader, str, yaml.Node], Any
    ] = _ff_obj_yaml_constructor,
    yaml_representer: Callable[[yaml.Dumper, Any], yaml.MappingNode] = _ff_obj_yaml_representer,
) -> type[_T]:
    """Decorator that makes a class serializable to/from YAML.

    This decorator wraps the __init__ and __new__ methods of a class to store their
    arguments, enabling automatic YAML serialization and deserialization. It also
    registers custom YAML constructor and representer functions for the class.

    When a class uses this decorator, instances will have the following attributes
    automatically added to store constructor arguments:
    - __newargs_ex__: tuple containing (args, kwargs) passed to __new__
    - __initargs_ex__: tuple containing (args, kwargs) passed to __init__

    The decorator ensures that both the current class and all its subclasses
    have their __init__ and __new__ methods wrapped when they are overridden.

    Args:
        cls: The class to decorate.
        tag: YAML tag to use for this class (default: "!ff.obj").
        yaml_constructor: Function to construct objects from YAML
            (default: _ff_obj_yaml_constructor).
        yaml_representer: Function to represent objects as YAML
            (default: _ff_obj_yaml_representer).

    Returns:
        The decorated class with YAML serialization capabilities and wrapped methods.

    Example:
        @yamlable
        class MyClass:
            def __init__(self, value: int):
                self.value = value

        # Objects can now be serialized to/from YAML automatically
        obj = MyClass(42)
        yaml_str = yaml.dump(obj)
        restored_obj = yaml.load(yaml_str, Loader=yaml.Loader)
    """
    _wrap_to_save_method_args(cls, "__new__", "__init__")

    def wrap_init_subclass(
        func: Callable[Concatenate[type[_DerivedT], _P], None], is_custom_impl: bool
    ) -> Callable[Concatenate[type[_DerivedT], _P], None]:
        def wrapper(subcls: type[_DerivedT], /, *args: _P.args, **kwargs: _P.kwargs) -> None:
            if is_custom_impl:
                # since func is a classmethod, the first argument is passed automatically,
                # subcls should not be passed to avoid an extra positional argument.
                func(*args, **kwargs)  # type: ignore[arg-type]
            else:
                super(cls, subcls).__init_subclass__(*args, **kwargs)  # type: ignore[misc]
            _wrap_to_save_method_args(subcls, "__new__", "__init__")

        return classmethod(wrapper)  # type: ignore[return-value]

    setattr(
        cls,
        "__init_subclass__",
        wrap_init_subclass(cls.__init_subclass__, _has_custom_method(cls, "__init_subclass__")),  # type: ignore[arg-type]
    )

    yaml.add_multi_constructor(tag, yaml_constructor)
    yaml.add_multi_representer(cls, yaml_representer)

    return cls
