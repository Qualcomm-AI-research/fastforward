# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import copy
import functools

from typing import Any

import yaml


def _has_custom_method(cls: type, method_name: str) -> bool:
    """Check if a class has a custom method implementation.

    Args:
        cls: The class to check.
        method_name: The name of the method to check (e.g., '__new__', '__init__').

    Returns:
        True if the class has a custom method implementation, False if it uses the default.
        If there is no such method, the AttributeError will be raised.
    """
    class_method = getattr(cls, method_name)
    super_method = getattr(super(cls, cls), method_name)
    return class_method is not super_method


def yamlable(cls: type) -> type:
    """Class decorator that stores initialization arguments for YAML serialization.

    Wraps the class's __init__ method to store their respective args/kwargs in __getinitargs_ex__
    for later reconstruction from YAML. Only wraps methods that are not the default superclass
    implementations.

    Args:
        cls: The class to be decorated.

    Returns:
        The decorated class with wrapped __new__ and/or __init__ methods.
    """
    original_init = getattr(cls, "__init__")

    @functools.wraps(original_init)
    def wrapper_init(self: Any, *args: Any, **kwargs: Any) -> None:
        initargs_ex = copy.deepcopy((args, kwargs))
        original_init(self, *args, **kwargs)
        setattr(self, "__initargs_ex__", initargs_ex)
        setattr(self.__class__, "__getinitargs_ex__", lambda self: self.__initargs_ex__)

    # Wrap __init__ method to save its arguments if a class defines own implementation
    if _has_custom_method(cls, "__init__"):
        setattr(cls, "__init__", wrapper_init)
    return cls


def register_yaml_handlers() -> None:
    """Register YAML handlers for custom object serialization and deserialization.

    Sets up custom YAML representers and constructors for Quantizer, Tag, and
    Granularity classes to enable proper serialization/deserialization of these
    objects to/from YAML format using the !ff.obj tag.
    """
    from ._import import QualifiedNameReference, fully_qualified_name
    from .nn.quantizer import Quantizer, Tag
    from .quantization.granularity import Granularity

    def _ff_obj_yaml_representer(dumper: yaml.Dumper, data: Any) -> yaml.MappingNode:
        type_ = fully_qualified_name(data.__class__)

        if not hasattr(data, "__getinitargs_ex__") and not hasattr(data, "__getnewargs_ex__"):
            msg = f"The object can not be serialized to yaml because it is not clear what arguments should be passed to the __init__ to deserialize object. Please define '__getnewargs_ex__' attribute in the type '{type}'."
            raise RuntimeError(msg)
        state = {
            "name": f"{type_}",
        }
        if hasattr(data, "__getnewargs_ex__"):
            state.update(newargs=data.__getnewargs_ex__())
        if hasattr(data, "__getinitargs_ex__"):
            state.update(initargs=data.__getinitargs_ex__())
        # in python 3.11+ object provides a default implementation of `__getstate__`.
        # state is stored only if there is a way to restore it.
        if hasattr(data, "__setstate__") and hasattr(data, "__getstate__"):
            state.update(state=data.__getstate__())
        return dumper.represent_mapping(
            "!ff.obj",
            state,
        )

    def _ff_obj_yaml_constructor(loader: yaml.Loader, _tag: str, node: yaml.MappingNode) -> Any:
        values = loader.construct_mapping(node, deep=True)
        class_name = values.pop("name")
        newargs, newkwargs = values.pop("newargs", ((), {}))
        initargs, initkwargs = values.pop("initargs", ((), {}))
        cls = QualifiedNameReference(class_name).import_()
        if _has_custom_method(cls, "__new__"):
            obj = cls.__new__(cls, *newargs, **newkwargs)
        if _has_custom_method(cls, "__init__"):
            obj = cls(*initargs, **initkwargs)
        if obj is None:
            raise RuntimeError("cant deserialise an object")
        if "state" in values:
            obj.__setstate__(values["state"])
        return obj

    yaml.add_multi_representer(Granularity, _ff_obj_yaml_representer)  # type: ignore[type-abstract]
    yaml.add_multi_representer(Quantizer, _ff_obj_yaml_representer)
    yaml.add_multi_representer(Tag, _ff_obj_yaml_representer)
    yaml.add_multi_constructor("!ff.obj", _ff_obj_yaml_constructor)  # type: ignore[arg-type]
