# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import dataclasses
import inspect
import types

from typing import Any, Callable

import torch

from typing_extensions import Self

from fastforward.type_common import MethodType, method_type


@dataclasses.dataclass
class FunctionContext:
    """Context information for a function or method, including its module and metadata.

    This class stores information about a function or method: its associated
    module (either a Python module or PyTorch module), method type, and
    parameter information for instance/class methods.

    Attributes:
        torch_module: The PyTorch module class if the function belongs to a torch.nn.Module
        func: The actual function or method object
        method_type: Type of method (METHOD, CLASS_METHOD, STATIC_METHOD, or NO_METHOD)
        instance_var: Name of the instance parameter (typically 'self') for instance methods
        class_var: Name of the class parameter (typically 'cls') for class methods
        py_module: The Python module if the function belongs to a regular Python module
    """

    torch_module: type[torch.nn.Module] | None = None
    func: Callable[..., Any] | None = None
    method_type: MethodType | None = None
    instance_var: str | None = None
    class_var: str | None = None
    py_module: types.ModuleType | None = None

    @classmethod
    def from_function_reference(
        cls,
        func_ref: Callable[..., Any] | None,
        module: types.ModuleType | torch.nn.Module | type[torch.nn.Module] | None,
    ) -> Self:
        if func_ref is None or module is None:
            return cls(func=func_ref)

        if isinstance(module, torch.nn.Module):
            return cls._from_torch_module(type(module), func_ref)
        elif isinstance(module, type) and issubclass(module, torch.nn.Module):
            return cls._from_torch_module(module, func_ref)
        elif isinstance(module, types.ModuleType):
            return cls._from_py_module(module, func_ref)
        else:
            return cls()  # type: ignore[unreachable]

    @classmethod
    def _from_torch_module(
        cls, module: type[torch.nn.Module], func_ref: Callable[..., Any]
    ) -> Self:
        meth_type = method_type(module, func_ref.__name__)

        instance_var, class_var = None, None
        if meth_type in (MethodType.METHOD, MethodType.CLASS_METHOD):
            # Get the underlying function object from a method
            func_ref = func_ref.__func__ if isinstance(func_ref, types.MethodType) else func_ref

            param = next(iter(inspect.signature(func_ref).parameters), None)
            instance_var = param if meth_type is MethodType.METHOD else None
            class_var = param if meth_type is MethodType.CLASS_METHOD else None

        return cls(
            torch_module=module,
            func=func_ref,
            method_type=meth_type,
            instance_var=instance_var,
            class_var=class_var,
        )

    @classmethod
    def _from_py_module(cls, module: types.ModuleType, func_ref: Callable[..., Any]) -> Self:
        func_type = method_type(module, func_ref.__name__)
        assert func_type is MethodType.NO_METHOD

        return cls(
            py_module=module,
            func=func_ref,
            method_type=func_type,
            instance_var=None,
            class_var=None,
        )

    @classmethod
    def from_method(cls, module: type, name: str) -> Self:
        return cls.from_function_reference(getattr(module, name), module)
