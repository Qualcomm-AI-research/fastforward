# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


from typing import Any, Callable

import pytest
import torch

from fastforward._autoquant.function_context import FunctionContext
from fastforward.type_common import MethodType


# Example PyTorch module for testing
class ExampleTorchModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def instance_method(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @classmethod
    def class_method(cls, x: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def static_method(x: torch.Tensor) -> torch.Tensor:
        return x


# PyTorch module with custom parameter names
class CustomModule(torch.nn.Module):
    def custom_method(custom_self, x: torch.Tensor) -> torch.Tensor:
        return x

    @classmethod
    def custom_class_method(custom_cls, x: torch.Tensor) -> torch.Tensor:
        return x


# PyTorch module with parameterless method
class NoParamModule(torch.nn.Module):
    @staticmethod
    def no_param_method() -> torch.Tensor:
        return torch.tensor([1.0])


# Example Python module functions
def module_function(x: Any) -> Any:
    return x


def another_module_function(x: Any, y: Any) -> Any:
    return x + y


@pytest.mark.parametrize(
    "module_type, method_name, expected_method_type, expected_instance_var, expected_class_var, expected_func",
    [
        (
            ExampleTorchModule,
            "instance_method",
            MethodType.METHOD,
            "self",
            None,
            ExampleTorchModule.instance_method,
        ),
        (
            ExampleTorchModule,
            "class_method",
            MethodType.CLASS_METHOD,
            None,
            "cls",
            ExampleTorchModule.class_method.__func__,  # type: ignore[attr-defined]
        ),
        (
            ExampleTorchModule,
            "static_method",
            MethodType.STATIC_METHOD,
            None,
            None,
            ExampleTorchModule.static_method,
        ),
        (
            CustomModule,
            "custom_method",
            MethodType.METHOD,
            "custom_self",
            None,
            CustomModule.custom_method,
        ),
        (
            CustomModule,
            "custom_class_method",
            MethodType.CLASS_METHOD,
            None,
            "custom_cls",
            CustomModule.custom_class_method.__func__,  # type: ignore[attr-defined]
        ),
        (
            NoParamModule,
            "no_param_method",
            MethodType.STATIC_METHOD,
            None,
            None,
            NoParamModule.no_param_method,
        ),
    ],
    ids=[
        "instance-method",
        "class-method",
        "static-method",
        "custom-instance-method",
        "custom-class-method",
        "no-param-static-method",
    ],
)
def test_from_method(
    module_type: type[torch.nn.Module],
    method_name: str,
    expected_method_type: MethodType,
    expected_instance_var: str | None,
    expected_class_var: str | None,
    expected_func: Any,
) -> None:
    """Test FunctionContext creation from various method types."""
    # GIVEN a module_type and method_name
    # WHEN creating FunctionContext from method
    context = FunctionContext.from_method(module_type, method_name)

    # THEN context correctly identifies the method type and parameters
    assert context.torch_module == module_type
    assert context.method_type == expected_method_type
    assert context.instance_var == expected_instance_var
    assert context.class_var == expected_class_var
    assert context.func == expected_func


@pytest.mark.parametrize(
    "func_ref,module,expected_func,expected_torch_module,expected_py_module,expected_method_type",
    [
        (None, ExampleTorchModule(), None, None, None, None),
        (module_function, None, module_function, None, None, None),
        (None, None, None, None, None, None),
    ],
)
def test_from_function_reference_with_none_values(
    func_ref: Callable[..., Any] | None,
    module: torch.nn.Module | None,
    expected_func: Callable[..., Any] | None,
    expected_torch_module: torch.nn.Module | None,
    expected_py_module: Any | None,
    expected_method_type: Any | None,
) -> None:
    """Test FunctionContext creation with None function reference and/or module."""
    # GIVEN a function reference and module
    # WHEN creating FunctionContext
    context = FunctionContext.from_function_reference(func_ref, module)

    # THEN context has expected values
    assert context.func == expected_func
    assert context.torch_module == expected_torch_module  # type: ignore[comparison-overlap]
    assert context.py_module == expected_py_module
    assert context.method_type == expected_method_type
    # GIVEN a function refrence and module
    # WHEN creating FunctionContext
    context = FunctionContext.from_function_reference(func_ref, module)

    # THEN context has expected values
    assert context.func == expected_func
    assert context.torch_module == expected_torch_module  # type: ignore[comparison-overlap]
    assert context.py_module == expected_py_module
    assert context.method_type == expected_method_type
