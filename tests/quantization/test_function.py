# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import Any

import pytest
import torch

from fastforward.quantization import function
from fastforward.quantized_tensor import QuantizedTensor


class MockQuantizationFunction(function.QuantizationFunction):
    @staticmethod
    def quantize(  # type: ignore[override]
        data: torch.Tensor, scale: float | torch.Tensor, extra_arg: float | torch.Tensor
    ) -> torch.Tensor:
        return data * scale

    @staticmethod
    def dequantize(  # type: ignore[override]
        data: torch.Tensor, scale: float | torch.Tensor, extra_arg: float | torch.Tensor
    ) -> torch.Tensor:
        return data / scale


class MockExplicitAutogradQuantizationFunction(function.QuantizationAutogradFunction):
    @staticmethod
    def quantize(  # type: ignore[override]
        ctx: Any, data: torch.Tensor, scale: float | torch.Tensor, extra_arg: float | torch.Tensor
    ) -> torch.Tensor:
        return data * scale

    @staticmethod
    def dequantize(  # type: ignore[override]
        data: torch.Tensor, scale: float | torch.Tensor, extra_arg: float | torch.Tensor
    ) -> torch.Tensor:
        return data / scale

    @staticmethod
    def quant_dequant_backward(
        ctx: torch.autograd.function.FunctionCtx, grad: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        # This gradient is intentionally wrong so that we can test it is called
        return grad * 3.0, None, None


@pytest.fixture
def quant_args() -> dict[str, float | torch.Tensor]:
    return {"scale": torch.tensor([0.1], requires_grad=True), "extra_arg": 2.0}


@pytest.fixture
def bound_quant_func(
    quant_args: dict[str, float | torch.Tensor],
) -> function.BoundQuantizationFunction:
    return MockQuantizationFunction.bind(**quant_args)


def test_implicit_dequantize():
    x = torch.randn((3, 3), requires_grad=True)
    x_ = x.clone().detach()
    x_.requires_grad_()

    def _dequantize_fn(data: torch.Tensor) -> torch.Tensor:
        return data * 4.0

    x_out = function._ImplicitDequantize.apply(_dequantize_fn, x)
    x_out.sum().backward()

    _dequantize_fn(x_).sum().backward()

    torch.testing.assert_close(x_out, x)
    torch.testing.assert_close(x.grad, x_.grad)


def test_quantization_function_bind():
    bound_function = MockQuantizationFunction.bind(extra_arg=10.0, scale=20.0)
    assert bound_function.quant_func == MockQuantizationFunction
    assert bound_function.args == (20.0, 10.0)


def test_quantization_function_apply(quant_args: dict[str, float | torch.Tensor]):
    data = torch.randn((3, 3))
    quant_data = MockQuantizationFunction.apply(data, **quant_args)

    expected_raw_data = MockQuantizationFunction.quantize(data, **quant_args)
    torch.testing.assert_close(quant_data.raw_data, expected_raw_data)

    dequant_data = quant_data.dequantize()
    expected_dequant_data = MockQuantizationFunction.dequantize(expected_raw_data, **quant_args)
    torch.testing.assert_close(dequant_data, expected_dequant_data)


def test_quantization_function_attach(quant_args: dict[str, float | torch.Tensor]):
    data = torch.randn((3, 3))
    quant_data = MockQuantizationFunction.attach(data, **quant_args)
    assert isinstance(quant_data, QuantizedTensor)
    assert quant_data.quant_func == MockQuantizationFunction
    torch.testing.assert_close(quant_data.raw_data, data)


def test_quantizatin_function_implicit_autograd(quant_args: dict[str, float | torch.Tensor]):
    data = torch.randn((3, 3), requires_grad=True)
    quant_data = MockQuantizationFunction.apply(data, **quant_args)
    quant_data.dequantize().sum().backward()
    torch.testing.assert_close(data.grad, torch.ones(data.shape))


def test_quantization_function_explcit_autograd(quant_args: dict[str, float | torch.Tensor]):
    data = torch.randn((3, 3), requires_grad=True)
    quant_data = MockExplicitAutogradQuantizationFunction.apply(data, **quant_args)
    quant_data.dequantize().sum().backward()
    torch.testing.assert_close(data.grad, torch.ones(data.shape) * 3.0)


def test_quant_args():
    args = function.QuantArgs(arg1=1, arg2=2)
    assert args["arg1"] == 1
    assert args["arg2"] == 2
    assert list(args) == ["arg1", "arg2"]
    assert len(args) == 2
    assert {**args} == {"arg1": 1, "arg2": 2}


def test_bound_quantization_function_clone(
    bound_quant_func: function.BoundQuantizationFunction,
):
    cloned_quant_func = bound_quant_func.clone()
    args = bound_quant_func.arguments()
    cloned_args = cloned_quant_func.arguments()

    assert args.scale is not cloned_args.scale
    assert args.scale.data_ptr != cloned_args.scale.data_ptr
    torch.testing.assert_close(args.scale, cloned_args.scale)
    assert cloned_args.scale.grad_fn is not None


def test_bound_quantization_function_detach_arguments(
    bound_quant_func: function.BoundQuantizationFunction,
):
    detached_quant_func = bound_quant_func.detach_arguments()
    args = bound_quant_func.arguments()
    cloned_args = detached_quant_func.arguments()

    assert args.scale is not cloned_args.scale
    torch.testing.assert_close(args.scale, cloned_args.scale)
    assert cloned_args.scale.grad_fn is None


def test_bound_quantization_function_arguments(
    bound_quant_func: function.BoundQuantizationFunction,
    quant_args: dict[str, float | torch.Tensor],
):
    assert {**bound_quant_func.arguments()} == quant_args


def test_bound_quantization_function_quant_dequant(
    bound_quant_func: function.BoundQuantizationFunction,
    quant_args: dict[str, float | torch.Tensor],
):
    data = torch.randn((3, 3))
    quantized_data = bound_quant_func(data)
    assert isinstance(quantized_data, QuantizedTensor)
    torch.testing.assert_close(quantized_data.raw_data, data * quant_args["scale"])

    quantized_data = quantized_data.detach()
    quantized_data.requires_grad_()
    dequantized_data = bound_quant_func.dequantize(quantized_data)

    assert not isinstance(dequantized_data, QuantizedTensor)

    # Test if STE is used for backward pass
    dequantized_data.sum().backward()
    torch.testing.assert_close(quantized_data.grad, torch.ones(quantized_data.shape))


def test_bound_quantization_function_to(
    bound_quant_func: function.BoundQuantizationFunction,
):
    moved_func = bound_quant_func.to("meta")
    assert moved_func.arguments().scale.device == torch.device("meta")
    assert moved_func.arguments().extra_arg == bound_quant_func.arguments().extra_arg


def test_bound_quantization_function_rebind(
    bound_quant_func: function.BoundQuantizationFunction,
):
    rebound_func = bound_quant_func.rebind(extra_arg=20.0)
    assert rebound_func.arguments().extra_arg == 20.0

    with pytest.raises(TypeError):
        bound_quant_func.rebind(nonexisting_arg=4.0)


def test_bound_quantization_function_attach(
    bound_quant_func: function.BoundQuantizationFunction,
):
    data = torch.rand((3, 3))
    quantized_tensor = bound_quant_func.attach(data)
    assert quantized_tensor.quant_func == bound_quant_func.quant_func


def test_bound_quantization_function_rebind_and_attach(
    bound_quant_func: function.BoundQuantizationFunction,
    quant_args: dict[str, float | torch.Tensor],
):
    data = torch.rand((3, 3))
    quantized_tensor = bound_quant_func.attach(data)
    rebound_tensor = bound_quant_func.rebind_and_attach(quantized_tensor, extra_arg=20.0)
    assert rebound_tensor.quant_func == bound_quant_func.quant_func
    assert rebound_tensor.quant_args().extra_arg == 20.0

    quantized_tensor = MockExplicitAutogradQuantizationFunction.attach(data, **quant_args)
    with pytest.raises(ValueError):
        bound_quant_func.rebind_and_attach(quantized_tensor, extra_arg=20.0)
