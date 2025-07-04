# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import copy
import dataclasses

from contextlib import contextmanager
from typing import Any, Callable
from unittest.mock import MagicMock

import fastforward as ff
import pytest
import torch

from fastforward.quantization import granularity
from fastforward.quantization.affine import StaticAffineQuantParams, quantize_per_tensor
from fastforward.quantization.function import QuantizationContext, QuantizationFunction
from fastforward.quantization.random import random_quantized
from fastforward.quantized_tensor import QuantizedTensor


class _MockQuantizationFunction(QuantizationFunction[StaticAffineQuantParams]):
    @classmethod
    def quantize(cls, data: torch.Tensor, params: StaticAffineQuantParams) -> ff.QuantizedTensor:
        quantized_data = data * params.scale
        context = QuantizationContext(cls, params)
        return ff.QuantizedTensor(quantized_data, context)

    @classmethod
    def dequantize(cls, data: torch.Tensor, params: StaticAffineQuantParams) -> torch.Tensor:
        return data / params.scale


def test_quantize_dequantize(_seed_prngs: int) -> None:
    """Assert that quantize and dequant _transform_ data."""
    torch.manual_seed(7480)
    data = torch.randn(10, 10)
    scale = 4.0

    params = StaticAffineQuantParams(
        scale=scale, offset=None, num_bits=3, granularity=ff.PerTensor()
    )
    quantized_data = _MockQuantizationFunction.quantize(data, params)
    dequantized_data = quantized_data.dequantize()

    torch.testing.assert_close(data * scale, quantized_data.raw_data)
    torch.testing.assert_close(data, dequantized_data)


@pytest.mark.parametrize(
    "dtype",
    "double,float,half,bfloat16,long,int,short,char,cdouble,cfloat,chalf,bool,byte".split(","),
)
def test_dtype_methods(dtype: str, _seed_prngs: int) -> None:
    torch.manual_seed(7480)
    data = torch.randn(10, 10)

    params = StaticAffineQuantParams(scale=1.0, offset=None, num_bits=3, granularity=ff.PerTensor())
    quantized_data = _MockQuantizationFunction.quantize(data, params)
    torch.testing.assert_close(getattr(data, dtype)(), getattr(quantized_data, dtype)())


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
        torch.bool,
        torch.cdouble,
        torch.cfloat,
        torch.chalf,
        torch.complex128,
        torch.complex32,
        torch.complex64,
        torch.double,
        torch.float,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.half,
        torch.int,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.int8,
        torch.long,
        torch.short,
        torch.uint8,
    ],
)
def test_to_dtype(dtype: torch.dtype, _seed_prngs: int) -> None:
    torch.manual_seed(7480)
    data = torch.randn(10, 10)
    params = StaticAffineQuantParams(scale=1.0, offset=None, num_bits=3, granularity=ff.PerTensor())
    quantized_data = _MockQuantizationFunction.quantize(data, params)
    torch.testing.assert_close(data.to(dtype), quantized_data.to(dtype))


@ff.flags.context(ff.strict_quantization, False)
def test_quantized_tensor_cpu_cuda(_seed_prngs: int) -> None:
    """Test `Tensor.cpu` and `Tensor.cuda`.

    Test if quantized tensor and associated tensor quantization parameters are moved between
    devices and if quant.
    """
    torch.manual_seed(7480)
    data = torch.randn(10, 10)
    scale = torch.tensor(3.0)
    num_bits = 3
    params = StaticAffineQuantParams(
        scale=scale, offset=None, num_bits=num_bits, granularity=ff.PerTensor()
    )
    quantized_data = _MockQuantizationFunction.quantize(data, params)

    cuda_tensor = quantized_data.cuda()
    assert cuda_tensor.is_cuda, "Quantized tensor should move to cuda"

    cuda_tensor_args = cuda_tensor.quant_args()
    assert isinstance(cuda_tensor_args, StaticAffineQuantParams)

    cuda_tensor_scale = cuda_tensor_args.scale
    assert isinstance(cuda_tensor_scale, torch.Tensor)
    assert cuda_tensor_scale.is_cuda, "Scale tensor should move to cuda"
    assert cuda_tensor_args.num_bits == num_bits, "Non-tensor parameter should not change"
    assert cuda_tensor.quant_func == quantized_data.quant_func, "quant_func should persist"
    assert torch.allclose(cuda_tensor.dequantize(), quantized_data.dequantize().cuda()), (
        "Cuda transfer should not affect data values"
    )

    cpu_tensor = cuda_tensor.cpu()
    assert cpu_tensor.device.type == "cpu", "Quantized tensor should move back to cpu"
    cpu_tensor_args = cpu_tensor.quant_args()
    assert isinstance(cpu_tensor_args, StaticAffineQuantParams)

    cpu_scale_tensor = cpu_tensor_args.scale
    assert isinstance(cpu_scale_tensor, torch.Tensor)
    assert cpu_scale_tensor.device.type == "cpu", "Scale tensor should move back to cpu"
    assert cpu_tensor_args.num_bits == num_bits, "Non-tensor parameter should not change"
    assert cpu_tensor.quant_func == quantized_data.quant_func, "quant_func should persist"
    assert torch.allclose(cpu_tensor.dequantize(), quantized_data.dequantize()), (
        "CPU transfer should not affect data values"
    )


@ff.flags.context(ff.strict_quantization, False)
def test_quantized_tensor_to(_seed_prngs: int) -> None:
    """Test `Tensor.cuda` and `Tensor.cpu`.

    Test if quantized tensor and associated tensor quantization parameters are moved between
    devices and if quant.
    """
    data = torch.randn(10, 10)
    scale = torch.tensor(3.0)
    num_bits = 3
    params = StaticAffineQuantParams(
        scale=scale, offset=None, num_bits=num_bits, granularity=ff.PerTensor()
    )
    quantized_data = _MockQuantizationFunction.quantize(data, params)

    cuda_tensor = quantized_data.to("cuda")
    assert cuda_tensor.is_cuda, "Quantized tensor should move to cuda"

    cuda_tensor_args = cuda_tensor.quant_args()
    assert isinstance(cuda_tensor_args, StaticAffineQuantParams)

    cuda_tensor_scale = cuda_tensor_args.scale
    assert isinstance(cuda_tensor_scale, torch.Tensor)
    assert cuda_tensor_scale.is_cuda, "Scale tensor should move to cuda"
    assert cuda_tensor_args.num_bits == num_bits, "Non-tensor parameter should not change"
    assert cuda_tensor.quant_func == quantized_data.quant_func, "quant_func should persist"
    torch.testing.assert_close(
        cuda_tensor.dequantize(),
        quantized_data.dequantize().cuda(),
        rtol=0,
        atol=0,
        msg="Cuda transfer should not affect data values",
    )

    cpu_tensor = cuda_tensor.cpu()
    assert cpu_tensor.device.type == "cpu", "Quantized tensor should move back to cpu"
    cpu_tensor_args = cpu_tensor.quant_args()
    assert isinstance(cpu_tensor_args, StaticAffineQuantParams)

    cpu_scale_tensor = cpu_tensor_args.scale
    assert isinstance(cpu_scale_tensor, torch.Tensor)
    assert cpu_scale_tensor.device.type == "cpu", "Scale tensor should move back to cpu"
    assert cpu_tensor_args.num_bits == num_bits, "Non-tensor parameter should not change"
    assert cpu_tensor.quant_func == quantized_data.quant_func, "quant_func should persist"
    torch.testing.assert_close(
        cpu_tensor.dequantize(),
        quantized_data.dequantize(),
        rtol=0,
        atol=0,
        msg="CPU transfer should not affect data values",
    )


def test_quantized_tensor_grad_backward(_seed_prngs: int) -> None:
    """Test if graph is properly created and gradient are accumulated using backward."""
    torch.manual_seed(7480)
    scale = torch.tensor(3.0, requires_grad=True)

    data = torch.randn(10, 10, requires_grad=True)

    params = StaticAffineQuantParams(
        scale=scale, offset=None, num_bits=3, granularity=ff.PerTensor()
    )
    quantized_data = _MockQuantizationFunction.quantize(data, params)

    (quantized_data.dequantize() * 5.0).sum().backward()

    expected_grad = torch.ones(data.shape) * 5
    torch.testing.assert_close(data.grad, expected_grad)

    expected_scale_grad = torch.tensor(0.0)
    torch.testing.assert_close(expected_scale_grad, scale.grad)


def test_quantized_tensor_dispatches() -> None:
    torch.manual_seed(7480)
    quantized_data = random_quantized((10, 10), scale=0.2)
    dequant_data = quantized_data.dequantize()

    # Check that these methods (which have a custom kernel) do not use the
    # quantization fallback, but do give the same result as when using a
    # dequant fallback.

    @contextmanager
    def _mock_dispatcher_function(dispatch_key: str, dispatch_pos: int) -> Any:
        # Temporarily mock dispatch item, ideally we would directly mock the
        # function that is being dispatched to, but that does not seem to be
        # possible as the dispatches are already registered when the
        # QuantizedTensor is imported.

        from fastforward.dispatcher import _DISPATCHER, DispatcherItem

        old_dispatch_item = _DISPATCHER[dispatch_key][dispatch_pos]
        mock = MagicMock(wraps=old_dispatch_item.fn)
        _DISPATCHER[dispatch_key][dispatch_pos] = DispatcherItem(old_dispatch_item.predicate, mock)
        yield mock
        _DISPATCHER[dispatch_key][dispatch_pos] = old_dispatch_item
        return

    @contextmanager
    def _assert_ran_with_kernel(
        kernel_reference: Callable[..., Any],
        dispatch_key: str,
        dispatch_pos: int = 0,
    ) -> Any:
        with _mock_dispatcher_function(dispatch_key, dispatch_pos) as mock:
            assert mock._mock_wraps is kernel_reference
            yield
        mock.assert_called_once()

    # Use strict quantization context to make sure the dequantization fallback is not called
    with ff.strict_quantization(True):
        with _assert_ran_with_kernel(ff.quantization._linear_quantized_ops.scalar_multiply, "mul"):
            out = quantized_data * 3
        torch.testing.assert_close(out.dequantize(), dequant_data * 3)

        with _assert_ran_with_kernel(ff.quantization._linear_quantized_ops.view, "view"):
            out = quantized_data.view(-1)
        torch.testing.assert_close(out.dequantize(), dequant_data.view(-1))

        with _assert_ran_with_kernel(ff.quantization._linear_quantized_ops.reshape, "reshape"):
            out = quantized_data.reshape(1, dequant_data.numel())
        torch.testing.assert_close(out.dequantize(), dequant_data.reshape(1, dequant_data.numel()))

        with _assert_ran_with_kernel(ff.quantization._linear_quantized_ops.transpose, "transpose"):
            out = quantized_data.transpose(0, 1)
        torch.testing.assert_close(out.dequantize(), dequant_data.transpose(0, 1))

        with _assert_ran_with_kernel(ff.quantization._linear_quantized_ops.cat, "cat"):
            out = torch.cat([quantized_data, quantized_data])
        torch.testing.assert_close(
            out.dequantize(), torch.cat([dequant_data, dequant_data]).contiguous()
        )


def test_quantized_tensor_detach(_seed_prngs: int) -> None:
    data = torch.randn((2, 2), requires_grad=True)
    scale = torch.tensor([0.01])
    quantized_data = quantize_per_tensor(data, scale, None, 3)
    quantized_detached = quantized_data.detach()

    assert quantized_detached.grad_fn is None
    assert not quantized_detached.requires_grad
    assert quantized_detached.data_ptr() == quantized_data.data_ptr()


@ff.flags.context(ff.strict_quantization, False)
def test_quantized_tensor_clone(_seed_prngs: int) -> None:
    data = torch.randn((2, 2)).clamp(-2.5, 3)
    scale = torch.tensor([1.0])
    data.requires_grad_()
    quantized_data = quantize_per_tensor(data, scale, None, 3)

    quantized_cloned = quantized_data.clone()

    assert quantized_cloned.grad_fn is not None
    assert quantized_cloned.requires_grad
    assert quantized_cloned.data_ptr() != quantized_data.data_ptr()

    # Test if gradient 'flows' back to leaf through cloning
    mul_value = 3.0
    quantized_cloned.sum().mul(mul_value).backward()
    actual_grad = data.grad
    expected_grad = torch.ones(data.shape) * mul_value
    torch.testing.assert_close(actual_grad, expected_grad)


def test_deepcopy() -> None:
    scale = torch.tensor([0.1, 0.2, 0.3])
    offset = 0
    quantized_data = random_quantized(
        (3, 3), scale=scale, offset=offset, granularity=granularity.PerChannel(1)
    )
    copied_data = copy.deepcopy(quantized_data)

    with ff.strict_quantization(False):
        assert (quantized_data == copied_data).all()

    assert quantized_data is not copied_data
    assert quantized_data.data_ptr != copied_data.data_ptr

    quantized_args = quantized_data.quant_args()
    copied_args = copied_data.quant_args()

    for arg_key in dataclasses.asdict(quantized_args).keys():
        original_arg = getattr(quantized_args, arg_key)
        copied_arg = getattr(copied_args, arg_key)
        if isinstance(original_arg, torch.Tensor):
            assert (original_arg == copied_arg).all(), f"{arg_key} is not equal"
            assert original_arg is not copied_arg, f"{arg_key} is not deepcopied"
            assert original_arg.data_ptr is not copied_arg.data_ptr, (
                f"{arg_key} deepcopy aliases storage"
            )
        else:
            assert original_arg == copied_arg, f"{arg_key} is not equal"


def test_contiguous(_seed_prngs: int) -> None:
    # Construct a quantized tensor with non contiguous data and parameters
    scale = torch.randn(3, 3)[:, 0]
    offset = torch.ones((3, 3))[:, 0]
    quantized_tensor = random_quantized(
        (3, 3), scale=scale, offset=offset, granularity=granularity.PerChannel(0)
    )
    quantized_tensor = QuantizedTensor(
        quantized_tensor.raw_data.T, quantized_tensor.quantization_context
    )

    assert not quantized_tensor.is_contiguous()
    assert not quantized_tensor.quant_args().scale.is_contiguous()  # type: ignore[attr-defined]

    contiguous_tensor = quantized_tensor.contiguous()
    assert contiguous_tensor.is_contiguous()
    assert contiguous_tensor.quant_args().scale.is_contiguous()  # type: ignore[attr-defined]
    assert contiguous_tensor.quant_args().offset.is_contiguous()  # type: ignore[attr-defined]


def test_not_implemented_registration() -> None:
    quantized_data = random_quantized((3, 3))
    with pytest.raises(NotImplementedError):
        reversed(quantized_data)


class TestView:
    @pytest.fixture
    def shape(self) -> tuple[int, ...]:
        return 1, 1, 8, 8

    @pytest.fixture
    def new_shape(self, shape: torch.Size) -> tuple[int, ...]:
        del shape
        return 1, 1, 1, 64

    @pytest.fixture
    def qx_per_channel(self, shape: torch.Size, _seed_prngs: int) -> QuantizedTensor:
        scale = torch.randn(shape[-1])
        return ff.random.random_quantized(shape, scale=scale, granularity=ff.PerChannel(-1))

    @pytest.fixture
    def qx_per_tensor(self, shape: torch.Size, _seed_prngs: int) -> QuantizedTensor:
        scale = torch.randn(1)
        return ff.random.random_quantized(shape, scale=scale, granularity=ff.PerTensor())

    def test_per_tensor_with_tuple_shape(
        self, qx_per_tensor: QuantizedTensor, new_shape: torch.Size
    ) -> None:
        qx = qx_per_tensor
        out_ff = qx.view(new_shape)
        torch.testing.assert_close(out_ff.dequantize(), qx.dequantize().view(new_shape))

    def test_per_tensor_with_int_shape(
        self, qx_per_tensor: QuantizedTensor, new_shape: torch.Size
    ) -> None:
        qx = qx_per_tensor
        out_ff = qx.view(*new_shape)
        torch.testing.assert_close(out_ff.dequantize(), qx.dequantize().view(*new_shape))

    def test_per_channel_with_same_tuple_shape(
        self, qx_per_channel: QuantizedTensor, shape: torch.Size
    ) -> None:
        qx = qx_per_channel
        out_ff = qx.view(shape)
        torch.testing.assert_close(out_ff.dequantize(), qx.dequantize().view(shape))

    def test_per_channel_with_same_int_shape(
        self, qx_per_channel: QuantizedTensor, shape: torch.Size
    ) -> None:
        qx = qx_per_channel
        out_ff = qx.view(*shape)
        torch.testing.assert_close(out_ff.dequantize(), qx.dequantize().view(*shape))
