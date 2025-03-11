# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

# pylint: disable=missing-function-docstring
import math

from typing import Any, cast

import fastforward
import pytest
import torch

from fastforward.nn.linear_quantizer import LinearQuantizer
from fastforward.quantization import granularity
from fastforward.quantization.affine import AffineQuantizationFunction, StaticAffineQuantParams
from fastforward.quantized_tensor import QuantizedTensor


def test_linear_quantizer_function() -> None:
    data = torch.linspace(-8, 8, 17)
    scale = torch.tensor(2.0)
    offset = None
    num_bits = 2

    def parameter_dictionary() -> dict[str, Any]:
        return {
            "scale": scale,
            "offset": offset,
            "granularity": fastforward.PerTensor(),
            "num_bits": num_bits,
            "quantized_dtype": None,
        }

    params = parameter_dictionary()

    quant_params = StaticAffineQuantParams(**params)
    quant_data = AffineQuantizationFunction.quantize(data, quant_params)
    raw_quant_data = quant_data.raw_data
    expected_raw = torch.tensor([
        -2.0,
        -2.0,
        -2.0,
        -2.0,
        -2.0,
        -2.0,
        -1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ])
    torch.testing.assert_close(expected_raw, raw_quant_data)

    dequant_data = quant_data.dequantize()
    expected_dequant = expected_raw * scale - (offset or 0.0)  # type: ignore[redundant-expr]
    torch.testing.assert_close(expected_dequant, dequant_data)

    # Symmetric case
    params["offset"] = None
    quant_params = StaticAffineQuantParams(**params)
    symmetric_output = AffineQuantizationFunction.quantize(data, quant_params).dequantize()
    min_int = -(2 ** (num_bits - 1))
    max_int = 2 ** (num_bits - 1) - 1
    expected_symmetric = torch.clamp(torch.round(data / scale), min_int, max_int) * scale
    torch.testing.assert_close(expected_symmetric, symmetric_output)


def test_linear_quantizer_op_gradients() -> None:
    data = torch.linspace(-8, 8, 17)
    scale = torch.tensor(2.0)
    offset = torch.tensor(2.3)
    num_bits = 2

    data.requires_grad = True
    scale.requires_grad = True
    offset.requires_grad = True

    quant_params = StaticAffineQuantParams(
        scale=scale, offset=offset, num_bits=num_bits, granularity=fastforward.PerTensor()
    )
    quant_data = AffineQuantizationFunction.quantize(data, quant_params)
    quant_data.backward(torch.ones_like(quant_data))

    assert data.grad is not None
    assert scale.grad is not None
    assert offset.grad is not None

    assert torch.count_nonzero(data.grad)
    assert torch.count_nonzero(scale.grad)
    assert torch.count_nonzero(offset.grad)


@pytest.mark.parametrize("quantized_dtype", [torch.float16, torch.float32, torch.int16])
@pytest.mark.parametrize("param_dtype", [torch.float16, torch.float32])
def test_linear_quantizer_initialisation_behavior(
    quantized_dtype: torch.dtype, param_dtype: torch.dtype
) -> None:
    data = torch.rand((32, 14, 17))
    num_bits = 2
    quantizer = LinearQuantizer(
        num_bits=num_bits, symmetric=False, quantized_dtype=quantized_dtype, param_dtype=param_dtype
    )

    with pytest.raises(ValueError):
        quantizer(data)

    quantizer.quantization_range = (torch.min(data), torch.max(data))

    quantized = quantizer(data)

    assert quantized is not None
    assert isinstance(quantized, QuantizedTensor)
    assert quantized.raw_data.dtype == quantized_dtype

    for parameter in quantizer.parameters():
        assert parameter.dtype == param_dtype


def test_linear_quantizer_quantisation_range_setter() -> None:
    data = torch.rand((32, 14, 17))
    num_bits = 2
    quantizer = LinearQuantizer(num_bits=num_bits, symmetric=False)

    min_data, max_data = torch.min(data), torch.max(data)
    assert quantizer.has_uninitialized_params

    ## Ignoring the mypy errors here to check the error raising.
    with pytest.raises(ValueError):
        quantizer.quantization_range = (min_data, max_data, max_data)  # type: ignore[assignment]

    with pytest.raises(ValueError):
        quantizer.quantization_range = min_data  # type: ignore[assignment]

    quantizer.quantization_range = (torch.min(data), torch.max(data))
    assert not quantizer.has_uninitialized_params


@pytest.mark.parametrize("channel_dim", [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)])
@pytest.mark.parametrize("data_shape", [(3, 4, 9), (32, 14, 17)])
def test_linear_quantizer_quantisation_range_setter_correct_shape_per_channel(
    data_shape: tuple[int, ...], channel_dim: tuple[int, ...]
) -> None:
    data = torch.rand(data_shape)
    num_bits = 2
    quantizer = LinearQuantizer(
        num_bits=num_bits,
        symmetric=False,
        granularity=granularity.PerChannel(channel_dim),
    )

    n_elements = math.prod([data.shape[dim] for dim in channel_dim])

    quantizer.quantization_range = (torch.zeros(n_elements), torch.ones(n_elements))

    # Assertion just to check that the quantizer call does not produce an error
    assert quantizer(data) is not None


@pytest.mark.parametrize("channel_dim", [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)])
@pytest.mark.parametrize("data_shape", [(3, 4, 9), (32, 14, 17)])
def test_linear_quantizer_quantisation_range_setter_incorrect_shape_per_channel(
    data_shape: tuple[int, ...], channel_dim: tuple[int, ...]
) -> None:
    data = torch.rand(data_shape)
    num_bits = 2
    quantizer = LinearQuantizer(
        num_bits=num_bits,
        symmetric=False,
        granularity=granularity.PerChannel(channel_dim),
    )

    n_elements = 2

    quantizer.quantization_range = (torch.zeros(n_elements), torch.ones(n_elements))

    with pytest.raises(RuntimeError):
        quantizer(data)


def test_linear_quantizer_asymmetric_per_tensor() -> None:
    batch_size = 32
    data = torch.linspace(-2, 14, 17)
    data = torch.stack([data] * batch_size)
    scale = torch.tensor(2.0)
    offset = torch.tensor(4.0)
    num_bits = 2

    quantizer = LinearQuantizer(num_bits=num_bits, symmetric=False)
    quantizer.quantization_range = (torch.min(data), torch.max(data))

    assert quantizer.offset is not None

    # Set scale and offset explicitly, i.e., overwrite implicit scale/grad from
    # quantization range for test.
    with torch.no_grad():
        quantizer.scale.fill_(scale)
        quantizer.offset.fill_(offset)

    quant_data = cast(QuantizedTensor, quantizer(data))
    raw_quant_data = quant_data.raw_data
    # fmt: off
    expected_raw = torch.tensor(
        [
            -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0,
            -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0
        ]
    )
    # fmt: on
    expected_raw = torch.stack([expected_raw] * batch_size)
    torch.testing.assert_close(expected_raw, raw_quant_data)


def test_linear_quantizer_asymmetric_per_tensor_gradients() -> None:
    data = torch.linspace(-8, 8, 17)
    num_bits = 3

    quantizer = LinearQuantizer(num_bits, symmetric=False)
    assert quantizer.offset is not None
    # Choose range such that clipping occurs, otherwise offset gradient may be zero.
    quantizer.quantization_range = (-4.0, 9.0)

    data.requires_grad = True

    quant_data = quantizer(data)
    quant_data.retain_grad()

    quant_data.dequantize().sum().backward()
    assert data.grad is not None
    assert quantizer.scale.grad is not None
    assert quantizer.offset.grad is not None

    assert torch.count_nonzero(data.grad)
    assert torch.count_nonzero(quantizer.scale.grad)
    assert torch.count_nonzero(quantizer.offset.grad)


@pytest.mark.parametrize("axis", [0, 1])
def test_linear_quantizer_asymmetric_per_channel(axis: int, _seed_prngs: int) -> None:
    batch_size = 16
    num_features = 32
    num_bits = 3

    data = torch.randn(batch_size, num_features)

    output_collection = []
    for channel_data in data.permute(axis, axis - 1):
        quantizer = LinearQuantizer(num_bits=num_bits, symmetric=False)
        assert quantizer.per_tensor, "quantizer should be per_tensor quantizer"

        quantizer.quantization_range = (torch.min(channel_data), torch.max(channel_data))
        output_collection.append(quantizer(channel_data).dequantize())

    expected_output = torch.stack(output_collection)

    quantizer = LinearQuantizer(
        num_bits=num_bits,
        granularity=granularity.PerChannel(axis),
        symmetric=False,
    )
    assert quantizer.per_channel, "quantizer should be per_channel quantizer"
    quantizer.quantization_range = (
        torch.min(data, axis - 1).values,
        torch.max(data, axis - 1).values,
    )
    output = quantizer(data).dequantize()

    torch.testing.assert_close(output, expected_output.permute(axis, axis - 1))


def test_linear_quantizer_asymmetric_per_channel_gradients() -> None:
    batch_size = 2
    num_features = 3
    num_bits = 3

    data = torch.linspace(-2.0, 2.0, batch_size * num_features).reshape(batch_size, num_features)
    offset = torch.arange(num_features) - num_features / 2
    scale = torch.linspace(0.1, 0.3, num_features)

    quantizer = LinearQuantizer(
        num_bits=num_bits,
        granularity=granularity.PerChannel(-1),
        symmetric=False,
    )
    assert quantizer.offset is not None

    quantizer.quantization_range = (torch.min(data, 0).values, torch.max(data, 0).values)
    with torch.no_grad():
        quantizer.scale.copy_(scale)
        quantizer.offset.copy_(offset)

    data.requires_grad = True

    quant_data = quantizer(data)
    quant_data.backward(torch.ones_like(quant_data))

    assert data.grad is not None
    assert quantizer.scale.grad is not None
    assert quantizer.offset.grad is not None

    assert torch.count_nonzero(data.grad)
    assert torch.count_nonzero(quantizer.scale.grad)
    assert torch.count_nonzero(quantizer.offset.grad)


@pytest.mark.slow
@pytest.mark.parametrize("tile_step", [1, 2, 4, 8])
def test_linear_quantizer_asymmetric_per_tile(tile_step: int, _seed_prngs: int) -> None:
    batch_size = 16
    num_features = 32
    num_bits = 3

    data = torch.randn(batch_size, num_features)
    tile_batch_size = batch_size // tile_step
    tile_num_features = num_features // tile_step
    tile_size = (tile_batch_size, tile_num_features)

    min_values, max_values = [], []

    expected_output = torch.zeros(data.shape)
    for row_idx in range(tile_step):
        for col_idx in range(tile_step):
            start_row, start_col = row_idx * tile_batch_size, col_idx * tile_num_features
            end_row, end_col = (row_idx + 1) * tile_batch_size, (col_idx + 1) * tile_num_features

            sub_data = data[start_row:end_row, start_col:end_col]
            quantizer = LinearQuantizer(num_bits=num_bits, symmetric=False)
            assert quantizer.per_tensor, "quantizer should be per_tensor quantizer"
            curr_min_value, curr_max_value = sub_data.min(), sub_data.max()
            quantizer.quantization_range = (sub_data.min(), sub_data.max())
            min_values.append(curr_min_value)
            max_values.append(curr_max_value)

            output = quantizer(sub_data).dequantize()
            expected_output[start_row:end_row, start_col:end_col] = output

    tile_min_values = torch.stack(min_values)
    tile_max_values = torch.stack(max_values)

    per_tile_granularity = granularity.PerTile(tile_size)
    quantizer = LinearQuantizer(
        num_bits=num_bits, symmetric=False, granularity=per_tile_granularity
    )
    quantizer.quantization_range = (tile_min_values, tile_max_values)
    output = quantizer(data).dequantize()
    torch.testing.assert_close(output, expected_output)


def test_linear_quantizer_asymmetric_per_tile_gradients(_seed_prngs: int) -> None:
    batch_size = 2
    num_features = 4
    num_bits = 3

    tile_size = (batch_size // 2, num_features // 2)
    gran = granularity.PerTile(tile_size)

    data = torch.linspace(-1.5, 1.5, batch_size * num_features).reshape(batch_size, num_features)
    offset = torch.arange(num_features) - num_features / 2
    scale = torch.linspace(0.1, 0.3, num_features)

    num_parameters = gran.parameter_dimensionality(data.shape)

    quantizer = LinearQuantizer(
        num_bits=num_bits,
        granularity=gran,
        symmetric=False,
    )
    assert quantizer.offset is not None
    quantizer.quantization_range = (torch.randn(num_parameters), torch.randn(num_parameters))
    with torch.no_grad():
        quantizer.scale.copy_(scale)
        quantizer.offset.copy_(offset)

    data.requires_grad = True

    quant_data = quantizer(data)
    quant_data.backward(torch.ones_like(quant_data))

    assert data.grad is not None
    assert quantizer.scale.grad is not None
    assert quantizer.offset.grad is not None

    assert torch.count_nonzero(data.grad)
    assert torch.count_nonzero(quantizer.scale.grad)
    assert torch.count_nonzero(quantizer.offset.grad)


def test_linear_quantizer_unsigned_quantizer() -> None:
    quantizer = fastforward.nn.LinearQuantizer(num_bits=4, symmetric=True, allow_one_sided=True)
    assert quantizer.offset is not None
    quantizer.quantization_range = (4.0, 16.0)
    assert quantizer.offset.item() == 8.0
    assert quantizer.symmetric

    quantizer.quantization_range = (-4.0, 16.0)
    assert quantizer.offset.item() == 0.0
    assert quantizer.symmetric

    quantizer = fastforward.nn.LinearQuantizer(num_bits=4, symmetric=True, allow_one_sided=False)
    quantizer.quantization_range = (4.0, 16.0)
    assert quantizer.offset is None
    assert quantizer.symmetric

    quantizer.quantization_range = (-4.0, 16.0)
    assert quantizer.offset is None
    assert quantizer.symmetric
