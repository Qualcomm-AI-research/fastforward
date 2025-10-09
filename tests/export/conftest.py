# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


from typing import Any, TypeAlias

import fastforward as ff
import pytest
import torch

from fastforward.quantization.quant_init import QuantizerCollection

QuantizedModelFixture: TypeAlias = tuple[torch.nn.Module, QuantizerCollection, QuantizerCollection]


@pytest.fixture
def simple_model() -> QuantizedModelFixture:
    class FFNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = ff.nn.QuantizedLinear(10, 10)
            self.relu1 = ff.nn.QuantizedRelu()
            self.fc2 = ff.nn.QuantizedLinear(10, 10)
            self.relu2 = ff.nn.QuantizedRelu()
            self.fc3 = ff.nn.QuantizedLinear(10, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.fc3(x)

            return x

    quant_model = FFNet()

    activation_quantizers = ff.find_quantizers(quant_model, "**/[quantizer:activation/output]")
    activation_quantizers |= ff.find_quantizers(quant_model, "fc1/[quantizer:activation/input]")
    parameter_quantizers = ff.find_quantizers(quant_model, "**/[quantizer:parameter]")

    return quant_model, activation_quantizers, parameter_quantizers


@pytest.fixture
def model_with_kwargs() -> QuantizedModelFixture:
    class CustomQuantMul(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.output_quantizer = ff.nn.QuantizerStub(output_quantizer=True)

        def forward(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
            mul_output = ff.nn.functional.mul(
                input=input, other=other, output_quantizer=self.output_quantizer
            )

            return mul_output

    class SimpleModelWithKwargs(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first_input_quantizer = ff.nn.QuantizerStub(input_quantizer=True)
            self.first_kwarg_quantizer = ff.nn.QuantizerStub(input_quantizer=True)
            self.second_kwarg_quantizer = ff.nn.QuantizerStub(input_quantizer=True)
            self.add_quantizer = ff.nn.QuantizerStub(output_quantizer=True)

            self.fc1 = ff.nn.QuantizedLinear(10, 10)
            self.mul_module = CustomQuantMul()

        def forward(self, input: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            quantized_input = self.first_input_quantizer(input)
            first_kwarg: torch.Tensor = self.first_kwarg_quantizer(kwargs["first_kwarg"])
            second_kwarg: torch.Tensor = self.second_kwarg_quantizer(kwargs["second_kwarg"])

            added_kwargs = ff.nn.functional.add(
                input=first_kwarg, other=second_kwarg, output_quantizer=self.add_quantizer
            )
            quantized_linear_input = self.fc1(quantized_input)
            mul_output: torch.Tensor = self.mul_module(quantized_linear_input, other=added_kwargs)

            return mul_output

    # Create a simple model - no quantization to avoid conflicts
    quant_model = SimpleModelWithKwargs()

    # Return empty quantizer collections since this is a simple test model
    activation_quantizers = ff.find_quantizers(quant_model, "**/[quantizer:activation/input]")
    activation_quantizers |= ff.find_quantizers(quant_model, "**/[quantizer:activation/output]")
    activation_quantizers -= ff.find_quantizers(quant_model, "fc1/[quantizer:activation/input]")
    parameter_quantizers = ff.find_quantizers(quant_model, "**/[quantizer:parameter]")

    return quant_model, activation_quantizers, parameter_quantizers


@pytest.fixture
def multi_input_output_model() -> QuantizedModelFixture:
    class FFNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = ff.nn.QuantizedLinear(10, 10)
            self.fc2 = ff.nn.QuantizedLinear(10, 10)

        def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            add_to_y: torch.Tensor,
            subtract_from_x: torch.Tensor,
        ) -> tuple[torch.Tensor, ...]:
            x = torch.subtract(x, subtract_from_x)
            y = torch.add(y, add_to_y)
            add_x_y = torch.add(x, y)
            linear_x = self.fc1(x)
            linear_y = self.fc2(y)

            return add_x_y, linear_x, linear_y

    quant_model = FFNet()

    activation_quantizers = ff.find_quantizers(quant_model, "**/[quantizer:activation/output]")
    activation_quantizers |= ff.find_quantizers(quant_model, "fc1/[quantizer:activation/input]")
    parameter_quantizers = ff.find_quantizers(quant_model, "**/[quantizer:parameter]")

    return quant_model, activation_quantizers, parameter_quantizers


@pytest.fixture
def simple_quant_model_with_non_quant_ops() -> QuantizedModelFixture:
    class FFNet(torch.nn.Module):
        """Simple FF model with quantized linear/relu modules."""

        def __init__(self) -> None:
            super().__init__()
            net_in_out_dim = 10
            self.fc1 = ff.nn.QuantizedLinear(net_in_out_dim, net_in_out_dim, bias=False)
            self.relu1 = ff.nn.QuantizedRelu()
            self.fc2 = ff.nn.QuantizedLinear(net_in_out_dim, net_in_out_dim, bias=False)
            self.relu2 = ff.nn.QuantizedRelu()
            self.fc3 = ff.nn.QuantizedLinear(net_in_out_dim, net_in_out_dim, bias=False)

            self.extra_weight = torch.nn.Parameter(
                torch.rand(size=(net_in_out_dim, net_in_out_dim))
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.fc1(x)
            x = torch.reshape(x, (x.shape[1], x.shape[0]))
            x = torch.reshape(x, (x.shape[1], x.shape[0]))
            x = self.relu1(x)
            x = self.fc2(x)
            x = torch.nn.functional.softmax(x, dim=0)
            x = self.relu2(x)
            x = self.fc3(x)
            x = torch.matmul(x, self.extra_weight)

            return x

    quant_model = FFNet()

    activation_quantizers = ff.find_quantizers(quant_model, "**/[quantizer:activation/output]")
    activation_quantizers |= ff.find_quantizers(quant_model, "fc1/[quantizer:activation/input]")
    parameter_quantizers = ff.find_quantizers(quant_model, "**/[quantizer:parameter]")

    return quant_model, activation_quantizers, parameter_quantizers
