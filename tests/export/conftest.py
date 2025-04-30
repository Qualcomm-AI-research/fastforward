# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


import fastforward as ff
import pytest
import torch

from tests.export.export_utils import QuantizedModelFixture


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
