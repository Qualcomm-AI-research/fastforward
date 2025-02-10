# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from collections.abc import Sequence

import fastforward as ff
import pytest
import torch

from fastforward.nn.linear import QuantizedLinear
from fastforward.nn.linear_quantizer import LinearQuantizer
from fastforward.nn.quantized_module import QuantizedModule
from fastforward.quantization.local_error import Runner
from fastforward.quantization.output_mse import OutputMSE


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.linear1 = QuantizedLinear(20, 40)
        self.linear2 = QuantizedLinear(40, 40)
        self.linear3 = QuantizedLinear(40, 10)

    @staticmethod
    def quantize_linear_layer(layer: torch.nn.Module, num_bits: int) -> None:
        # quant_metadata = layer.weight_quantizer.quant_metadata
        layer.weight_quantizer = LinearQuantizer(num_bits=num_bits)
        # layer.weight_quantizer.quant_metadata = quant_metadata
        layer.weight_quantizer.quantization_range = (
            torch.min(layer.weight),
            torch.max(layer.weight),
        )

    def quantize_model(self, num_bits: int) -> None:
        self.quantize_linear_layer(self.linear1, num_bits)
        self.quantize_linear_layer(self.linear2, num_bits)
        self.quantize_linear_layer(self.linear3, num_bits)

    def get_all_quantizer_parameters(self) -> list[torch.nn.Parameter]:
        all_parameters = []
        for module in self.modules():
            if isinstance(module, QuantizedModule):
                parameters = [
                    param
                    for quantizer in module.quantizers(skip_stubs=True)
                    for param in quantizer.parameters()
                ]
                all_parameters.extend(parameters)
        return all_parameters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

    @staticmethod
    def setup_opt(params: Sequence[torch.Tensor], lr: float) -> torch.optim.Adam:
        opt = torch.optim.Adam(params, lr=lr)
        return opt


@pytest.mark.parametrize("num_parameters", [1, 2, 3])
def test_mse_output_optimizer_updates_quantization_parameters(num_parameters: int) -> None:
    num_bits, learning_rate = 4, 1e-03
    model = TinyModel()

    data = torch.rand(20, 40, 20)

    model.quantize_model(num_bits)
    quant_parameters = model.get_all_quantizer_parameters()
    to_optimize_parameters = quant_parameters[:num_parameters]
    starting_parameters = [param.item() for param in quant_parameters]

    output_mse = OutputMSE(
        [model.linear1, model.linear2, model.linear3],
        model.setup_opt(to_optimize_parameters, lr=learning_rate),
    )

    runner = Runner(target=model, method=output_mse)

    runner(data)
    with ff.strict_quantization(False):
        runner.start()

    optimized_parameters = [param.item() for param in quant_parameters]
    # Any parameters passed to the optimizer should be update (ie,
    # their final values should be different from their starting values)
    assert starting_parameters[:num_parameters] != optimized_parameters[:num_parameters]
    # Any parameters not passed to the optimizer should be the same (ie,
    # their final values should be exactly the same as their starting values)
    assert starting_parameters[num_parameters:] == optimized_parameters[num_parameters:]
