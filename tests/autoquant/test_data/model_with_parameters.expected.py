import fastforward
import torch

from tests.autoquant.test_data.model_with_parameters import ModelWithParameters


class QuantizedModelWithParameters(fastforward.nn.QuantizedModule, ModelWithParameters):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.quantizer_linear_1: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_linear_2: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_linear_3: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_self_bias: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_self_alt_bias: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_self_weight: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_input: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()

    def forward(self, input: torch.Tensor, aux: int) -> torch.Tensor:
        """Example that uses parameters.

        This example uses parameters that are defined on the instance. The
        parameters are expected to be quantized only if they are used. In this
        example, self.weight is used in every branch and can be quantized once
        in the top-level branch. For bias and alt_bias, however, this is not
        the case as they are only used in the true branch of the first and
        second if statement.
        """
        input = self.quantizer_input(input)
        self_weight = self.quantizer_self_weight(self.weight)
        if aux < 10:
            self_bias = self.quantizer_self_bias(self.bias)
            # normal weight and bias
            return fastforward.nn.functional.linear(
                input, self_weight, self_bias, output_quantizer=self.quantizer_linear_1
            )
        if aux < 20:
            self_alt_bias = self.quantizer_self_alt_bias(self.alt_bias)
            # normal weight alternative bias
            return fastforward.nn.functional.linear(
                input, self_weight, self_alt_bias, output_quantizer=self.quantizer_linear_2
            )
        else:
            # no bias
            return fastforward.nn.functional.linear(
                input, self_weight, output_quantizer=self.quantizer_linear_3
            )