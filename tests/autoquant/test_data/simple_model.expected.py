import fastforward
import torch

from tests.autoquant.test_data.simple_model import (
    AutoquantClassifier,
    AutoquantTestBlock,
    AutoquantTestModel,
)


class QuantizedAutoquantTestModel(fastforward.nn.QuantizedModule, AutoquantTestModel):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expected input shape: (B, 3, 128, 128)."""
        h = self.upcast(x)  # Transform to 'compute shape'
        h = self.l1(h)
        h = self.l2(h)

        # 'Pooling' and downcast
        h = self.downcast(h)

        output = self.classifier(h)

        return output


class QuantizedAutoquantTestBlock(fastforward.nn.QuantizedModule, AutoquantTestBlock):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.quantizer_relu: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_add: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_sigmoid: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_h_1: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_h_2: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_x: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quantizer_x(x)
        h = self.l1(x)
        h = self.quantizer_h_1(h)
        h = fastforward.nn.functional.relu(h, output_quantizer=self.quantizer_relu)
        h = self.l2(h)
        h = self.quantizer_h_2(h)

        # Residual
        h = fastforward.nn.functional.add(h, x, output_quantizer=self.quantizer_add)

        # Conditional activation
        if self.activate_output:
            h = fastforward.nn.functional.sigmoid(h, output_quantizer=self.quantizer_sigmoid)

        return h


class QuantizedAutoquantClassifier(fastforward.nn.QuantizedModule, AutoquantClassifier):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.quantizer_softmax: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_h: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transform from 3d to 1d representation.
        h = x.reshape((-1, self.num_features))
        h = self.l1(h)
        h = self.quantizer_h(h)

        # Compute probabilities
        h = fastforward.nn.functional.softmax(h, 1, output_quantizer=self.quantizer_softmax)
        return h
