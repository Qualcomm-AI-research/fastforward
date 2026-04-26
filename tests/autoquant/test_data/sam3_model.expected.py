import fastforward
import torch

from tests.autoquant.test_data.sam3_model import SAM3ModelInspired


class QuantizedSAM3ModelInspired(fastforward.nn.QuantizedModule, SAM3ModelInspired):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
