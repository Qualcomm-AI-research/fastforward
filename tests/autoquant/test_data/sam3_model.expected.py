import fastforward
import torch

from tests.autoquant.test_data.sam3_model import (
    SAM3AlreadyExplicitSuper,
    SAM3ModelInspired,
    SAM3SuperChild,
)


class QuantizedSAM3ModelInspired(fastforward.nn.QuantizedModule, SAM3ModelInspired):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.super_child(child_kw=x)
        _ = self.explicit_super(child_kw=x)
        return x


class QuantizedSAM3SuperChild(fastforward.nn.QuantizedModule, SAM3SuperChild):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()

    def forward(self, *, child_kw):
        return super(SAM3SuperChild, self).forward(parent_kw=child_kw)


class QuantizedSAM3AlreadyExplicitSuper(fastforward.nn.QuantizedModule, SAM3AlreadyExplicitSuper):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()

    def forward(self, *, child_kw):
        return super(SAM3AlreadyExplicitSuper, self).forward(parent_kw=child_kw)
