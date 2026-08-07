import fastforward
import torch

from tests.autoquant.test_data.sam3_model import (
    SAM3AlreadyExplicitSuper,
    SAM3ModelInspired,
    SAM3SuperChild,
    SAM3SuperParent,
)


class QuantizedSAM3ModelInspired(fastforward.nn.QuantizedModule, SAM3ModelInspired):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.super_child(child_kw=x)
        _ = self.explicit_super(child_kw=x)
        return x


class QuantizedSAM3SuperParent(fastforward.nn.QuantizedModule, SAM3SuperParent):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()

    def forward(self, *, parent_kw):
        return parent_kw


class QuantizedSAM3SuperChild(QuantizedSAM3SuperParent, SAM3SuperChild):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()

    def forward(self, *, child_kw):
        return super(QuantizedSAM3SuperChild, self).forward(parent_kw=child_kw)


class QuantizedSAM3AlreadyExplicitSuper(QuantizedSAM3SuperParent, SAM3AlreadyExplicitSuper):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()

    def forward(self, *, child_kw):
        return super(QuantizedSAM3AlreadyExplicitSuper, self).forward(parent_kw=child_kw)
