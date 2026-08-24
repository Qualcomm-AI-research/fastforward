import fastforward
import torch

from tests.autoquant.test_data.collision_alpha.modeling_alpha import (
    AlphaBlock,
    residual_op,
)
from tests.autoquant.test_data.collision_beta.modeling_beta import BetaBlock
from tests.autoquant.test_data.collision_beta.modeling_beta import (
    residual_op as collision_beta_residual_op,
)
from tests.autoquant.test_data.import_collision_model import ImportCollisionModel


class QuantizedImportCollisionModel(fastforward.nn.QuantizedModule, ImportCollisionModel):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _tmp_1 = self.alpha(x)
        return self.beta(_tmp_1)


class QuantizedAlphaBlock(fastforward.nn.QuantizedModule, AlphaBlock):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        op = residual_op
        return op(x, x)


class QuantizedBetaBlock(fastforward.nn.QuantizedModule, BetaBlock):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        op = collision_beta_residual_op
        return op(x, x)
