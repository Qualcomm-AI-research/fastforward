import fastforward
import torch

from tests.autoquant.test_data.gr00t_model import Gr00tModelInspited, Gr00tPadBlock


def quantized_get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    *_: object,
    quantizer_remainder: fastforward.nn.Quantizer,
    quantizer_pad: fastforward.nn.Quantizer,
    quantizer_emb: fastforward.nn.Quantizer,
    quantizer_embedding_dim: fastforward.nn.Quantizer,
) -> torch.Tensor:
    embedding_dim = quantizer_embedding_dim(embedding_dim)
    emb = timesteps[:, None].float()
    emb = quantizer_emb(emb)
    if (
        fastforward.nn.functional.remainder(embedding_dim, 2, output_quantizer=quantizer_remainder)
        == 1
    ):
        emb = fastforward.nn.functional.pad(emb, (0, 1, 0, 0), output_quantizer=quantizer_pad)
    return emb


class QuantizedGr00tModelInspited(fastforward.nn.QuantizedModule, Gr00tModelInspited):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x)
        return x


class QuantizedGr00tPadBlock(fastforward.nn.QuantizedModule, Gr00tPadBlock):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.quantizer_get_timestep_embedding_remainder: fastforward.nn.Quantizer = (
            fastforward.nn.QuantizerStub()
        )
        self.quantizer_get_timestep_embedding_pad: fastforward.nn.Quantizer = (
            fastforward.nn.QuantizerStub()
        )
        self.quantizer_get_timestep_embedding_emb: fastforward.nn.Quantizer = (
            fastforward.nn.QuantizerStub()
        )
        self.quantizer_get_timestep_embedding_embedding_dim: fastforward.nn.Quantizer = (
            fastforward.nn.QuantizerStub()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _tmp_2 = x.flatten()
        timesteps = _tmp_2.to(dtype=torch.float32)
        return quantized_get_timestep_embedding(
            timesteps,
            257,
            quantizer_remainder=self.quantizer_get_timestep_embedding_remainder,
            quantizer_pad=self.quantizer_get_timestep_embedding_pad,
            quantizer_emb=self.quantizer_get_timestep_embedding_emb,
            quantizer_embedding_dim=self.quantizer_get_timestep_embedding_embedding_dim,
        )

