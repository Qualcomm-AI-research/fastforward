import torch


class ModuleWithMultipleMethods(torch.nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward(x)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_cls(x)

    @classmethod
    def _forward_cls(cls, x: torch.Tensor) -> torch.Tensor:
        return cls._forward_static_1(x) + cls._forward_static_2(x)

    @staticmethod
    def _forward_static_1(x: torch.Tensor) -> torch.Tensor:
        return x * 2

    @staticmethod
    def _forward_static_2(x: torch.Tensor) -> torch.Tensor:
        return x / 2

def get_model() -> torch.nn.Module:
    return ModuleWithMultipleMethods()
