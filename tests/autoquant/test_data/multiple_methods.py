import torch

def helper(x: torch.Tensor) -> torch.Tensor:
    return x * 2

def helper_no_quant(x: torch.Tensor) -> torch.Tensor:
    """This function should not be quantized"""
    return x

class ModuleWithMultipleMethods(torch.nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward(x)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_cls(x)

    @classmethod
    def _forward_cls(cls, x: torch.Tensor) -> torch.Tensor:
        return cls._forward_static_1(x) + ModuleWithMultipleMethods._forward_static_2(x)

    @staticmethod
    def _forward_static_1(x: torch.Tensor) -> torch.Tensor:
        helper_no_quant(x)
        return x * 2

    @staticmethod
    def _forward_static_2(x: torch.Tensor) -> torch.Tensor:
        return helper(x)

def get_model() -> torch.nn.Module:
    return ModuleWithMultipleMethods()
