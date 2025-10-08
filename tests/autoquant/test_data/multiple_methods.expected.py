import fastforward
import torch

from tests.autoquant.test_data.multiple_methods import (
    ModuleWithMultipleMethods,
    helper_no_quant,
)


def quantized_helper(
    x: torch.Tensor,
    *,
    quantizer_mul: fastforward.nn.Quantizer,
    quantizer_x: fastforward.nn.Quantizer,
) -> torch.Tensor:
    x = quantizer_x(x)
    return fastforward.nn.functional.mul(x, 2, output_quantizer=quantizer_mul)


class QuantizedModuleWithMultipleMethods(fastforward.nn.QuantizedModule, ModuleWithMultipleMethods):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.quantizer__forward_cls_add: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer__forward_cls__tmp_1: fastforward.nn.Quantizer = (
            fastforward.nn.QuantizerStub()
        )
        self.quantizer__forward_cls__tmp_2: fastforward.nn.Quantizer = (
            fastforward.nn.QuantizerStub()
        )
        self.quantizer__forward_cls__forward_static_1_mul: fastforward.nn.Quantizer = (
            fastforward.nn.QuantizerStub()
        )
        self.quantizer__forward_cls__forward_static_1_x: fastforward.nn.Quantizer = (
            fastforward.nn.QuantizerStub()
        )
        self.quantizer__forward_cls__forward_static_2_helper_mul: fastforward.nn.Quantizer = (
            fastforward.nn.QuantizerStub()
        )
        self.quantizer__forward_cls__forward_static_2_helper_x: fastforward.nn.Quantizer = (
            fastforward.nn.QuantizerStub()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward(x)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_cls(
            x,
            quantizer_add=self.quantizer__forward_cls_add,
            quantizer__tmp_1=self.quantizer__forward_cls__tmp_1,
            quantizer__tmp_2=self.quantizer__forward_cls__tmp_2,
            quantizer__forward_static_1_mul=self.quantizer__forward_cls__forward_static_1_mul,
            quantizer__forward_static_1_x=self.quantizer__forward_cls__forward_static_1_x,
            quantizer__forward_static_2_helper_mul=self.quantizer__forward_cls__forward_static_2_helper_mul,
            quantizer__forward_static_2_helper_x=self.quantizer__forward_cls__forward_static_2_helper_x,
        )

    @classmethod
    def _forward_cls(
        cls,
        x: torch.Tensor,
        *,
        quantizer_add: fastforward.nn.Quantizer,
        quantizer__tmp_1: fastforward.nn.Quantizer,
        quantizer__tmp_2: fastforward.nn.Quantizer,
        quantizer__forward_static_1_mul: fastforward.nn.Quantizer,
        quantizer__forward_static_1_x: fastforward.nn.Quantizer,
        quantizer__forward_static_2_helper_mul: fastforward.nn.Quantizer,
        quantizer__forward_static_2_helper_x: fastforward.nn.Quantizer,
    ) -> torch.Tensor:
        _tmp_1 = cls._forward_static_1(
            x,
            quantizer_mul=quantizer__forward_static_1_mul,
            quantizer_x=quantizer__forward_static_1_x,
        )
        _tmp_1 = quantizer__tmp_1(_tmp_1)
        _tmp_2 = QuantizedModuleWithMultipleMethods._forward_static_2(
            x,
            quantizer_helper_mul=quantizer__forward_static_2_helper_mul,
            quantizer_helper_x=quantizer__forward_static_2_helper_x,
        )
        _tmp_2 = quantizer__tmp_2(_tmp_2)
        return fastforward.nn.functional.add(_tmp_1, _tmp_2, output_quantizer=quantizer_add)

    @staticmethod
    def _forward_static_1(
        x: torch.Tensor,
        *,
        quantizer_mul: fastforward.nn.Quantizer,
        quantizer_x: fastforward.nn.Quantizer,
    ) -> torch.Tensor:
        x = quantizer_x(x)
        helper_no_quant(x)
        return fastforward.nn.functional.mul(x, 2, output_quantizer=quantizer_mul)

    @staticmethod
    def _forward_static_2(
        x: torch.Tensor,
        *,
        quantizer_helper_mul: fastforward.nn.Quantizer,
        quantizer_helper_x: fastforward.nn.Quantizer,
    ) -> torch.Tensor:
        return quantized_helper(
            x, quantizer_mul=quantizer_helper_mul, quantizer_x=quantizer_helper_x
        )
