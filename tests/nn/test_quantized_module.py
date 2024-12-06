# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

# pylint: disable=missing-function-docstring
from copy import deepcopy

import pytest
import torch

import fastforward as ff

from fastforward.nn import QuantizerMetadata, QuantizerStub
from fastforward.nn.quantized_module import SKIP_QUANTIZATION


class _QuantizerSubclass(ff.nn.Quantizer):
    pass


class _MockQuantizedModule(ff.nn.QuantizedModule):
    def __init__(self):
        super().__init__()
        self.other_module1 = torch.nn.Linear(10, 10)
        self.other_module3 = torch.nn.Linear(10, 10)
        self.other_module3 = torch.nn.Linear(10, 10)

    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.quantizer1: ff.nn.Quantizer = QuantizerStub(input_quantizer=True)
        self.quantizer2: ff.nn.Quantizer = QuantizerStub(output_quantizer=True)
        self.quantizer3: ff.nn.Quantizer = QuantizerStub(bias_quantizer=True)
        self.quantizer4: ff.nn.Quantizer = QuantizerStub(weight_quantizer=True)


def test_quantizer_module_named_quantizers():
    module = _MockQuantizedModule()
    quantizer_names = {"quantizer1", "quantizer2", "quantizer3", "quantizer4"}
    quantizers = {module.quantizer1, module.quantizer2, module.quantizer3, module.quantizer4}

    assert len(list(module.named_quantizers())) == 0

    for name, quantizer in module.named_quantizers(skip_stubs=False):
        assert name in quantizer_names
        assert quantizer in quantizers
        quantizer_names.remove(name)
        quantizers.remove(quantizer)
    assert len(quantizer_names) == 0
    assert len(quantizers) == 0


def test_quantizer_module_named_quantizers_after_replace():
    module = _MockQuantizedModule()
    module.quantizer1 = ff.nn.Quantizer()
    module.quantizer2 = ff.nn.Quantizer()
    module.quantizer3 = ff.nn.Quantizer()
    module.quantizer4 = ff.nn.Quantizer()
    quantizer_names = {"quantizer1", "quantizer2", "quantizer3", "quantizer4"}
    quantizers = {module.quantizer1, module.quantizer2, module.quantizer3, module.quantizer4}

    for name, quantizer in module.named_quantizers():
        assert name in quantizer_names
        assert quantizer in quantizers
        quantizer_names.remove(name)
        quantizers.remove(quantizer)

    assert len(quantizer_names) == 0
    assert len(quantizers) == 0


def test_quantizer_module_metadata():
    module = _MockQuantizedModule()

    assert QuantizerMetadata(input_quantizer=True).is_extension(
        module._quantizer_metadata["quantizer1"]
    )
    assert QuantizerMetadata(output_quantizer=True).is_extension(
        module._quantizer_metadata["quantizer2"]
    )
    assert QuantizerMetadata(bias_quantizer=True).is_extension(
        module._quantizer_metadata["quantizer3"]
    )
    assert QuantizerMetadata(weight_quantizer=True).is_extension(
        module._quantizer_metadata["quantizer4"]
    )


def test_quantizer_module_quantizers():
    module = _MockQuantizedModule()
    quantizers = {module.quantizer1, module.quantizer2, module.quantizer3, module.quantizer4}

    for quantizer in module.quantizers(skip_stubs=False):
        assert quantizer in quantizers
        quantizers.remove(quantizer)

    assert len(quantizers) == 0

    module.quantizer2 = ff.nn.Quantizer()
    module.quantizer3 = ff.nn.Quantizer()
    quantizers = {module.quantizer2, module.quantizer3}

    for quantizer in module.quantizers():
        assert quantizer in quantizers
        quantizers.remove(quantizer)

    assert len(quantizers) == 0


class MyModule1(torch.nn.Module):
    pass


class MyModule2(MyModule1):
    pass


class MyModule3(MyModule2):
    pass


class MyUnquantizableModule4(MyModule3):
    pass


class MyQuantizedModule1(ff.nn.QuantizedModule, MyModule1):
    pass


class MyQuantizedModule2(ff.nn.QuantizedModule, MyModule2):
    pass


class MyQuantizedModule3(ff.nn.QuantizedModule, MyModule3):
    pass


def test_quantized_module_map():
    mapping = ff.nn.quantized_module_map()
    assert mapping[torch.nn.Linear] == ff.nn.QuantizedLinear
    assert mapping[torch.nn.Conv2d] == ff.nn.QuantizedConv2d
    assert mapping[torch.nn.Sequential] == ff.nn.QuantizedSequential

    assert mapping[MyModule1] == MyQuantizedModule1
    assert mapping[MyModule2] == MyQuantizedModule2
    assert mapping[MyModule3] == MyQuantizedModule3
    assert MyUnquantizableModule4 not in mapping
    assert ff.nn.activations.QuantizedActivation not in mapping

    for from_cls, to_cls in mapping.items():
        assert issubclass(from_cls, torch.nn.Module)
        assert issubclass(to_cls, torch.nn.Module)
        assert issubclass(to_cls, from_cls)

        assert not issubclass(from_cls, ff.nn.QuantizedModule)
        assert issubclass(to_cls, ff.nn.QuantizedModule)


def _quantizable_model() -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Linear(3, 3),
        MyModule1(),
        torch.nn.Conv2d(3, 3, 3),
        torch.nn.ModuleList([torch.nn.Embedding(3, 3)]),
        torch.nn.ReLU(),
    )


def test_quantize_model():
    quantizable_model = _quantizable_model()
    quantized_model = deepcopy(quantizable_model)
    ff.nn.quantize_model(quantized_model)

    quantizable_model_modules = list(quantizable_model.modules())
    quantized_model_modules = list(quantized_model.modules())
    quantized_model_modules = [
        module for module in quantized_model_modules if not isinstance(module, ff.nn.Quantizer)
    ]

    assert len(quantizable_model_modules) == len(quantized_model_modules)

    assert not any(
        isinstance(module, ff.nn.QuantizedModule) for module in quantizable_model_modules
    )
    assert all(isinstance(module, ff.nn.QuantizedModule) for module in quantized_model_modules)

    unquantizable_model = torch.nn.Sequential(
        torch.nn.Linear(3, 3),
        MyUnquantizableModule4(),
        torch.nn.Conv2d(3, 3, 3),
        torch.nn.ReLU(),
    )

    with pytest.raises(ff.exceptions.QuantizationError):
        ff.nn.quantize_model(unquantizable_model)


def test_quantize_model_skip_quantized():
    # Create a model and quantize a submodule. Then try to quantize again. This should fail
    quantizable_model = _quantizable_model()
    ff.nn.quantize_model(quantizable_model[0])
    with pytest.raises(ff.exceptions.QuantizationError):
        ff.nn.quantize_model(quantizable_model)

    # Create a model and quantize a submodule. Then try to quantize again with
    # skip_quantized_modules=True. This should succeed.
    quantizable_model = _quantizable_model()
    ff.nn.quantize_model(quantizable_model[0])
    ff.nn.quantize_model(quantizable_model, skip_quantized_modules=True)


def test_quantize_model_skip_quantized_module_flag():
    quantizable_model = _quantizable_model()
    ff.nn.quantize_model(quantizable_model, extra_conversion={torch.nn.Linear: SKIP_QUANTIZATION})
    assert quantizable_model[0].__class__ is torch.nn.Linear


def test_surrogate_quantized_modules():
    model = torch.nn.Sequential(MyUnquantizableModule4(), MyUnquantizableModule4(), MyModule1())

    module_map = ff.surrogate_quantized_modules(model)

    # Only a surrogate for MyUnquantizableModule4 should be included
    assert len(module_map) == 1
    assert MyUnquantizableModule4 in module_map
    assert MyModule1 not in module_map
    assert torch.nn.Sequential not in module_map
    assert torch.nn.Module not in module_map

    # The surrogates must be a subclass of the original model and QuantizedModule
    for module_type, quantized_module_type in module_map.items():
        assert not isinstance(quantized_module_type, ff.nn.quantized_module.SkipQuantization)
        assert issubclass(quantized_module_type, module_type)
        assert issubclass(quantized_module_type, ff.nn.QuantizedModule)
        assert quantized_module_type.__name__.endswith("Surrogate")

    # Any surrogate quantized module that was created should not
    # appear in the default module map
    default_module_map = ff.nn.quantized_module.quantized_module_map()
    assert module_map[MyUnquantizableModule4] not in default_module_map
