# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Any

import pytest
import torch

from fastforward import get_strict_quantization, strict_quantization, strict_quantization_for_module
from fastforward.exceptions import QuantizationError
from fastforward.quantization.random import random_quantized


def test_strict_quantization() -> None:
    a = random_quantized((5, 1, 3))
    b = random_quantized((1, 2, 1), scale=0.2, offset=3)

    # Perform test within this with block such that the
    # original state is reset at the end of the test
    with strict_quantization(True):
        assert get_strict_quantization()

        with strict_quantization(False):
            assert not get_strict_quantization()
        assert get_strict_quantization()

        strict_quantization(False)
        assert not get_strict_quantization()

        strict_quantization(True)
        assert get_strict_quantization()

        with strict_quantization(True):
            with pytest.raises(QuantizationError):
                _ = a + b
        with strict_quantization(False):
            _ = a + b

        strict_quantization(True)
        with pytest.raises(QuantizationError):
            _ = a + b

        strict_quantization(False)
        _ = a + b


def test_module_strict_quantization(_seed_prngs: int) -> None:
    strict_results: list[bool] = []

    class MockModule(torch.nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            strict_results.append(get_strict_quantization())
            return input

    class MockNetwork(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer1 = MockModule()
            self.layer2 = MockModule()
            self.layer3 = MockModule()
            self.layer4 = MockModule()
            self.layer5 = MockModule()
            self.layer6 = MockModule()

        def forward(self, input: torch.Tensor) -> Any:
            h = self.layer1(input)
            h = self.layer2(h)
            h = self.layer3(h)
            h = self.layer4(h)
            h = self.layer5(h)
            h = self.layer6(h)
            return h

    network = MockNetwork()

    with strict_quantization_for_module(False, network.layer3, network.layer5):
        network(torch.randn(3, 3))

    assert strict_results == [True, True, False, True, False, True]
    strict_results = []

    network(torch.randn(3, 3))
    assert strict_results == [True, True, True, True, True, True]
    strict_results = []

    handle = strict_quantization_for_module(False, network.layer1, network.layer2, network.layer3)
    network(torch.randn(3, 3))
    assert strict_results == [False, False, False, True, True, True]
    strict_results = []
    handle.remove()

    network(torch.randn(3, 3))
    assert strict_results == [True, True, True, True, True, True]
