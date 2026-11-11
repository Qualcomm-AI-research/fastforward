# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

# pylint: disable=missing-function-docstring

from typing import Iterator

import fastforward
import pytest
import torch

from fastforward.nn.linear_quantizer import LinearQuantizer
from fastforward.quantization import granularity


@pytest.fixture(autouse=True)
def _reset_dynamo() -> Iterator[None]:
    """Reset dynamo before and after each test."""
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


pytestmark = [
    # Torch<=2.13 instantiate autograd.Function classes while dynamo traces them, emitting a DeprecationWarning.
    pytest.mark.filterwarnings("ignore:.*should not be instantiated:DeprecationWarning"),
    # Importing torch._inductor on torch 2.11+ emits the torch.jit.script_method DeprecationWarning.
    pytest.mark.filterwarnings(
        "ignore:.*torch.jit.script_method. is deprecated:DeprecationWarning"
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize(
    "quant_granularity",
    [granularity.PerTensor(), granularity.PerChannel(0), granularity.PerTile((2, 4))],
    ids=["per_tensor", "per_channel", "per_tile"],
)
def test_linear_quantizer_compiles_without_graph_breaks(
    quant_granularity: granularity.Granularity,
) -> None:
    # GIVEN an initialized quantizer
    quantizer = LinearQuantizer(num_bits=8, granularity=quant_granularity)
    data = torch.randn(4, 8, requires_grad=True)
    with fastforward.estimate_ranges(quantizer, fastforward.range_setting.running_minmax):
        quantizer(data)

    def quantize(x: torch.Tensor) -> torch.Tensor:
        with fastforward.strict_quantization(False):
            return quantizer(x)

    # WHEN it is compiled as a single graph in qdq mode
    compiled = torch.compile(quantize, fullgraph=True, backend="eager")
    with fastforward.qdq_mode(True):
        actual = compiled(data)
        expected = quantize(data)
    actual.sum().backward()

    # THEN it matches eager and propagates gradients
    torch.testing.assert_close(actual, expected)
    assert data.grad is not None


@pytest.mark.slow
@pytest.mark.parametrize(
    "quant_granularity",
    [granularity.PerTensor(), granularity.PerChannel(2), granularity.PerChannel((1, 2))],
    ids=["per_tensor", "per_channel", "per_channel_multi_dim"],
)
def test_linear_quantizer_compiles_fullgraph_with_changing_batch_size(
    quant_granularity: granularity.Granularity,
) -> None:
    # GIVEN an initialized quantizer
    quantizer = LinearQuantizer(num_bits=8, granularity=quant_granularity)
    with fastforward.estimate_ranges(quantizer, fastforward.range_setting.running_minmax):
        quantizer(torch.randn(2, 5, 8))

    def quantize(x: torch.Tensor) -> torch.Tensor:
        with fastforward.strict_quantization(False):
            return quantizer(x)

    compiled = torch.compile(quantize, fullgraph=True, backend="eager")

    # WHEN the compiled function sees different batch sizes (e.g. a smaller last batch),
    # which makes dynamo recompile with symbolic sizes
    with fastforward.qdq_mode(True):
        for batch_size in (4, 3, 7):
            data = torch.randn(batch_size, 5, 8)

            # THEN it matches the eager result for every batch size
            torch.testing.assert_close(compiled(data), quantize(data))


@pytest.mark.slow
@pytest.mark.parametrize(
    "quant_granularity",
    [granularity.PerTensor(), granularity.PerChannel(2), granularity.PerChannel((1, 2))],
    ids=["per_tensor", "per_channel", "per_channel_multi_dim"],
)
def test_linear_quantizer_compiles_with_changing_batch_size(
    quant_granularity: granularity.Granularity,
) -> None:
    # GIVEN an initialized quantizer
    quantizer = LinearQuantizer(num_bits=8, granularity=quant_granularity)
    with fastforward.estimate_ranges(quantizer, fastforward.range_setting.running_minmax):
        quantizer(torch.randn(2, 5, 8))

    def quantize(x: torch.Tensor) -> torch.Tensor:
        with fastforward.strict_quantization(False):
            return quantizer(x)

    compiled = torch.compile(quantize, fullgraph=False, backend="eager")

    # WHEN the compiled function sees different batch sizes (e.g. a smaller last batch),
    # which makes dynamo recompile with symbolic sizes
    with fastforward.qdq_mode(True):
        for batch_size in (4, 3, 7):
            data = torch.randn(batch_size, 5, 8)

            # THEN it matches the eager result for every batch size
            torch.testing.assert_close(compiled(data), quantize(data))
