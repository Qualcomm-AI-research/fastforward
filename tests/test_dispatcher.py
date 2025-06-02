# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import unittest

from typing import Any, Callable

import fastforward
import pytest
import torch

from fastforward.dispatcher import _DISPATCHER, Predicate, dispatch, register
from fastforward.quantization.random import random_quantized
from fastforward.quantized_tensor import QuantizedTensor

_ORIGINAL_DISPATCHER = _DISPATCHER.copy()


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    for k in _DISPATCHER:
        _DISPATCHER[k] = []

    for k, v in _ORIGINAL_DISPATCHER.items():
        _DISPATCHER[k] = v


def make_mock(fn: Callable[..., Any]) -> Any:
    return unittest.mock.Mock(spec_set=fn, wraps=fn)


def test_predicate() -> None:
    p = Predicate(lambda x: "test" == x)
    assert p("test")
    assert not p("not test")


def test_predicate_not() -> None:
    p = ~Predicate(lambda x: "test" == x)
    assert not p("test")
    assert p("not test")


def test_predicate_and() -> None:
    m1 = make_mock(lambda x, y: "test" == x)
    m2 = make_mock(lambda x, y: "not" == y)
    p = Predicate(m1) & Predicate(m2)
    assert p("test", "not")
    m1.assert_called_once()
    m2.assert_called_once()

    m1.reset_mock()
    m2.reset_mock()
    assert not p("not", "not")
    m1.assert_called_once()
    m2.assert_not_called()


def test_predicate_or() -> None:
    m1 = make_mock(lambda x, y: "test" == x)
    m2 = make_mock(lambda x, y: "not" == y)
    p = Predicate(m1) | Predicate(m2)
    assert p("test", "not")
    m1.assert_called_once()
    m2.assert_not_called()

    m1.reset_mock()
    m2.reset_mock()
    assert not p("not", "test")
    m1.assert_called_once()
    m2.assert_called_once()


def test_dispatch(_seed_prngs: int) -> None:
    predicate = make_mock(lambda input, dim: input.dtype == torch.int8)
    kernel = make_mock(lambda input, dim: torch.softmax(input.to(torch.float), dim))
    register("softmax", Predicate(predicate), kernel)

    input = torch.randint(low=0, high=10, size=(5, 16), device="cuda", dtype=torch.int8)
    dim = 1

    kernel = dispatch("softmax", input, dim)
    assert kernel is not None

    torch.testing.assert_close(kernel(input, dim), torch.softmax(input.to(torch.float), dim))
    predicate.assert_called_once()
    kernel.assert_called_once()  # type: ignore[attr-defined]


def test_dispatch_order(_seed_prngs: int) -> None:
    p1 = make_mock(lambda input, dim: input.dtype == torch.int8)
    k1 = make_mock(lambda input, dim: torch.softmax(input.to(torch.float), dim))
    register("softmax", Predicate(p1), k1)
    p2 = make_mock(lambda input, dim: dim == 1)
    k2 = make_mock(lambda input, dim: torch.softmax(input.to(torch.float), dim))
    register("softmax", Predicate(p2), k2)
    p3 = make_mock(lambda input, dim: isinstance(input, str))
    k3 = make_mock(lambda input, dim: torch.softmax(input.to(torch.float), dim))
    register("softmax", Predicate(p3), k3)

    input = torch.randint(low=0, high=10, size=(5, 16), device="cuda", dtype=torch.int8)
    dim = 1
    kernel = dispatch("softmax", input, dim)
    assert kernel is not None
    kernel(input, dim)

    p3.assert_called_once()
    k3.assert_not_called()

    p2.assert_called_once()
    k2.assert_called_once()

    p1.assert_not_called()
    k1.assert_not_called()


def test_dispatch_registration_hook(_seed_prngs: int) -> None:
    predicate = make_mock(lambda input, dim: input.dtype == torch.int8)
    kernel = make_mock(lambda input, dim: torch.softmax(input.to(torch.float), dim))

    input = torch.randint(low=0, high=10, size=(5, 16), device="cuda", dtype=torch.int8)
    dim = 1

    with register("softmax", Predicate(predicate), kernel):
        kernel = dispatch("softmax", input, dim)
        assert kernel is not None

        torch.testing.assert_close(kernel(input, dim), torch.softmax(input.to(torch.float), dim))
        predicate.assert_called_once()
        kernel.assert_called_once()  # type: ignore[attr-defined]

    kernel = dispatch("softmax", input, dim)
    assert kernel is None


@fastforward.flags.context(fastforward.strict_quantization, False)
def test_dispatchers_are_all_the_same() -> None:
    from fastforward.nn import functional

    a = random_quantized(
        (5, 1, 3),
    )
    b = random_quantized((1, 2, 1), scale=0.2, offset=3)

    print(a)

    out_ff_functional_default = functional.add(
        a, b, output_quantizer=None, strict_quantization=False
    )
    out_torch_functional_default = torch.add(a, b)
    out_torch_tensor_method_default = a + b

    expected_output_add = a.dequantize() + b.dequantize()

    torch.testing.assert_close(out_ff_functional_default, expected_output_add)
    torch.testing.assert_close(out_torch_functional_default, expected_output_add)
    torch.testing.assert_close(out_torch_tensor_method_default, expected_output_add)

    predicate = Predicate(lambda *_, **__: True)

    def kernel(input: QuantizedTensor, other: QuantizedTensor, **_kwargs: Any) -> torch.Tensor:
        return input.dequantize() - other.dequantize()

    kernel_mock = make_mock(kernel)
    predicate_mock = make_mock(predicate)
    with register("add", Predicate(predicate_mock), kernel_mock):
        out_ff_functional_kernel = functional.add(
            a, b, output_quantizer=None, strict_quantization=False
        )
    predicate_mock.assert_called_once()
    kernel_mock.assert_called_once()
    kernel_mock.assert_called_with(
        a, b, alpha=1.0, output_quantizer=None, strict_quantization=False
    )

    kernel_mock = make_mock(kernel)
    predicate_mock = make_mock(predicate)
    with register("add", Predicate(predicate_mock), kernel_mock):
        out_torch_functional_kernel = torch.add(a, b)
    predicate_mock.assert_called_once()
    kernel_mock.assert_called_once()
    kernel_mock.assert_called_with(a, b)

    kernel_mock = make_mock(kernel)
    predicate_mock = make_mock(predicate)
    with register("add", Predicate(predicate_mock), kernel_mock):
        out_torch_tensor_method_kernel = a + b
    predicate_mock.assert_called_once()
    kernel_mock.assert_called_once()
    kernel_mock.assert_called_with(a, b)

    expected_output_subtract = a.dequantize() - b.dequantize()
    torch.testing.assert_close(out_ff_functional_kernel, expected_output_subtract)
    torch.testing.assert_close(out_torch_functional_kernel, expected_output_subtract)
    torch.testing.assert_close(out_torch_tensor_method_kernel, expected_output_subtract)
