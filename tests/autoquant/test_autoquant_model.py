# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import importlib

from typing import Iterator

import fastforward as ff
import pytest
import syrupy

from fastforward._autoquant.cst.pattern import PatternRule
from fastforward._autoquant.pybuilder.writer import StdoutWriter
from fastforward.testing.package_mock import PackageMock

_MODULE_A_SOURCE = """
import torch

def foo(x: torch.Tensor):
    my_var = 1
    return x + my_var
"""

_MODULE_B_SOURCE = """
import torch

def foo(x: torch.Tensor):
    my_var = 1
    return x + my_var
"""


def _model_source(package_name: str) -> str:
    return f"""
import torch

from {package_name} import module_a
from {package_name} import module_b


class Model(torch.nn.Module):
    def forward(self, x):
        a = module_a.foo(x)
        b = module_b.foo(x)
        return a + b


def my_model():
    return Model()
"""


@pytest.fixture
def fake_my_package() -> Iterator[str]:
    """Yields the dotted name of an in-memory package with model/module_a/module_b."""
    name = "my_package"
    pkg = PackageMock({
        f"{name}.module_a": _MODULE_A_SOURCE,
        f"{name}.module_b": _MODULE_B_SOURCE,
        f"{name}.model": _model_source(name),
    })
    with pkg:
        yield name


@pytest.mark.slow
def test_pattern_rule_robust_type_annotation_on_external_functions(
    fake_my_package: str,
    snapshot: syrupy.assertion.SnapshotAssertion,
) -> None:
    """Rule with `on=` only applies inside the named scope when the source comes from an in-memory package."""
    # GIVEN: An in-memory python package with two python modules and a torch model
    package_name = fake_my_package
    model = importlib.import_module(f"{package_name}.model")
    m = model.my_model()

    # WHEN: Autoquant is applied to the model with a replacement pattern that
    #       annotates `my_var` as `int` only in `module_a.foo`.
    autoquant_code = ff.autoquantize(
        m,
        replacement_patterns=[
            PatternRule.from_str(
                pattern="my_var = {value}",
                replacement="my_var: int = {value}",
                on=f"{package_name}.module_a.foo",
            ),
        ],
        use_type_inference=True,
        code_writer=StdoutWriter(f"quantized_{package_name}"),
    )

    # THEN: The generated quantized module matches the snapshot, with the
    #       type annotation injected only into the function targeted by `on`.
    assert autoquant_code.code == snapshot
