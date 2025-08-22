# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import importlib.util
import io
import pathlib
import sys

from typing import Iterator

import fastforward as ff
import pytest
import torch

from fastforward._autoquant.pybuilder import TextIOWriter
from fastforward.testing.string import assert_strings_match_verbose


def _find_test_cases() -> Iterator[tuple[str, torch.nn.Module, str]]:
    """Find test cases in `test_data` folder.

    Test cases consists of a `<name>.py` and `<name>.expected.py` file.
    `<name>.py` must contain a function `get_model` which returns a single
    `torch.nn.Module` instance. `<name>.expected.py` contains the __exact__
    output that is expected to be produced by `ff.autoquantize`.
    """
    test_cases_path = pathlib.Path(__file__).parent.resolve() / "test_data"
    test_cases: dict[str, set[str]] = {}
    for filename in test_cases_path.glob("*.py"):
        case_name = filename.stem.removesuffix(".expected")
        test_cases.setdefault(case_name, set()).add(filename.name)

    expected_suffixes = (".py", ".expected.py")
    for case_name, files in test_cases.items():
        if {f"{case_name}{suffix}" for suffix in expected_suffixes}.issubset(files):
            yield _load_case(case_name, test_cases_path)


def _load_case(case_name: str, test_cases_path: pathlib.Path) -> tuple[str, torch.nn.Module, str]:
    """Load test case given `case_name`."""
    module_file_path = test_cases_path / f"{case_name}.py"
    package_name, _ = __name__.rsplit(".", 1)
    test_case_package = f"{package_name}.test_data"
    module_name = f"{test_case_package}.{case_name}"
    spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    assert spec is not None
    assert spec.loader is not None

    case_py_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = case_py_module
    spec.loader.exec_module(case_py_module)

    assert (module_file := case_py_module.__file__) is not None
    expected_file = pathlib.Path(module_file).with_suffix(".expected.py")

    return case_name, case_py_module.get_model(), expected_file.read_text()


@pytest.mark.parametrize(
    "module,expected_output",
    [
        pytest.param(module, expected_output, id=name)
        for name, module, expected_output in _find_test_cases()
    ],
)
@pytest.mark.slow
def test_autoquant_end_to_end(module: torch.nn.Module, expected_output: str) -> None:
    # GIVEN a PyTorch module
    # WHEN the module is autoquantized
    code_writer = TextIOWriter("TestModule", writer=(buffer := io.StringIO()))
    ff.autoquantize(module, code_writer=code_writer, auto_import=False, use_type_inference=False)

    # Then the generated code must match expectations
    assert_strings_match_verbose(expected_output.strip(), buffer.getvalue().strip())
