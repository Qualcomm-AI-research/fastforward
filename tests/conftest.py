# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import random
import time
import warnings

from pathlib import Path
from typing import Iterator

import pytest
import torch

from fastforward.testing import seed_prngs


def pytest_addoption(parser: pytest.Parser) -> None:
    """Adds the `--include-slow` and `--include-benchmark` options to the unit tests.

    These allow the user to choose whether to run tests marked as `slow` or
    `benchmark`, both of which are deselected by default.
    """
    parser.addoption(
        "--include-slow",
        action="store_const",
        const=True,
        default=False,
        help="run tests marked as `slow`, too",
    )
    parser.addoption(
        "--include-benchmark",
        action="store_const",
        const=True,
        default=False,
        help="run tests marked as `benchmark`, too",
    )

    parser.addoption("--timeout", help="set timeout (in seconds) for `slow` tests", type=float)


def _strip_marker_clause(markexpr: str, marker: str) -> str:
    """Removes the `not <marker>` clause from a pytest mark expression.

    The longer `and`-joined forms are replaced first so no dangling `and`
    operator is left behind in the expression.
    """
    for clause in [f"and not {marker}", f"not {marker} and", f"not {marker}"]:
        markexpr = markexpr.replace(clause, "")
    return markexpr


def pytest_configure(config: pytest.Config) -> None:
    """Configures pytest to modify the mark expression based on the `--include-*` options.

    The default mark expression deselects both `slow` and `benchmark` tests.
    Passing `--include-slow` strips only the `not slow` clause and passing
    `--include-benchmark` strips only the `not benchmark` clause, so the two
    options are independent: each leaves the other marker's clause intact.
    """
    if config.option.include_slow:
        config.option.markexpr = _strip_marker_clause(config.option.markexpr, "slow")
    if config.option.include_benchmark:
        config.option.markexpr = _strip_marker_clause(config.option.markexpr, "benchmark")
    config.option.markexpr = config.option.markexpr.strip()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Called after collection has been performed. May filter or re-order the items in-place."""
    del config
    marker = pytest.mark.skipif(
        torch.__version__ < "2.5",
        reason="Export functionality is only supported for PyTorch version 2.5 and above.",
    )
    export_tests_path = str(Path(__file__).parent / "export")
    for item in items:
        if str(item.path).startswith(export_tests_path):
            item.add_marker(marker)


@pytest.fixture(scope="session", name="random_seed")
def random_seed_fixture() -> int:
    """Generates a global, random seed."""
    return random.randint(0, 2**64 - 1)


@pytest.fixture(name="_seed_prngs")
def seed_prngs_fixture(random_seed: int) -> int:
    """Seeds the common PRNGs and returns the seed for reproducibility."""
    seed_prngs(random_seed)
    return random_seed


@pytest.hookimpl(tryfirst=True, wrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Iterator[None]:
    if item.config.option.timeout is None:
        return (yield)
    start = time.time()
    yield
    duration = time.time() - start
    timeout = item.config.option.timeout
    if duration > timeout:
        pytest.fail(
            f"Execution time exceeded the {timeout} s limit."
            + f" Note: Tests should be marked as `slow` or finish within {timeout / 4} s to be non-flaky."
        )


@pytest.fixture(scope="session", autouse=True)
def enable_profiling_fixture() -> None:
    """Create initial load for torch to increase reproducibility of timing."""
    # Warmup torch._dynamo which is used by `torch.optim.optimizer`.
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="onnxscript.*")
            import torch._dynamo  # noqa: F401
    except ImportError:
        pass

    # Warmup libsct (libcst.matchers)
    import fastforward._autoquant.convert  # noqa: F401

    # Warmup autograd functionality (export `sympy` under the hood which takes a lot of time).
    # For more information please profile imports from the `_make_grads` function in `torch.autograd`.
    torch.ones(1, requires_grad=True).backward(torch.ones(1))

    # Warmup CUDA functionality
    if torch.cuda.is_available():
        _ = (torch.ones(1) + torch.ones(1)).item()
        torch.cuda.synchronize()


@pytest.fixture(name="tmp_home")
def tmp_home_fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """Sets the home directory to a temporary directory."""
    monkeypatch.setenv("HOME", str(tmp_path))
    yield tmp_path
