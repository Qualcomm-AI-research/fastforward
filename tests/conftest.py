# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import random

import pytest

from tests.utils import seed_prngs


def pytest_addoption(parser: pytest.Parser) -> None:
    """Adds the `--include-slow` option to the unit tests.

    This allows the user to choose whether to run tests marked as `slow`.
    """
    parser.addoption(
        "--include-slow",
        action="store_const",
        const=True,
        default=False,
        help="run tests marked as `slow`, too",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Configures pytest to modify the mark expression based on the `--include-slow` option.

    If the `--include-slow` option is used, modifies the mark expression to include
    tests marked as `slow`.
    """
    if config.option.include_slow:
        # Replacing 'and not slow' first avoids an invalid `and` left behind.
        for replacement in ["and not slow", "not slow"]:
            config.option.markexpr = config.option.markexpr.replace(replacement, "")


@pytest.fixture(scope="session", name="random_seed")
def random_seed_fixture() -> int:
    """Generates a global, random seed."""
    return random.randint(0, 2**64 - 1)


@pytest.fixture(name="_seed_prngs")
def seed_prngs_fixture(random_seed: int) -> int:
    """Seeds the common PRNGs and returns the seed for reproducibility."""
    seed_prngs(random_seed)
    return random_seed
