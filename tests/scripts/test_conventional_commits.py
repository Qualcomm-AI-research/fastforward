# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import pytest

from scripts.conventional_commits import title_error


@pytest.mark.parametrize(
    "title",
    [
        "docs: valid unscoped title",
        "fix(ci): valid scoped title",
        "feat!: valid breaking change",
        "refactor(export)!: valid scoped breaking change",
        "fix: description: with a colon",
    ],
)
def test_title_error_valid_title(title: str) -> None:
    # GIVEN a valid Conventional Commit title
    # WHEN checking the title
    error = title_error(title)

    # THEN no error is reported
    assert error is None


@pytest.mark.parametrize(
    "title",
    [
        "missing conventional commit prefix",
        "fix(ci):missing space after colon",
        "fix(ci): ",
        "feat(): empty scope",
        "fix(ci: unmatched scope",
        "feat!(ci): breaking marker before scope",
        "Fix: uppercase type",
        "feat2: digit in type",
    ],
)
def test_title_error_invalid_structure(title: str) -> None:
    # GIVEN a title with invalid Conventional Commit structure
    # WHEN checking the title
    error = title_error(title)

    # THEN the error explains the expected structure
    assert error == "expected 'type: description' or 'type(scope): description'"


def test_title_error_unknown_type() -> None:
    # GIVEN a structurally valid title with an unknown type
    title = "wip: unknown type"

    # WHEN checking the title
    error = title_error(title)

    # THEN the error lists the allowed types
    assert (
        error
        == "type 'wip' is not one of build, chore, ci, docs, feat, fix, perf, refactor, style, test"
    )
