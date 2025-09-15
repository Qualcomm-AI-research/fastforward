# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import difflib
import textwrap


def dedent_strip(*txt: str) -> tuple[str, ...]:
    """Returns dedented then stripped version of input string(s)."""
    return tuple(textwrap.dedent(t).strip() for t in txt)


def assert_strings_match_verbose(str1: str, str2: str) -> None:
    """Assert that `str1` and `str2` match exactly.

    Raise `AssertionError` with a verbose diff if strings don't match.
    """
    if not str1 == str2:
        output = "\n".join(difflib.unified_diff(str2.splitlines(), str1.splitlines()))
        msg = f"Transformed module does not match expected output:\n{output}"
        raise AssertionError(msg)
