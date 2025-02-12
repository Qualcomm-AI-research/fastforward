# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import dataclasses
import pathlib

from collections.abc import Iterator
from typing import Literal


@dataclasses.dataclass
class ReconstructTestCase:
    input: str = ""
    output: str = ""
    description: str = ""
    kind: str | None = None


RECONSTRUCT_TEST_CASE_FILE = pathlib.Path(__file__).parent / "reconstruct_test_case_data.py"


def test_cases() -> Iterator[ReconstructTestCase]:
    """Create iterator over CFG reconstruction test cases in
    `RECONSTRUCT_TEST_CASE_FILE`.

    Reads test cases from `RECONSTRUCT_TEST_CASE_FILE` and yields
    `ReconstructTestCase` instances for each.

    Returns:
        Iterator of `ReconstructTestCase`s.
    """
    yield from _read_test_cases(RECONSTRUCT_TEST_CASE_FILE)


def _read_test_cases(filename: pathlib.Path | str) -> Iterator[ReconstructTestCase]:
    """Read test_cases from `filename`.

    Cases are separated by CASE and ENDCASE markers. An EXPECT marker
    delineates the input and expected output example. If the expected output of
    code generated after an CST -> CFG -> CST round trip must match the input
    exactly, `EXPECT: exact` can be used instead of providing an expected
    output example.

    Args:
        filename: The file to read the test cases from.

    Returns:
        Iterator over `ReconstructTestCase`s
    """
    filename = pathlib.Path(filename)
    with filename.open() as f:
        state: Literal["outside", "input", "output"] = "outside"
        test_case = ReconstructTestCase()

        for line in f:
            match state:
                case "outside":
                    line = line.strip()
                    if not _read_label(line) == "CASE":
                        continue
                    test_case.description = _strip_label(line).strip()
                    state = "input"
                case "input":
                    if _read_label(line) == "EXPECT":
                        test_case.kind = _strip_label(line) or None
                        state = "output"
                    else:
                        test_case.input += line
                case "output":
                    if _read_label(line) == "ENDCASE":
                        state = "outside"
                        yield _complete_testcase(test_case)
                        test_case = ReconstructTestCase()
                    else:
                        test_case.output += line


def _complete_testcase(test_case: ReconstructTestCase) -> ReconstructTestCase:
    """Perform some post-processing on `test_case`.

    Post processing includes the following:
        - Remove any leading and trailing whitespace from input and output
        - If `test_case.kind` is exact. Make sure that no output example was
          provided and duplicate the input to the output.
        - Make sure that the `kind` type is supported.

    Args:
        test_case: Test case to post process.

    Returns:
        The post-processed test case.
    """
    test_case.input = test_case.input.strip()
    test_case.output = test_case.output.strip()
    match test_case.kind:
        case "exact":
            if test_case.output:
                raise ValueError(
                    f"Test case '{test_case.description}' was marked as 'exact' but "
                    + "expected output is non-empty."
                )
            test_case.output = test_case.input
        case None:
            pass
        case _:
            raise ValueError(f"'{test_case.kind}' is not supported")
    return test_case


def _read_label(line: str) -> str | None:
    """Helper to read the test case label from a raw line of text."""
    line = line.strip()
    if not line.startswith("#"):
        return None

    label = line = line[1:].strip().split(":")[0]
    if label in ["CASE", "ENDCASE", "EXPECT"]:
        return label

    return None


def _strip_label(line: str) -> str:
    """Helper to strip the test case label from a raw line of text."""
    splits = line.split(":", maxsplit=1)
    return splits[-1].strip() if len(splits) > 1 else ""
