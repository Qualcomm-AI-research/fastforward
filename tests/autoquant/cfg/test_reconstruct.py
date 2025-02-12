# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import difflib

import libcst
import pytest

from fastforward.autoquant.cfg import construct, reconstruct

from tests.autoquant.cfg import reconstruct_test_cases

_RECONSTRUCTION_CASES = list(reconstruct_test_cases.test_cases())


@pytest.mark.parametrize(
    "input,output,description",
    ([(case.input, case.output, case.description) for case in _RECONSTRUCTION_CASES]),
    ids=[case.description.replace(" ", "_") for case in _RECONSTRUCTION_CASES],
)
def test_reconstruct(input: str, output: str, description: str) -> None:
    """Load example from `reconstruct_test_case_data.py`.

    Examples are separated by CASE, ENDCASE markers. An EXPECT marker
    delineates the input and expected output example. If the expected output of
    code generated after an CST -> CFG -> CST round trip must match the input
    exactly, `EXPECT: exact` can be used instead of providing an expected
    output example.

    Here, each example is loaded, converted into a CST, followed by a CST ->
    CFG -> CST round trip. The resulting CST is used to generate code and match
    against the expected output.
    """
    # GIVEN a CST of a function
    cst = libcst.parse_statement(input)
    assert isinstance(cst, libcst.FunctionDef)

    # WHEN a CFG is created from the CST and then a CST from the CFG
    cfg = construct(cst)
    cst_ = reconstruct(cfg)

    # THEN the code generated using the CST must match the expected output exactly
    gen_output = libcst.Module((cst_,)).code.strip()
    if gen_output != output:
        msg = "\n".join(difflib.unified_diff(output.splitlines(), gen_output.splitlines()))
        raise AssertionError(
            "Reconstructed function does not match expected output.\n"
            + f"Description: {description}\n Diff:\n{msg}"
        )
