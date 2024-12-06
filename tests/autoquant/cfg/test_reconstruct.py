# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import textwrap

import libcst

from fastforward.autoquant.cfg import _construct, _reconstruct


def _get_cst(src: str) -> libcst.CSTNode:
    src = textwrap.dedent(src).strip()
    return libcst.parse_module(src).body[0]


def test_reconstruct():
    src = """
    def example_func(a: int) -> int:
        if a > 10:
            return a
        else:
            return 10
    """
    cst = _get_cst(src)

    assert isinstance(cst, libcst.FunctionDef)

    cfg = _construct.construct(cst)
    reconstructed = _reconstruct.reconstruct(cfg)

    assert cst.deep_equals(reconstructed)
