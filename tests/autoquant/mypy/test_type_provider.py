# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import fastforward as ff
import libcst
import pytest

from fastforward._autoquant.cst.filter import filter_nodes_by_type
from fastforward._autoquant.mypy.type_provider import MypyTypeProvider, TypeInfo


@pytest.mark.slow
def test_mypy_type_provider() -> None:
    # Given: A simple Python code snippet with type annotations
    (code,) = ff.testing.string.dedent_strip("""
    a: float = 3.14
    b: float = 2.7
    c = a + b
    """)
    cst = libcst.parse_module(code)

    # When: We resolve the MypyTypeProvider metadata
    type_data = libcst.MetadataWrapper(cst, unsafe_skip_copy=True).resolve(MypyTypeProvider)
    assert type_data is not None

    # Then: The type provider should correctly infer types for all assignments
    node_types = (libcst.Assign, libcst.AnnAssign)
    assign_nodes = list(filter_nodes_by_type(cst, node_types))
    assert len(assign_nodes) == 3
    for assign in assign_nodes:
        assert isinstance(assign, node_types)
        target = assign.target if isinstance(assign, libcst.AnnAssign) else assign.targets[0]
        assert target in type_data
        assert isinstance(type_data[target], TypeInfo)

        if assign.value is not None:
            assert assign.value in type_data
            assert isinstance(type_data[assign.value], TypeInfo)
            assert type_data[assign.value].typ == type_data[target].typ
