# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from fastforward._quantops import symtypes


def test_union_replace() -> None:
    union_type = symtypes.Tensor | symtypes.QuantizedTensor | symtypes.Float
    replaced_union_type = union_type.replace(symtypes.Tensor | symtypes.Float, symtypes.Int)

    assert replaced_union_type == symtypes.QuantizedTensor | symtypes.Int

    single_type = union_type.replace(symtypes.Tensor | symtypes.QuantizedTensor, symtypes.Float)
    assert single_type == symtypes.Float


def test_union_order_independent() -> None:
    # Explicitly construct unions to change order of internal repr of parameters.
    # symtypes.Union[] does not support this
    lhs = symtypes._GenericUnionType("Union", (symtypes.Int, symtypes.Float))
    rhs = symtypes._GenericUnionType("Union", (symtypes.Float, symtypes.Int))
    assert lhs == rhs


def test_unwrap_optional() -> None:
    opttype = symtypes.Optional[symtypes.Int]
    opttype2 = symtypes.Optional[symtypes.Int | symtypes.Float]
    nonopttype = symtypes.Float
    nonoptunion = symtypes.Float | symtypes.Int

    assert symtypes.unwrap_optional(opttype) == symtypes.Int
    assert symtypes.unwrap_optional(opttype2) == symtypes.Int | symtypes.Float
    assert symtypes.unwrap_optional(nonopttype) == symtypes.Float
    assert symtypes.unwrap_optional(nonoptunion) == nonoptunion
