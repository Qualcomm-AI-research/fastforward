# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from collections import defaultdict

import pytest
import torch

from fastforward._quantops import OperatorTable
from fastforward._quantops.optable import _QualifiedNameReference


@pytest.fixture()
def op_table() -> OperatorTable:
    table = OperatorTable()
    table.add(
        "linear(first: Quantized, second: Quantized) -> Quantized",
        "torch.nn.functional.linear",
    )
    table.add("relu(input: Quantized) -> Quantized", "torch.relu")
    return table


def test_optable_creation(op_table: OperatorTable) -> None:
    linear_op = op_table["torch.nn.functional.linear"]
    relu_op = op_table["torch.relu"]

    assert linear_op.identifier == "linear"
    assert relu_op.identifier == "relu"


def test_optable_get(op_table: OperatorTable) -> None:
    assert op_table[torch.relu] is not None
    assert op_table["torch.relu"] is not None
    assert op_table.get(torch.relu) is not None
    assert op_table.get("torch.relu") is not None

    with pytest.raises(KeyError):
        op_table[torch.sigmoid]
    with pytest.raises(KeyError):
        op_table.get(torch.sigmoid)
    with pytest.raises(KeyError):
        op_table["torch.sigmoid"]
    with pytest.raises(KeyError):
        op_table.get("torch.sigmoid")


def test_optable_alias(op_table: OperatorTable) -> None:
    op_table.add_alias("sigmoid", torch.relu)

    assert op_table["sigmoid"] is op_table[torch.relu]
    assert op_table.get("sigmoid") is op_table.get("torch.relu")


def test_qualified_name_reference():
    ref = _QualifiedNameReference("collections.defaultdict")
    assert defaultdict is ref.import_()

    ref = _QualifiedNameReference("collections.defaultdict.copy")
    assert defaultdict.copy is ref.import_()

    ref = _QualifiedNameReference("collections.defaultdict.does_not_exist")
    with pytest.raises(ImportError):
        ref.import_()

    ref = _QualifiedNameReference("collections.does_not_exist")
    with pytest.raises(ImportError):
        ref.import_()

    ref = _QualifiedNameReference("does_not_exist.attribute")
    with pytest.raises(ImportError):
        ref.import_()
