# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


import pytest
import torch

from fastforward._quantops import OperatorTable
from fastforward._quantops.optable import STR_ALIASES_EXTENSIONS


@pytest.fixture()
def op_table() -> OperatorTable:
    table = OperatorTable(alias_extensions=STR_ALIASES_EXTENSIONS)
    table.add(
        "linear(first: Quantized, second: Quantized) -> Quantized",
        "torch.nn.functional.linear",
    )
    table.add("relu(input: Quantized) -> Quantized", "torch.relu")
    return table


def test_optable_creation(op_table: OperatorTable) -> None:
    linear_op = next(op_table["torch.nn.functional.linear"])
    relu_op = next(op_table["torch.relu"])

    assert linear_op.identifier == "linear"
    assert relu_op.identifier == "relu"


def test_optable_get(op_table: OperatorTable) -> None:
    assert next(op_table[torch.relu]) is not None
    assert next(op_table["torch.relu"]) is not None
    assert next(op_table.get(torch.relu)) is not None
    assert next(op_table.get("torch.relu")) is not None

    with pytest.raises(KeyError):
        list(op_table[torch.sigmoid])
    with pytest.raises(KeyError):
        list(op_table.get(torch.sigmoid))
    with pytest.raises(KeyError):
        list(op_table["torch.sigmoid"])
    with pytest.raises(KeyError):
        list(op_table.get("torch.sigmoid"))


def test_optable_alias(op_table: OperatorTable) -> None:
    op_table.add_alias("sigmoid", torch.relu)

    assert list(op_table["sigmoid"]) == list(op_table[torch.relu])
    assert list(op_table.get("sigmoid")) == list(op_table.get("torch.relu"))


def test_multiple_operators_same_fallback() -> None:
    """Test that multiple operators can be added for the same fallback operation."""
    # GIVEN an `OperatorTable`
    table = OperatorTable(alias_extensions=STR_ALIASES_EXTENSIONS)

    # WHEN three Operator implementation for `torch.relu` are dded
    table.add("relu(input: Quantized) -> Quantized", torch.relu)
    table.add("relu(input: Quantized, inplace: bool) -> Quantized", torch.relu)
    table.add("relu(input: Quantized, threshold: float) -> Quantized", torch.relu)

    # Get all operators for torch.relu
    # THEN a lookup for `torch.relu` must return three operators
    operators = list(table.get("torch.relu"))
    assert len(operators) == 3, f"Expected 3 operators, got {len(operators)}"

    # THEN all operators must all have the same identifier
    assert all(op.identifier == "relu" for op in operators)


def test_eval(op_table: OperatorTable) -> None:
    # GIVEN An operator Table
    # WHEN A membership test is performed
    # THEN the membership test succeeds/fails based on the actual membership
    assert torch.relu in op_table
    assert torch.sigmoid not in op_table
