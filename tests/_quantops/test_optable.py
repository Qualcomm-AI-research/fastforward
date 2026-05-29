# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


from collections.abc import Callable
from typing import Any

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


def test_callable_alias_registers_additional_index_key() -> None:
    # GIVEN a table with an op whose canonical fallback is `torch.relu`
    # and `torch.nn.functional.relu` registered as a semantic alias
    table = OperatorTable(alias_extensions=STR_ALIASES_EXTENSIONS)
    table.add(
        "relu(input: Quantized) -> Quantized",
        "torch.relu",
        aliases=("torch.nn.functional.relu",),
    )

    # WHEN looking up by either the canonical callable, the alias callable, or their
    # qualified-name strings
    # THEN all forms resolve to the same operator entry
    assert torch.relu in table
    assert torch.nn.functional.relu in table
    assert list(table.get(torch.relu)) == list(table.get(torch.nn.functional.relu))
    assert list(table.get("torch.relu")) == list(table.get("torch.nn.functional.relu"))


def test_callable_alias_does_not_duplicate_operator_specs() -> None:
    # GIVEN a table with one canonical op and one semantic alias
    table = OperatorTable(alias_extensions=STR_ALIASES_EXTENSIONS)
    table.add(
        "relu(input: Quantized) -> Quantized",
        "torch.relu",
        aliases=("torch.nn.functional.relu",),
    )

    # WHEN iterating all operators in the table
    # THEN there is exactly one entry — aliases are additional lookup keys,
    # not extra operator definitions
    assert len(list(table.operators())) == 1


def test_callable_alias_collision_raises() -> None:
    # GIVEN a table where one op already claims `torch.relu` as its canonical
    table = OperatorTable(alias_extensions=STR_ALIASES_EXTENSIONS)
    table.add("relu(input: Quantized) -> Quantized", "torch.relu")

    # WHEN another op tries to register `torch.relu` as an alias
    # THEN a ValueError is raised
    with pytest.raises(ValueError, match="already registered"):
        table.add(
            "sigmoid(input: Quantized) -> Quantized",
            "torch.sigmoid",
            aliases=("torch.relu",),
        )


def test_callable_alias_skips_unimportable() -> None:
    # GIVEN a table whose alias list contains a qualified name that cannot be imported
    table = OperatorTable(alias_extensions=STR_ALIASES_EXTENSIONS)

    # WHEN the op is added with the unimportable alias
    table.add(
        "relu(input: Quantized) -> Quantized",
        "torch.relu",
        aliases=("torch.nonexistent.never_exists",),
    )

    # THEN the canonical operator is still registered and usable
    assert torch.relu in table


def test_callable_alias_idempotent_on_overload() -> None:
    # GIVEN a table where both an op and its overload register the same alias
    table = OperatorTable(alias_extensions=STR_ALIASES_EXTENSIONS)
    table.add(
        "relu(input: Quantized) -> Quantized",
        "torch.relu",
        aliases=("torch.nn.functional.relu",),
    )

    # WHEN the alias is registered again as part of an overload of the same op
    table.add(
        "relu(input: Quantized, inplace: bool) -> Quantized",
        "torch.relu",
        aliases=("torch.nn.functional.relu",),
    )

    # THEN no error is raised and both forms still resolve to the overload set
    relu_ops = list(table.get(torch.nn.functional.relu))
    assert len(relu_ops) == 2
    assert relu_ops == list(table.get(torch.relu))


def test_callable_alias_string_lookup_round_trips_through_resolve_alias() -> None:
    # GIVEN a table with a callable alias registered against an op
    table = OperatorTable(alias_extensions=STR_ALIASES_EXTENSIONS)
    table.add(
        "relu(input: Quantized) -> Quantized",
        "torch.relu",
        aliases=("torch.nn.functional.relu",),
    )

    # WHEN looking up the alias via its qualified-name string
    # THEN _resolve_alias maps the string back to the alias callable, and the
    # subsequent index lookup yields the same operator(s) as the canonical
    assert table._resolve_alias("torch.nn.functional.relu") is torch.nn.functional.relu
    assert list(table["torch.nn.functional.relu"]) == list(table[torch.relu])
    assert list(table.get("torch.nn.functional.relu")) == list(table.get(torch.relu))


def test_default_optable_resolves_torch_functional_aliases() -> None:
    # GIVEN the default optable loaded directly from the shipped YAML
    table = OperatorTable.from_yaml(alias_extensions=STR_ALIASES_EXTENSIONS)
    pairs: list[tuple[Callable[..., Any], Callable[..., Any]]] = [
        (torch.relu, torch.nn.functional.relu),
        (torch.sigmoid, torch.nn.functional.sigmoid),
        (torch.softmax, torch.nn.functional.softmax),
    ]

    # WHEN looking up each alias callable
    for canonical, alias in pairs:
        # THEN the alias is present in the table and resolves to the same operator as its canonical
        assert alias in table
        assert list(table.get(alias)) == list(table.get(canonical))
