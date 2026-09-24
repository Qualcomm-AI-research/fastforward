# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import pathlib

from pathlib import Path
from types import ModuleType

import fastforward as ff
import libcst
import pytest

from fastforward._autoquant.cst.filter import filter_nodes_by_type
from fastforward._autoquant.mypy.type_provider import MypyTypeProvider, TypeInfo
from fastforward._autoquant.mypy.type_provider_impl import (
    _get_mypy_tree_and_checker,
    mypy_call_scoped_cache,
    mypy_module_context,
)


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


@pytest.mark.slow
def test_stored_scalar_facts_distinguish_unknown_and_mixed_types() -> None:
    # GIVEN: Stored types for numeric values, unions, literals and unknown values.
    cst = libcst.parse_module("""
from typing import Any, Literal

def values(integer: int, floating: float, boolean: bool, numeric: int | float,
           optional: int | None, unknown: Any, text: str, literal: Literal[1]):
    return integer, floating, boolean, numeric, optional, unknown, text, literal
""")
    metadata = libcst.MetadataWrapper(cst, unsafe_skip_copy=True).resolve(MypyTypeProvider)
    returned = next(filter_nodes_by_type(cst, libcst.Return)).value
    assert isinstance(returned, libcst.Tuple)

    # WHEN: Read the stored proof for each returned expression.
    proven = [metadata[element.value].is_proven_scalar() for element in returned.elements]

    # THEN: Every possible type must be numeric; Any and optional types are not proofs.
    assert proven == [True, True, True, True, False, False, False, True]


@pytest.mark.slow
def test_mypy_module_context_restored_after_error() -> None:
    # GIVEN standalone code analyzed outside any source module
    code = "value: int = 1\n"
    with mypy_call_scoped_cache():
        standalone = _get_mypy_tree_and_checker(code)

        # WHEN nested module analysis exits with an exception
        with mypy_module_context("outer.model", None):
            outer = _get_mypy_tree_and_checker(code)
            with pytest.raises(RuntimeError, match="analysis failed"):
                with mypy_module_context("inner.model", None):
                    inner = _get_mypy_tree_and_checker(code)
                    assert inner is not None
                    assert inner[0].fullname == "inner.model"
                    msg = "analysis failed"
                    raise RuntimeError(msg)

            # THEN both module context and cached results return to the outer scope
            assert outer is not None
            assert outer[0].fullname == "outer.model"
            assert _get_mypy_tree_and_checker(code) is outer

        assert standalone is not None
        assert standalone[0].fullname == "_ff_evaluation_module__"
        assert _get_mypy_tree_and_checker(code) is standalone


@pytest.mark.slow
def test_mypy_cache_distinguishes_source_paths(tmp_path: Path) -> None:
    # GIVEN identical source and module names with dependencies in different roots
    code = "from .dependency import value\nresult = value\n"
    paths = []
    for root, value in (("first", "1"), ("second", '"text"')):
        package = tmp_path / root / "cache_package"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (package / "dependency.py").write_text(f"value = {value}\n")
        path = package / "model.py"
        path.write_text(code)
        paths.append(path)

    # WHEN both sources are analyzed within the same cache scope
    inferred_types = []
    with mypy_call_scoped_cache():
        for path in paths:
            with mypy_module_context("cache_package.model", str(path)):
                cst = libcst.parse_module(code)
                metadata = libcst.MetadataWrapper(cst, unsafe_skip_copy=True).resolve(
                    MypyTypeProvider
                )
                assignment = next(filter_nodes_by_type(cst, libcst.Assign))
                inferred_types.append(str(metadata[assignment.value].typ))

    # THEN each import resolves against its own source root, without stale types
    assert inferred_types == ["int", "str"]


@pytest.mark.slow
@pytest.mark.parametrize("module", [libcst, pathlib])
def test_mypy_library_module_preserves_library_stubs(module: ModuleType) -> None:
    # GIVEN library source that uses a typeshed-provided module
    code = "from typing_extensions import final\n@final\nclass Example:\n    pass\n"

    # WHEN mypy analyzes source under the library module's identity
    with mypy_module_context(module.__name__, module.__file__):
        result = _get_mypy_tree_and_checker(code)

    # THEN library source directories do not shadow mypy's bundled stubs
    assert result is not None
    assert result[0].fullname == module.__name__
