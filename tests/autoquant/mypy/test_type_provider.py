# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Check behavior specific to the native mypy adapter.

Common inference checks live in tests/autoquant/test_type_provider.py and use the
selected backend. These checks intentionally use mypy for its native inheritance
semantics, snapshot bookkeeping, cached analysis objects, and bundled stubs.
"""

import contextlib
import pathlib

from types import ModuleType

import libcst
import pytest

from fastforward._autoquant.cst.filter import filter_nodes_by_type
from fastforward._autoquant.mypy.type_provider import MypyTypeProvider
from fastforward._autoquant.mypy.type_provider_impl import (
    MypyClient,
    _get_mypy_tree_and_checker,
    mypy_call_scoped_cache,
)
from fastforward._autoquant.tsp.client import StaleSnapshotError
from fastforward._autoquant.tsp.context import module_context
from fastforward._autoquant.tsp.types import TypeInfo


@pytest.mark.slow
def test_shared_mypy_provider_preserves_native_subtyping() -> None:
    """Keep mypy's inheritance semantics when wrapping its types in shared metadata.

    A native Child must still match Base, including inside a sequence. Remote
    TSP descriptions lack this inheritance information, so this regression
    belongs to the mypy adapter rather than the common provider contract.
    """
    # GIVEN inheritance and sequence types that require native checker semantics.
    module = libcst.parse_module("""
from typing import Any
class Base: pass
class Child(Base): pass
class Other: pass
def values(child: Child, other: Other, children: list[Child], others: list[Other], unknown: Any):
    return child, other, children, others, unknown
""")

    # WHEN mypy types are collected through the common provider.
    metadata = libcst.MetadataWrapper(module, unsafe_skip_copy=True).resolve(MypyTypeProvider)
    returned = next(filter_nodes_by_type(module, libcst.Return)).value
    assert isinstance(returned, libcst.Tuple)
    infos = [metadata[element.value] for element in returned.elements]
    name = "_ff_evaluation_module__.Base"

    # THEN the shared metadata retains subclass, sequence, and Any behavior.
    assert all(isinstance(info, TypeInfo) for info in infos)
    assert [info.is_subtype(name) for info in infos] == [True, False, False, False, True]
    assert [info.is_sequence_of_subtype(name) for info in infos] == [
        False,
        False,
        True,
        False,
        True,
    ]


@pytest.mark.slow
def test_mypy_client_rejects_stale_snapshots() -> None:
    """Prevent closed-document types from being read through an old snapshot.

    Mypy has no TSP snapshots of its own; its adapter maintains them to obey the
    shared client contract. Closing a document must invalidate previous queries
    and remove its metadata so later analysis cannot reuse stale types.
    """
    # GIVEN an analyzed document and its snapshot.
    with contextlib.closing(MypyClient()) as client:
        client.open_document("file:///first.py", "value = 1\n")
        snapshot = client.get_snapshot()
        code_range = libcst.metadata.CodeRange(
            libcst.metadata.CodePosition(1, 8), libcst.metadata.CodePosition(1, 9)
        )
        assert client.get_computed_type("file:///first.py", code_range, snapshot) is not None

        # WHEN closing the document invalidates the analysis state.
        client.close_document("file:///first.py")

        # THEN its previous types cannot leak into subsequent queries.
        with pytest.raises(StaleSnapshotError, match="no longer current"):
            client.get_computed_type("file:///first.py", code_range, snapshot)
        assert (
            client.get_computed_type("file:///first.py", code_range, client.get_snapshot()) is None
        )


@pytest.mark.slow
def test_mypy_analysis_cache_is_restored_after_nested_module_error() -> None:
    """Reuse the outer mypy analysis after a nested conversion fails.

    Identical code analyzed under different module names needs distinct cached
    mypy trees. Restoring the shared module context must recover the exact outer
    cached result, including after an exception, without rebuilding its analysis.
    """
    # GIVEN standalone code analyzed outside any source module
    code = "value: int = 1\n"
    with mypy_call_scoped_cache():
        standalone = _get_mypy_tree_and_checker(code)

        # WHEN nested module analysis exits with an exception
        with module_context("outer.model", None):
            outer = _get_mypy_tree_and_checker(code)
            with pytest.raises(RuntimeError, match="analysis failed"):
                with module_context("inner.model", None):
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
@pytest.mark.parametrize("module", [libcst, pathlib])
def test_mypy_library_module_preserves_library_stubs(module: ModuleType) -> None:
    """Keep library source paths from shadowing mypy's bundled dependency stubs.

    Analyzing a library module changes mypy's import search roots. Its typeshed
    imports must still resolve, for both a third-party package and the standard
    library, or Autoquant cannot analyze code from those libraries.
    """
    # GIVEN library source that uses a typeshed-provided module
    code = "from typing_extensions import final\n@final\nclass Example:\n    pass\n"

    # WHEN mypy analyzes source under the library module's identity
    with module_context(module.__name__, module.__file__):
        result = _get_mypy_tree_and_checker(code)

    # THEN library source directories do not shadow mypy's bundled stubs
    assert result is not None
    assert result[0].fullname == module.__name__
