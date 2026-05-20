# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import importlib
import inspect
import linecache
import sys
import textwrap

import pytest

from fastforward.testing.package_mock import PackageMock


def test_single_leaf_module_imports_and_executes() -> None:
    # GIVEN: A PackageMock with one leaf module
    pkg = PackageMock().add_module("fp_test_single.leaf", "value = 42")

    # WHEN: Entered and the module is imported
    with pkg:
        mod = importlib.import_module("fp_test_single.leaf")

        # THEN: The module's body has executed
        assert mod.value == 42


def test_implicit_parent_packages_are_created() -> None:
    # GIVEN: A PackageMock with a deeply nested module name
    pkg = PackageMock().add_module("fp_test_nested.a.b.c", "x = 1")

    # WHEN: The leaf is imported
    with pkg:
        importlib.import_module("fp_test_nested.a.b.c")

        # THEN: Each parent prefix is importable as a package
        for parent in ("fp_test_nested", "fp_test_nested.a", "fp_test_nested.a.b"):
            parent_mod = importlib.import_module(parent)
            assert parent_mod.__package__ == parent
            assert parent_mod.__path__ == []


def test_explicit_source_on_parent_becomes_init_content() -> None:
    # GIVEN: A package name registered with source AND a child registered too
    pkg = (
        PackageMock()
        .add_module("fp_test_init", "exposed = 'parent'")
        .add_module("fp_test_init.child", "x = 1")
    )

    # WHEN: The package is imported
    with pkg:
        parent_mod = importlib.import_module("fp_test_init")

        # THEN: The package executes its source as __init__ AND is a package
        assert parent_mod.exposed == "parent"
        assert parent_mod.__path__ == []
        assert parent_mod.__package__ == "fp_test_init"


def test_inspect_getsource_returns_dedented_source() -> None:
    # GIVEN: A module whose source string is given with leading indentation
    indented_source = """
        def foo() -> int:
            return 7
    """
    pkg = PackageMock().add_module("fp_test_inspect.mod", indented_source)
    expected = textwrap.dedent(indented_source)

    # WHEN: The module is imported
    with pkg:
        mod = importlib.import_module("fp_test_inspect.mod")

        # THEN: inspect.getsource returns the dedented source and the function works
        assert inspect.getsource(mod) == expected
        assert mod.foo() == 7


def test_inspect_getsource_works_on_packages() -> None:
    # GIVEN: A package with explicit source and a child module
    pkg = (
        PackageMock()
        .add_module("fp_test_pkg_src", "marker = 'init'")
        .add_module("fp_test_pkg_src.child", "")
    )

    # WHEN: The package is imported
    with pkg:
        pkg_mod = importlib.import_module("fp_test_pkg_src")

        # THEN: inspect.getsource returns the package's __init__ content
        assert inspect.getsource(pkg_mod) == "marker = 'init'"


def test_exit_removes_finder_and_clears_caches() -> None:
    # GIVEN: A PackageMock that has been entered and used
    pkg = PackageMock().add_module("fp_test_cleanup.leaf", "v = 1")
    with pkg:
        importlib.import_module("fp_test_cleanup.leaf")
        finder_count_inside = sum(
            1 for f in sys.meta_path if f.__class__.__name__ == "_PackageMockFinder"
        )
        assert finder_count_inside == 1
        assert "fp_test_cleanup" in sys.modules
        assert "fp_test_cleanup.leaf" in sys.modules
        assert "<package-mock:fp_test_cleanup.leaf>" in linecache.cache

    # THEN: After exit, the finder is gone and synthetic state is cleared
    assert not any(f.__class__.__name__ == "_PackageMockFinder" for f in sys.meta_path)
    assert "fp_test_cleanup" not in sys.modules
    assert "fp_test_cleanup.leaf" not in sys.modules
    assert "<package-mock:fp_test_cleanup.leaf>" not in linecache.cache
    assert "<package-mock:fp_test_cleanup>" not in linecache.cache


def test_cleanup_runs_when_body_raises() -> None:
    # GIVEN: A PackageMock entered with a body that raises
    pkg = PackageMock().add_module("fp_test_raise.leaf", "v = 1")

    # WHEN: The body raises an exception
    with pytest.raises(RuntimeError, match="err"):
        with pkg:
            importlib.import_module("fp_test_raise.leaf")
            raise RuntimeError("err")

    # THEN: Cleanup still ran
    assert not any(f.__class__.__name__ == "_PackageMockFinder" for f in sys.meta_path)
    assert "fp_test_raise" not in sys.modules
    assert "fp_test_raise.leaf" not in sys.modules


def test_reentry_after_exit_works() -> None:
    # GIVEN: A PackageMock that has already been entered and exited
    pkg = PackageMock().add_module("fp_test_reentry.leaf", "v = 'first'")
    with pkg:
        first = importlib.import_module("fp_test_reentry.leaf")
        assert first.v == "first"

    # WHEN: The same instance is entered again
    with pkg:
        second = importlib.import_module("fp_test_reentry.leaf")

        # THEN: Imports succeed against a fresh module instance
        assert second.v == "first"
        assert second is not first


def test_two_disjoint_package_mocks_nest() -> None:
    # GIVEN: Two PackageMock instances with disjoint namespaces
    outer = PackageMock().add_module("fp_test_outer.x", "name = 'outer'")
    inner = PackageMock().add_module("fp_test_inner.x", "name = 'inner'")

    # WHEN: They are nested
    with outer, inner:
        # THEN: Both namespaces resolve correctly
        outer_mod = importlib.import_module("fp_test_outer.x")
        inner_mod = importlib.import_module("fp_test_inner.x")
        assert outer_mod.name == "outer"
        assert inner_mod.name == "inner"

    # AND: Both namespaces are cleaned up on exit
    assert "fp_test_outer.x" not in sys.modules
    assert "fp_test_inner.x" not in sys.modules


def test_add_module_after_enter_raises() -> None:
    # GIVEN: A PackageMock that has been entered
    pkg = PackageMock().add_module("fp_test_sealed.a", "")
    with pkg:
        # WHEN/THEN: add_module raises a RuntimeError
        with pytest.raises(RuntimeError, match="sealed"):
            pkg.add_module("fp_test_sealed.b", "")


def test_invalid_dotted_name_rejected() -> None:
    # GIVEN: A PackageMock
    pkg = PackageMock()

    # WHEN/THEN: add_module rejects names that aren't valid dotted identifiers
    with pytest.raises(ValueError, match="not a valid dotted module name"):
        pkg.add_module("123invalid", "")
    with pytest.raises(ValueError, match="not a valid dotted module name"):
        pkg.add_module("a..b", "")
    with pytest.raises(ValueError, match="not a valid dotted module name"):
        pkg.add_module("", "")


def test_entering_without_modules_raises() -> None:
    # GIVEN: An empty PackageMock
    pkg = PackageMock()

    # WHEN/THEN: __enter__ raises
    with pytest.raises(RuntimeError, match="at least one module"):
        with pkg:
            pass


def test_submodule_resolves_via_meta_path() -> None:
    # GIVEN: A package with two siblings, one importing the other
    pkg = (
        PackageMock()
        .add_module("fp_test_siblings.a", "value = 'A'")
        .add_module(
            "fp_test_siblings.b",
            "from fp_test_siblings import a\nresult = a.value + 'B'",
        )
    )

    # WHEN: The cross-importing module is loaded
    with pkg:
        mod = importlib.import_module("fp_test_siblings.b")

        # THEN: The sibling import resolves through the synthetic finder
        assert mod.result == "AB"


def test_unregistered_names_not_shadowed() -> None:
    # GIVEN: A PackageMock with an unrelated name registered
    pkg = PackageMock().add_module("fp_test_passthrough.x", "")

    # WHEN: A real stdlib module is imported inside the context
    with pkg:
        json_mod = importlib.import_module("json")

        # THEN: The real module is loaded (the mock finder returned None for it)
        assert json_mod.loads('{"a": 1}') == {"a": 1}
        assert "package-mock" not in (json_mod.__file__ or "")


def test_constructor_dict_is_equivalent_to_add_module() -> None:
    # GIVEN: A PackageMock built from the constructor dict shortcut
    pkg = PackageMock({
        "fp_test_ctor.a": "value = 1",
        "fp_test_ctor.b": "value = 2",
    })

    # WHEN/THEN: Both modules import correctly
    with pkg:
        a = importlib.import_module("fp_test_ctor.a")
        b = importlib.import_module("fp_test_ctor.b")
        assert a.value == 1
        assert b.value == 2
