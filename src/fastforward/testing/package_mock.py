# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""In-memory python package mock, useful for testing purposes."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import linecache
import sys
import textwrap

from types import ModuleType, TracebackType
from typing import Iterable

_ORIGIN_PREFIX = "<package-mock:"


def _origin_for(name: str) -> str:
    return f"{_ORIGIN_PREFIX}{name}>"


class PackageMock:
    """Context manager exposing in-memory Python modules through `import`.

    `PackageMock` is a context manager that registers Python source code from
    strings as importable modules. While the context is active, the registered
    names resolve through `sys.meta_path` and behave like a real package: they
    are importable with the standard `import` statement, expose `__file__`,
    `__package__`, `__path__`, and have their source registered in
    `linecache.cache` so `inspect.getsource(...)` works.

    On exit the finder is removed from `sys.meta_path`, the synthetic entries
    are popped from `sys.modules`, and the corresponding `linecache.cache`
    entries are cleared, so tests don't leak state into each other.

    Notes:
     - Modules must be registered before `__enter__`; `add_module` after enter
       raises `RuntimeError`. The instance can be re-entered after exit.
     - Modules are registered by fully-qualified dotted name with their source
       as a string.
     - Source is `textwrap.dedent`-ed so triple-quoted indented strings work.
     - Parent packages are inferred from dots: registering `a.b.c` implicitly
       creates packages `a` and `a.b`.
     - If a name is both explicitly registered AND a parent of another module,
       its source becomes the package's `__init__`-equivalent content.

    Example:
    pkg = PackageMock({"my_pkg.a": "def foo(): return 1"})
    pkg.add_module("my_pkg.b", "from my_pkg import a; bar = a.foo() + 1")
    with pkg:
        import my_pkg.a
        import my_pkg.b
        assert my_pkg.a.foo() == 1
        assert my_pkg.b.bar == 2

    """

    def __init__(self, sources: dict[str, str] | None = None) -> None:
        self._sources: dict[str, str] = {}
        self._finder: _PackageMockFinder | None = None
        if sources:
            for name, src in sources.items():
                self.add_module(name, src)

    def add_module(self, qualified_name: str, source: str = "") -> "PackageMock":
        """Register a module's source. Returns self for chaining."""
        if self._finder is not None:
            raise RuntimeError(
                "PackageMock is sealed; add modules before entering the `with` block"
            )
        if not qualified_name or not all(part.isidentifier() for part in qualified_name.split(".")):
            msg = f"'{qualified_name}' is not a valid dotted module name"
            raise ValueError(msg)
        self._sources[qualified_name] = textwrap.dedent(source)
        return self

    def __enter__(self) -> "PackageMock":
        if not self._sources:
            raise RuntimeError("PackageMock requires at least one module before entering")
        packages = _infer_parent_packages(self._sources.keys())
        # If a registered name is also a parent of another, it's a package
        # whose source is the __init__ content.
        finder_sources = dict(self._sources)
        for parent in packages:
            finder_sources.setdefault(parent, "")
        self._finder = _PackageMockFinder(sources=finder_sources, packages=packages)
        sys.meta_path.insert(0, self._finder)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        finder = self._finder
        self._finder = None
        if finder is None:
            return
        try:
            sys.meta_path.remove(finder)
        except ValueError:
            pass
        # Pop sys.modules in reverse depth order so a parent isn't yanked
        # while a child still references it via __package__.
        for name in sorted(finder.sources.keys(), key=lambda n: n.count("."), reverse=True):
            sys.modules.pop(name, None)
        for name in finder.sources:
            linecache.cache.pop(_origin_for(name), None)


def _infer_parent_packages(names: Iterable[str]) -> set[str]:
    packages: set[str] = set()
    for name in names:
        parts = name.split(".")
        for i in range(1, len(parts)):
            packages.add(".".join(parts[:i]))
    return packages


class _PackageMockFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, *, sources: dict[str, str], packages: set[str]) -> None:
        self.sources = sources
        self.packages = packages

    def find_spec(
        self,
        fullname: str,
        path: object | None,
        target: object | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        del path, target
        if fullname not in self.sources:
            return None
        return importlib.machinery.ModuleSpec(
            fullname,
            self,
            is_package=fullname in self.packages,
            origin=_origin_for(fullname),
        )

    def exec_module(self, module: ModuleType) -> None:
        name = module.__name__
        filename = _origin_for(name)
        source = self.sources[name]
        is_package = name in self.packages

        module.__dict__["__file__"] = filename
        if is_package:
            module.__dict__["__path__"] = []
            module.__dict__["__package__"] = name
        else:
            module.__dict__["__package__"] = name.rpartition(".")[0]

        # Register source with linecache so `inspect.getsource(module)`
        # works for in-memory modules. Always register, even for empty
        # package source, so `inspect.getsource(pkg)` returns "" cleanly.
        linecache.cache[filename] = (
            len(source),
            None,
            source.splitlines(keepends=True),
            filename,
        )

        if source:
            exec(compile(source, filename, "exec"), module.__dict__)
