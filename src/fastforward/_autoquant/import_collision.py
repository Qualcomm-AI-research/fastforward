# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import dataclasses
import itertools
import logging
import re

from collections import defaultdict
from collections.abc import Iterator, Sequence, Set

import libcst

from . import pybuilder
from .name_allocation import RenameHelperRefsTransformer, alloc_unique_name
from .pysource.scope import ImportSymbol, is_valid_identifier

logger = logging.getLogger(__name__)


def resolve_import_collisions(builder: pybuilder.ModuleBuilder) -> None:
    """Alias imported symbols that would bind the same local name.

    Autoquant collects required imports per builder and emits their union (see
    `pybuilder.ModuleBuilder.import_statements`). When two symbols from different
    modules share a local binding name emitting both makes the second import
    silently shadow the first and the generated code calls the wrong function.
    Keep one symbol bound to the plain name, rename the others to a
    module-qualified alias, and rewrite the reference sites in the bodies that
    required them.

    This must run after helper functions have been renamed so that all final
    module-level names are known.

    Example:
        If a module requires the two improt statements:
            `from package_a.model import foo`
            `from package_b.model import foo`
        Then this will be rewritten as:
            `from package_a.model import foo`
            `from package_b.model import foo as package_b_foo`
    """
    # Imports of a `ClassBuilder` itself (its base class) are pinned: the
    # reference site is the class' base list rather than a `Name` in a function
    # body, so it cannot be rewritten here. Base class aliases are already made
    # unique by `QuantizedClassNameAllocator`.
    pinned_symbols: set[ImportSymbol] = set()
    for class_builder in builder.classes():
        pinned_symbols.update(class_builder.own_required_imports)

    symbol_owners: dict[ImportSymbol, list[pybuilder.FunctionBuilder]] = defaultdict(list)
    for func_builder in builder.functions():
        for symbol in func_builder.required_imports:
            symbol_owners[symbol].append(func_builder)

    # Bucket every symbol by the name it would bind at module level. Symbols
    # in the same bucket are candidates for collision (they want the same name)
    binding_groups: dict[str, list[ImportSymbol]] = defaultdict(list)
    for symbol in itertools.chain(pinned_symbols, symbol_owners):
        binding_groups[_import_binding_name(symbol)].append(symbol)

    # Names that already exists at module level in the generated code
    used_names: set[str] = {cls_builder.name for cls_builder in builder.classes()}
    used_names.update(func_builder.name for func_builder in builder.functions())
    used_names.update(binding_groups)

    # Resolve collisions for every binding group
    for binding, symbols in sorted(binding_groups.items()):
        # Find objects targeted by import symbols in this bucket.
        # If there are multiple targets, we have collisions to resolve.
        targets = {_import_target(symbol) for symbol in symbols}
        if len(targets) < 2:
            continue

        # Elect a single winner target
        ordered = sorted(symbols, key=_import_sort_key)
        winner = _select_collision_winner(ordered, pinned_symbols)
        winner_target = _import_target(winner)

        # Rewrite all the non-winner target symbols
        for symbol in ordered:
            if _import_target(symbol) == winner_target:
                continue

            if symbol.module is None:
                # Import symbol is `import a.b.c` style
                logger.warning(
                    f"Name conflict between import statements for '{binding}'. "
                    f"Import '{symbol.name}' cannot be renamed, as renaming would turn a dotted "
                    "attribute chain into a bare name. Generated code may refer to the wrong "
                    "imported symbol.",
                )
                continue

            if symbol in pinned_symbols:
                # Reference site (class bases) isn't rewritable
                logger.warning(
                    f"Name conflict between import statements for '{binding}'. "
                    f"Import '{symbol.name}' from '{symbol.module}' cannot be renamed; "
                    f"generated code may refer to the wrong imported symbol.",
                )
                continue

            # Get a valid module-qualified alias (`{foo}_{bar}`) for renaming
            alias = _alloc_import_alias(symbol, binding, used_names)
            aliased = dataclasses.replace(symbol, asname=alias)

            # Rename original symbol in all the functions requiring it
            for owner in symbol_owners.get(symbol, ()):
                same_binding = sum(
                    1 for req in owner.required_imports if _import_binding_name(req) == binding
                )
                if same_binding > 1:
                    # The body binds this name to more than one symbol
                    logger.warning(
                        f"'{owner.name}' requires multiple imports binding the name '{binding}'."
                        "Leaving them unaliased because is not possible to disambiguate references. "
                        "Generated code may refer to the wrong imported symbol.",
                    )
                    continue

                # Rename every name reference with its aliased version in `owner` function
                owner.required_imports = tuple(
                    aliased if req == symbol else req for req in owner.required_imports
                )
                renamed = owner.cst.visit(
                    RenameHelperRefsTransformer({binding: libcst.Name(alias)})
                )
                assert isinstance(renamed, libcst.FunctionDef)
                owner.cst = renamed


def _import_binding_name(symbol: ImportSymbol) -> str:
    """Return the local name that an import symbol introduces into the module's namespace."""
    # import a as x -> binds x
    if symbol.asname is not None:
        return symbol.asname
    # from a.b import c -> binds c
    if symbol.module is not None:
        return symbol.name
    # import a.b.c -> binds a
    return symbol.name.split(".")[0]


def _import_target(symbol: ImportSymbol) -> tuple[str, str]:
    """Return the object that `symbol` binds, for collision detection.

    Two symbols that bind the same name but resolve to the same target are
    compatible. In particular `import torch` and `import torch.nn.functional`
    both bind `torch` to the same package and may coexist.
    """
    if symbol.module is None and symbol.asname is None:
        return ("", symbol.name.split(".")[0])
    return (symbol.module or "", symbol.name)


def _import_sort_key(symbol: ImportSymbol) -> tuple[str, str, str]:
    """Order symbols as `pybuilder.ModuleBuilder.import_statements` emits them."""
    return (symbol.module or "", symbol.name, symbol.asname or "")


def _sanitize_identifier_part(part: str) -> str:
    """Coerce a single dotted-module component into a valid identifier fragment."""
    sanitized = re.sub(r"\W", "_", part)
    if sanitized and sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized


def _alias_prefix_candidates(module: str) -> Iterator[str]:
    """Yield module-derived alias prefixes, shortest and most specific first.

    For `transformers.models.pixtral.modeling_pixtral` this yields `pixtral`,
    `models_pixtral`, `transformers_models_pixtral` and finally the full dotted
    path joined by underscores. The containing package is preferred over the
    module itself because it is both shorter and more recognizable
    (`pixtral` rather than `modeling_pixtral`).
    """
    parts = [_sanitize_identifier_part(part) for part in module.split(".")]
    for start in range(len(parts) - 2, -1, -1):
        yield "_".join(parts[start:-1])
    yield "_".join(parts)


def _alloc_import_alias(symbol: ImportSymbol, binding: str, used_names: set[str]) -> str:
    """Allocate a module-qualified alias for `symbol`, avoiding names in `used_names`."""
    for prefix in _alias_prefix_candidates(symbol.module or ""):
        candidate = f"{prefix}_{binding}"
        if is_valid_identifier(candidate) and candidate not in used_names:
            used_names.add(candidate)
            return candidate
    return alloc_unique_name(binding, used_names)


def _select_collision_winner(
    ordered: Sequence[ImportSymbol], pinned_symbols: Set[ImportSymbol]
) -> ImportSymbol:
    """Pick the symbol of a colliding group that keeps the plain binding name.

    Symbols that cannot be aliased take precedence, so that the symbols that
    must be renamed are the ones that can be.
    """
    for symbol in ordered:
        if symbol in pinned_symbols:
            return symbol
    for symbol in ordered:
        if symbol.module is None:
            return symbol
    return ordered[0]
