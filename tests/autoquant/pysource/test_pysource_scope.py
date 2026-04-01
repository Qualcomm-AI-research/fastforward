# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Any, cast

import libcst
import pytest

from fastforward._autoquant.pybuilder.builder import ModuleBuilder
from fastforward._autoquant.pysource.scope import (
    ImportSymbol,
    _resolve_relative_module_name,
    find_required_imports,
    infer_scopes,
)


@pytest.mark.parametrize(
    ("relative_root", "relative_levels", "module_name", "expected"),
    [
        ("path.tocurrent.module", 0, "other.submodule", "other.submodule"),
        ("path.tocurrent.module", 1, "other.submodule", "path.tocurrent.other.submodule"),
        ("path.tocurrent.module", 2, "other.submodule", "path.other.submodule"),
        # Going beyond package root is clamped to the current package root.
        ("path.tocurrent.module", 3, "other.submodule", "path.tocurrent.other.submodule"),
        ("tree", 1, "sequence", "tree.sequence"),
        ("tree", 1, None, "tree"),
        # With no imported module, over-deep traversal is clamped to package root.
        ("tree.leaf", 2, None, "tree"),
        ("tree.leaf", 3, None, "tree"),
        # Reproduces invalid autoquant import like
        # `from diffusers.models.utils import deprecate` when the original source
        # uses `from ...utils import deprecate` inside a nested module such as
        # `diffusers.models.unets.unet_2d_condition`.
        ("diffusers.models.unets.unet_2d_condition", 3, "utils", "diffusers.utils"),
    ],
)
def test_resolve_relative_module_name(
    relative_root: str, relative_levels: int, module_name: str | None, expected: str
) -> None:
    assert _resolve_relative_module_name(relative_root, relative_levels, module_name) == expected


@pytest.mark.parametrize(
    ("module_lines", "module_name", "expected_import"),
    [
        (
            [
                "from .sequence import _sequence_like",
                "",
                "def demo():",
                "    return _sequence_like(1)",
            ],
            "tree",
            ImportSymbol(name="_sequence_like", module="tree.sequence"),
        ),
        (
            [
                "from ...utils import deprecate",
                "",
                "def demo(msg):",
                '    return deprecate("scale", "1.0.0", msg)',
            ],
            "diffusers.models.unets.unet_2d_condition",
            ImportSymbol(name="deprecate", module="diffusers.utils"),
        ),
    ],
)
def test_find_required_imports_resolves_relative_imports(
    module_lines: list[str], module_name: str, expected_import: ImportSymbol
) -> None:
    module = libcst.parse_module("\n".join(module_lines))
    funcdef = module.body[1]
    assert isinstance(funcdef, libcst.FunctionDef)

    scopes = infer_scopes(module)
    function_scope = scopes[funcdef]

    imports = find_required_imports(funcdef, function_scope, module_name=module_name)
    assert expected_import in imports


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [
        (ImportSymbol(name="torch"), True),
        (ImportSymbol(name="torch.nn.functional"), True),
        (ImportSymbol(name="sigmoid", module="torch"), True),
        (
            ImportSymbol(
                name="Siglip2EncoderLayer",
                module="transformers_modules.Eagle-Block2A-2B-v2.modeling_siglip2",
            ),
            False,
        ),
        (ImportSymbol(name="invalid-name", module="torch.nn"), False),
        (ImportSymbol(name="valid_name", module="torch.nn", asname="invalid-alias"), False),
    ],
)
def test_import_symbol_is_valid(symbol: ImportSymbol, expected: bool) -> None:
    if not hasattr(symbol, "is_valid"):
        pytest.xfail("Phase 2 implementation pending: ImportSymbol.is_valid()")
    assert symbol.is_valid() is expected


class _StubStatementBuilder:
    def __init__(self, required_imports: tuple[ImportSymbol, ...]) -> None:
        self.required_imports = required_imports

    def build(self, quantizer_refs: object) -> libcst.SimpleStatementLine:
        del quantizer_refs
        statement = libcst.parse_statement("pass")
        assert isinstance(statement, libcst.SimpleStatementLine)
        return statement


def _render_imports(imports: tuple[ImportSymbol, ...]) -> str:
    module_builder = ModuleBuilder(origin=None)
    module_builder.add_function(cast(Any, _StubStatementBuilder(required_imports=imports)))
    module = libcst.Module([])
    return "\n".join(module.code_for_node(stmt) for stmt in module_builder.import_statements())


def test_import_statements_uses_getattr_for_invalid_attribute_name() -> None:
    if not hasattr(ImportSymbol, "is_valid"):
        pytest.xfail("Phase 2 implementation pending: dynamic import fallback emission")

    rendered = _render_imports((
        ImportSymbol(name="invalid-attr", module="torch.nn.functional", asname="local_name"),
    ))

    assert "import importlib" in rendered
    assert (
        'local_name = getattr(importlib.import_module("torch.nn.functional"), "invalid-attr")'
        in rendered
    )


def test_import_statements_dedupes_importlib_and_keeps_valid_imports() -> None:
    if not hasattr(ImportSymbol, "is_valid"):
        pytest.xfail("Phase 2 implementation pending: importlib dedup/order behavior")

    rendered = _render_imports((
        ImportSymbol(
            name="Siglip2EncoderLayer",
            module="transformers_modules.Eagle-Block2A-2B-v2.modeling_siglip2",
        ),
        ImportSymbol(
            name="AnotherLayer", module="transformers_modules.Eagle-Block2A-2B-v2.modeling_siglip2"
        ),
        ImportSymbol(name="torch"),
    ))

    assert rendered.count("import importlib") == 1
    assert "import torch" in rendered
    assert (
        'Siglip2EncoderLayer = importlib.import_module("transformers_modules.Eagle-Block2A-2B-v2.modeling_siglip2").Siglip2EncoderLayer'
        in rendered
    )
    assert (
        'AnotherLayer = importlib.import_module("transformers_modules.Eagle-Block2A-2B-v2.modeling_siglip2").AnotherLayer'
        in rendered
    )
