# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Any, cast

import libcst
import pytest

from fastforward._autoquant.pybuilder.builder import ModuleBuilder
from fastforward._autoquant.pysource.scope import ImportSymbol, _resolve_relative_module_name


def test_resolve_relative_module_name() -> None:
    relative_root = "path.tocurrent.module"
    module_name = "other.submodule"

    resolved = _resolve_relative_module_name(relative_root, 0, module_name)
    assert resolved == "other.submodule"

    resolved = _resolve_relative_module_name(relative_root, 1, module_name)
    assert resolved == "path.tocurrent.other.submodule"

    resolved = _resolve_relative_module_name(relative_root, 2, module_name)
    assert resolved == "path.other.submodule"

    resolved = _resolve_relative_module_name(relative_root, 3, module_name)
    assert resolved == "other.submodule"


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
