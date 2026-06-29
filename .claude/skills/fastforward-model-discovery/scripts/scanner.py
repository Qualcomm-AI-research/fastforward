# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Scan a model source directory for torch.nn.Module subclasses and factory patterns."""

from __future__ import annotations

import ast
import json
import logging
import sys

from pathlib import Path

_PRIORITY_DIRS = frozenset({"examples", "scripts", "tests", "demo", "notebooks"})

_FACTORY_NAMES = frozenset({
    "from_pretrained",
    "build_model",
    "create_model",
    "load_model",
    "load_checkpoint",
    "get_model",
    "make_model",
    "build",
    "load",
})


def _read_py_files(root: Path) -> list[Path]:
    return [f for f in root.rglob("*.py") if f.is_file() and "site-packages" not in str(f)]


def _prioritised(files: list[Path]) -> list[Path]:
    priority, rest = [], []
    for f in files:
        if {p.lower() for p in f.parts} & _PRIORITY_DIRS:
            priority.append(f)
        else:
            rest.append(f)
    return priority + rest


def _module_subclass_names(tree: ast.Module) -> list[str]:
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if "Module" in ast.unparse(base):
                    names.append(node.name)
                    break
    return names


def _factory_call_names(tree: ast.Module) -> list[str]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr in _FACTORY_NAMES:
                found.append(ast.unparse(node.func))
            elif isinstance(node.func, ast.Name) and node.func.id in _FACTORY_NAMES:
                found.append(node.func.id)
    return found


def _dedupe(items: list[str], limit: int = 5) -> list[str]:
    return list(dict.fromkeys(items))[:limit]


def scan(root: Path) -> dict:
    """Scan root for torch.nn.Module subclasses and factory call patterns."""
    py_files = _read_py_files(root)
    module_classes: list[str] = []
    factory_calls: list[str] = []
    parse_errors = 0

    for py_file in _prioritised(py_files)[:200]:
        try:
            tree = ast.parse(py_file.read_text(errors="replace"))
            module_classes.extend(_module_subclass_names(tree))
            factory_calls.extend(_factory_call_names(tree))
        except Exception as exc:  # noqa: BLE001
            parse_errors += 1
            logging.debug("Parse error in %s: %s", py_file, exc)

    return {
        "module_classes": _dedupe(module_classes),
        "factory_calls": _dedupe(factory_calls),
        "parse_errors": parse_errors,
        "files_scanned": len(py_files),
    }


def main() -> None:
    """CLI entry point: scan a model directory and print JSON results."""
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scanner.py <model_path>")
    result = scan(Path(sys.argv[1]).expanduser().resolve())
    print(json.dumps(result))


if __name__ == "__main__":
    main()
