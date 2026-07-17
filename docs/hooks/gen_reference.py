# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Generate the code reference pages and navigation."""

import ast

from collections.abc import Sequence
from pathlib import Path

import mkdocs_gen_files

from mkdocs_gen_files.nav import Nav


def generate_nav() -> None:
    nav = Nav()

    root = Path(__file__).parent.parent.parent
    src = root / "src"

    nav[("fastforward")] = "summary.md"

    entries: list[tuple[tuple[str, ...], Path, Path, Path, bool]] = []
    for path in sorted(src.rglob("*.py")):
        module_path = path.relative_to(src).with_suffix("")
        doc_path = path.relative_to(src).with_suffix(".md")
        full_doc_path = Path("reference", doc_path)

        parts = tuple(module_path.parts)
        is_package = False

        if parts[-1] == "__init__":
            parts = parts[:-1]
            doc_path = doc_path.with_name("index.md")
            full_doc_path = full_doc_path.with_name("index.md")
            is_package = True
        elif parts[-1] == "__main__":
            continue

        if is_private(parts):
            continue

        entries.append((parts, doc_path, full_doc_path, path, is_package))

    children_of: dict[tuple[str, ...], list[tuple[str, bool, Path]]] = {}
    for parts, _, _, source_path, is_package in entries:
        if len(parts) < 2:
            continue
        parent = parts[:-1]
        children_of.setdefault(parent, []).append((parts[-1], is_package, source_path))

    for parts, doc_path, full_doc_path, path, is_package in entries:
        nav[parts] = doc_path.as_posix()

        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            ident = ".".join(parts)
            fd.write(f"::: {ident}\n")
            if is_package:
                children = sorted(children_of.get(parts, []))
                if children:
                    fd.write("\n### Submodules\n\n")
                    for name, child_is_package, source_path in children:
                        link = f"{name}/index.md" if child_is_package else f"{name}.md"
                        summary = get_module_summary(source_path)
                        if summary:
                            fd.write(
                                f"- [`{name}`]({link}) "
                                f'<span class="doc-submodule-summary">— {summary}</span>\n'
                            )
                        else:
                            fd.write(f"- [`{name}`]({link})\n")

        mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

    with mkdocs_gen_files.open("reference/summary.md", "w") as nav_file:
        nav_file.writelines(nav.build_literate_nav())


def is_private(parts: Sequence[str]) -> bool:
    return any([p.startswith("_") and not p.startswith("__") for p in parts])


def get_module_summary(path: Path) -> str:
    """Return the first sentence of the module-level docstring at ``path``."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, OSError, UnicodeDecodeError):
        return ""
    docstring = ast.get_docstring(tree)
    if not docstring:
        return ""
    text = docstring.strip()
    for i, ch in enumerate(text):
        if ch == "." and (i + 1 == len(text) or text[i + 1].isspace()):
            return text[: i + 1]
    return text.split("\n", 1)[0]


# NB! `__name__` would not be set to usual `__main__` by gen-files mkdocs plugin
# https://github.com/oprypin/mkdocs-gen-files/blob/v0.5.0/mkdocs_gen_files/plugin.py#L39
generate_nav()
