# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

"""Generate the code reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files

from mkdocs_gen_files.nav import Nav


def generate_nav():
    nav = Nav()

    root = Path(__file__).parent.parent
    src = root / "src"

    nav[("fastforward")] = "summary.md"
    for path in sorted(src.rglob("*.py")):
        module_path = path.relative_to(src).with_suffix("")
        doc_path = path.relative_to(src).with_suffix(".md")
        full_doc_path = Path("reference", doc_path)

        parts = tuple(module_path.parts)

        if parts[-1] == "__init__":
            parts = parts[:-1]
            doc_path = doc_path.with_name("index.md")
            full_doc_path = full_doc_path.with_name("index.md")
        elif parts[-1] == "__main__":
            continue

        if is_private(parts):
            continue

        nav[parts] = doc_path.as_posix()

        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            ident = ".".join(parts)
            fd.write(f"::: {ident}\n")

        mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

    with mkdocs_gen_files.open("reference/summary.md", "w") as nav_file:
        nav_file.writelines(nav.build_literate_nav())


def is_private(parts):
    return any([p.startswith("_") and not p.startswith("__") for p in parts])


# NB! `__name__` would not be set to usual `__main__` by gen-files mkdocs plugin
# https://github.com/oprypin/mkdocs-gen-files/blob/v0.5.0/mkdocs_gen_files/plugin.py#L39
generate_nav()
