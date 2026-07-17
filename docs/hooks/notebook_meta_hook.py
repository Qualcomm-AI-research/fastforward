# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Hide the primary sidebar on notebook (`.nb.py`) tutorial pages.

Notebook pages are authored as jupytext Python files and rendered by
mkdocs-jupyter, so they cannot carry mkdocs YAML front matter the way a
regular `.md` page can. This hook sets ``page.meta['hide']`` to
``["navigation"]`` for every `.nb.py`-derived page, which
mkdocs-material's ``main.html`` template reads to skip rendering the
primary sidebar — the same effect a page could achieve with::

    ---
    hide:
      - navigation
    ---

The TOC sidebar is intentionally left visible so tutorial readers get an
in-page section index.
"""

from typing import Any

import mkdocs.plugins

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.nav import Navigation
from mkdocs.structure.pages import Page

_HIDE = ("navigation",)


@mkdocs.plugins.event_priority(100)
def on_page_context(
    context: dict[str, Any],
    page: Page,
    config: MkDocsConfig,  # ruff:ignore[unused-function-argument]
    nav: Navigation,  # ruff:ignore[unused-function-argument]
) -> dict[str, Any]:
    if page.file.src_path.endswith(".nb.py"):
        hide = page.meta.setdefault("hide", [])
        for item in _HIDE:
            if item not in hide:
                hide.append(item)
    return context
