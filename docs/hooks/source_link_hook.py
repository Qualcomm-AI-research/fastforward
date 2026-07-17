# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Post-process mkdocstrings-python's collapsible source-code blocks into inline "[source]" links pointing at the public repository.

mkdocstrings emits, for each documented class/function::

    <details class="mkdocstrings-source">
      <summary>Source code in <code>src/fastforward/foo.py</code></summary>
      ... syntax-highlighted source ...
    </details>

The hook rewrites each such block into::

    <p class="doc-source-link">
      <a href="{repo_url}/blob/{ref}/src/fastforward/foo.py#L42">[source]</a>
    </p>

- `repo_url` is read from ``mkdocs.yml`` (``repo_url``).
- `ref` is read from the ``DOCS_VERSION`` environment variable (falls
  back to ``main`` when unset). Set this in the mike deploy script so
  links point at the tag being deployed, e.g.::

      DOCS_VERSION=v1.2.0 mike deploy 1.2.0
"""

import os
import re

from typing import Any

import mkdocs.plugins

from mkdocs.config.defaults import MkDocsConfig

_DEFAULT_REF = "main"

_DETAILS_RE = re.compile(
    r'<details\s+class="mkdocstrings-source">\s*'
    r"<summary>[^<]*<code>([^<]+)</code>\s*</summary>"
    r"(.*?)"
    r"</details>",
    re.DOTALL,
)

# The first line number appears inside pygments' line-number gutter, which
# can render as either <span class="normal">42</span> (table format) or
# <span class="linenos">42</span> (inline format). Try both.
_FIRST_LINENO_RE = re.compile(r'<span class="(?:normal|linenos)">\s*(\d+)\s*</span>')


def _first_lineno(details_body: str) -> int:
    match = _FIRST_LINENO_RE.search(details_body)
    if match:
        return int(match.group(1))
    return 1


@mkdocs.plugins.event_priority(-100)
def on_page_content(html: str, page: Any, config: MkDocsConfig, files: Any) -> str:  # ruff:ignore[unused-function-argument]
    repo_url = (config.repo_url or "").rstrip("/")
    if not repo_url:
        return html

    ref = os.environ.get("DOCS_VERSION", _DEFAULT_REF)

    def replace(match: "re.Match[str]") -> str:
        filepath = match.group(1).strip()
        lineno = _first_lineno(match.group(2))
        url = f"{repo_url}/blob/{ref}/{filepath}#L{lineno}"
        return (
            '<p class="doc-source-link">'
            f'<a href="{url}" target="_blank" rel="noopener">[source]</a>'
            "</p>"
        )

    return _DETAILS_RE.sub(replace, html)
