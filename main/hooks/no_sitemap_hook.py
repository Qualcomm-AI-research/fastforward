# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import mkdocs.plugins

from mkdocs.config.defaults import MkDocsConfig


@mkdocs.plugins.event_priority(-50)
def on_config(config: MkDocsConfig) -> MkDocsConfig | None:
    """Remove sitemap.xml from the static templates to avoid binaries in the repo."""
    config.theme.static_templates.remove("sitemap.xml")
    return config
