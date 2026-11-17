# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import os

import mkdocs.plugins

from mkdocs.config.defaults import MkDocsConfig


@mkdocs.plugins.event_priority(100)
def on_config(config: MkDocsConfig) -> MkDocsConfig | None:
    """Silence `datasets` warnings in executed notebooks.

    `datasets` logs the local cache path when loading a cached dataset, which would
    otherwise end up in the published notebook outputs. Notebook kernels inherit this
    environment variable from the mkdocs process.
    """
    os.environ.setdefault("DATASETS_VERBOSITY", "error")
    return config
