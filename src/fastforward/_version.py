# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import importlib_metadata

from packaging.version import Version

try:
    version = Version(importlib_metadata.version("fastforward"))
except Exception:
    version = Version("0.0-dev")
