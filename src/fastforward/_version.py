# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import importlib_metadata

from packaging.version import Version

try:
    version = Version(importlib_metadata.version("fastforward"))
except Exception:
    version = Version("0.0-dev")
