# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import torch

from packaging import version

TORCH_VERSION = version.parse(torch.__version__.split("+")[0])
OPSET_VERSION = 18 if TORCH_VERSION.release[:2] == (2, 8) else 17


def is_torch_version_at_least(min_version: str) -> bool:
    """Return whether the installed torch is at least ``min_version``.

    Compares on the release tuple so local build segments (e.g. ``+cu130``) and
    pre-release tags do not affect the result.
    """
    return TORCH_VERSION.release >= version.parse(min_version).release
