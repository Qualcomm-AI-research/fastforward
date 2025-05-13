# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import tempfile

from pathlib import Path


def create_output_directory():
    return Path(tempfile.TemporaryDirectory().name)
