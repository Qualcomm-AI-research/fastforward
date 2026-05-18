# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import torch

from packaging import version

TORCH_VERSION = version.parse(torch.__version__.split("+")[0])
OPSET_VERSION = 18 if TORCH_VERSION.release[:2] == (2, 8) else 17
