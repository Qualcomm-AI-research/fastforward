# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Post-training quantization algorithms that optimize one layer at a time."""

from .adaround import adaround as adaround
from .adascale import AdaScaleParameters as AdaScaleParameters
from .adascale import adascale as adascale
from .gptq import gptq as gptq
from .yaqa import yaqa as yaqa
