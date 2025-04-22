# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from . import dispatcher as dispatcher
from . import exceptions as exceptions
from . import logging_utils as logging_utils
from . import mpath as mpath
from . import nn as nn
from . import quantization as quantization
from . import range_setting as range_setting
from ._version import version
from .autoquant import autoquantize as autoquantize
from .flags import export_mode as export_mode
from .flags import get_export_mode as get_export_mode
from .flags import get_strict_quantization as get_strict_quantization
from .flags import set_export_mode as set_export_mode
from .flags import set_strict_quantization as set_strict_quantization
from .flags import strict_quantization as strict_quantization
from .nn.quantized_module import quantize_model as quantize_model
from .nn.quantized_module import quantized_module_map as quantized_module_map
from .nn.quantized_module import surrogate_quantized_modules as surrogate_quantized_modules
from .overrides import disable_quantization as disable_quantization
from .overrides import enable_quantization as enable_quantization
from .quantization import _quantizer_impl as _quantizer_impl
from .quantization import affine as affine
from .quantization import granularity as granularity
from .quantization import random as random
from .quantization.quant_init import QuantizationConfig as QuantizationConfig
from .quantization.quant_init import find_quantizers as find_quantizers
from .quantization.strict_quantization import (
    strict_quantization_for_module as strict_quantization_for_module,
)
from .quantized_tensor import QuantizedTensor as QuantizedTensor
from .range_setting import estimate_ranges as estimate_ranges

__version__ = version

PerTensor = granularity.PerTensor
PerChannel = granularity.PerChannel
PerTile = granularity.PerTile
