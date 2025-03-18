# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


from .function import AffineQuantizationFunction as AffineQuantizationFunction
from .function import DynamicAffineQuantParams as DynamicAffineQuantParams
from .function import DynamicParamInferenceFn as DynamicParamInferenceFn
from .function import StaticAffineQuantParams as StaticAffineQuantParams
from .range import integer_maximum as integer_maximum
from .range import integer_minimum as integer_minimum
from .range import parameters_for_range as parameters_for_range
from .range import quantization_range as quantization_range
from .static import quantization_context as quantization_context
from .static import quantize_by_tile as quantize_by_tile
from .static import quantize_per_block as quantize_per_block
from .static import quantize_per_channel as quantize_per_channel
from .static import quantize_per_granularity as quantize_per_granularity
from .static import quantize_per_tensor as quantize_per_tensor
