# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from .quantizer import Quantizer as Quantizer
from .quantizer import QuantizerMetadata as QuantizerMetadata
from .quantizer import QuantizerStub as QuantizerStub

# isort: split
# Import Quantizer before QuantizedModule
from .quantized_module import QuantizedModule as QuantizedModule
from .quantized_module import quantize_model as quantize_model
from .quantized_module import quantized_module_map as quantized_module_map

# isort: split
from . import functional as functional
from .activations import QuantizedRelu as QuantizedRelu
from .activations import QuantizedSilu as QuantizedSilu
from .container import QuantizedModuleDict as QuantizedModuleDict
from .container import QuantizedModuleList as QuantizedModuleList
from .container import QuantizedParameterDict as QuantizedParameterDict
from .container import QuantizedParameterList as QuantizedParameterList
from .container import QuantizedSequential as QuantizedSequential
from .conv import QuantizedConv1d as QuantizedConv1d
from .conv import QuantizedConv2d as QuantizedConv2d
from .dynamic_linear_quantizer import DynamicLinearQuantizer as DynamicLinearQuantizer
from .embedding import QuantizedEmbedding as QuantizedEmbedding
from .linear import QuantizedLinear as QuantizedLinear
from .linear_quantizer import LinearQuantizer as LinearQuantizer
from .normalization import QuantizedLayerNorm as QuantizedLayerNorm
