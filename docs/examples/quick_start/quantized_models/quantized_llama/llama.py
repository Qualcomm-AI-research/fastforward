# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import logging

from fastforward.nn import QuantizedModule
from transformers import LlamaForCausalLM, LlamaModel

logger = logging.getLogger(__name__)


class QuantizedLlamaForCausalLM(LlamaForCausalLM, QuantizedModule):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()

    # Required for LM eval
    def tie_weights(self):
        pass


class QuantizedLlamaModel(LlamaModel, QuantizedModule):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
