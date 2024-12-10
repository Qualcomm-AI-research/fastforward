# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import logging

from transformers import LlamaForCausalLM, LlamaModel

from fastforward.nn import QuantizedModule

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
