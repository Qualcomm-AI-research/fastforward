# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

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
