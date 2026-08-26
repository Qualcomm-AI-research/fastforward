# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Public API for FastForward's GGUF export stages.

Users pick a shipped :data:`LLAMA_ADAPTER`, :data:`QWEN2_ADAPTER`, or
:data:`QWEN3_ADAPTER`, or construct their own :class:`ArchAdapter` when
targeting a model with a non-standard module tree. The quantization format
is selected via :data:`GGUF_Q4_0` / :data:`GGUF_Q4_1` / :data:`GGUF_Q8_0`
or a custom :class:`GgufQuantFormat`.

Custom packing formats can be defined by implementing a function matching the
:class:`PackFn` protocol and wrapping it in a :class:`GgufQuantFormat`::

    from fastforward.export.stages.gguf import GgufQuantFormat, PackFn

    def pack_q5_0_blocks(int_codes, scales, offsets=None):
        ...  # produce (n_blocks, 22) uint8 tensor
        return packed

    GGUF_Q5_0 = GgufQuantFormat(
        name="Q5_0", num_bits=5, block_size=32, block_bytes=22,
        symmetric=True, pack_fn=pack_q5_0_blocks, file_type=8,
    )
"""

from fastforward.export.stages.gguf._arch import LLAMA_ADAPTER as LLAMA_ADAPTER
from fastforward.export.stages.gguf._arch import QWEN2_ADAPTER as QWEN2_ADAPTER
from fastforward.export.stages.gguf._arch import QWEN3_ADAPTER as QWEN3_ADAPTER
from fastforward.export.stages.gguf._arch import llama_rope_permute as llama_rope_permute
from fastforward.export.stages.gguf._config import GgufSourceConfig as GgufSourceConfig
from fastforward.export.stages.gguf._packing import GGUF_Q4_0 as GGUF_Q4_0
from fastforward.export.stages.gguf._packing import GGUF_Q4_1 as GGUF_Q4_1
from fastforward.export.stages.gguf._packing import GGUF_Q8_0 as GGUF_Q8_0
from fastforward.export.stages.gguf._packing import (
    default_format_registry as default_format_registry,
)
from fastforward.export.stages.gguf.adapter import ArchAdapter as ArchAdapter
from fastforward.export.stages.gguf.adapter import GgufFormatRegistry as GgufFormatRegistry
from fastforward.export.stages.gguf.adapter import GgufQuantFormat as GgufQuantFormat
from fastforward.export.stages.gguf.adapter import PackFn as PackFn
from fastforward.export.stages.gguf.adapter import TensorFusion as TensorFusion
from fastforward.export.stages.gguf.adapter import TensorTransformT as TensorTransformT
