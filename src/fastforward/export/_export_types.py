# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import dataclasses

from typing import Iterable, TypedDict

import torch

from typing_extensions import NotRequired


class QuantParametersDict(TypedDict):
    scale: torch.Tensor | float
    offset: NotRequired[torch.Tensor | float | int | None]
    num_bits: float | int
    tile_size: Iterable[int]
    data_shape: Iterable[int]
    output_dtype: NotRequired[torch.dtype]


@dataclasses.dataclass
class ProcessedQuantParams:
    scale: torch.Tensor
    offset: torch.Tensor
    qnn_offset: torch.Tensor
    bitwidth: int
    is_symmetric: bool
    data_shape: torch.Size
    tile_size: torch.Size


@dataclasses.dataclass(frozen=True)
class QNNDefaultConfig:
    """Default quantization config parameters for QNN operations/parameters missing explicit encodings.

    These settings are embedded in the encodings dictionary under the `quantizer_args` key.
    They are then used by the QNN quantizer(s) as fallback values for operations without
    defined quantization encodings.
    """

    activation_bitwidth: int = 16
    dtype: str = "int"
    is_symmetric: bool = True
    param_bitwidth: int = 8
    per_channel_quantization: bool = True
    quant_scheme: str = "min_max"


@dataclasses.dataclass
class LPBQConfig:
    """Configuration for LPBQ."""

    enabled: bool = False
    compressed_bw: int = 4
    decompressed_bw: int = 8

    def __post_init__(self) -> None:
        """Validate LPBQ configuration parameters."""
        if self.compressed_bw <= 0 or self.decompressed_bw <= 0:
            msg = f"Bitwidths cannot be 0 or negative (compressed_bitwidth={self.compressed_bw}, "
            msg += f"decompressed_bitwdith={self.decompressed_bw})"
            raise ValueError(msg)

        if self.compressed_bw >= self.decompressed_bw:
            msg = "Compressed bitwidth cannot be larger than decompressed bitwidth "
            msg += f"(compressed_bitwidth={self.compressed_bw}, "
            msg += f"decompressed_bitwdith={self.decompressed_bw})"
            raise ValueError(msg)

        if self.compressed_bw > 8:
            msg = f"Compressed bitwidth can be max 8, got compressed_bitwidth={self.compressed_bw}"
            raise ValueError(msg)

        if self.decompressed_bw > 32:
            msg = f"Decompressed bitwidth can be max 32, got decompressed_bitwidth={self.decompressed_bw}"
            raise ValueError(msg)
