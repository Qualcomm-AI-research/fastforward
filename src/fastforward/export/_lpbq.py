# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


from typing import Any, cast

import torch

import fastforward as ff

from fastforward.export._export_types import ProcessedQuantParams
from fastforward.quantization.granularity import granularity_from_sizes


class LPBQProcessor:
    """Handles conversion from PerBlock quantization to LPBQ format."""

    def __init__(self, compressed_bw: int = 4, decompressed_bw: int = 8) -> None:
        self.compressed_bw = compressed_bw
        self.decompressed_bw = decompressed_bw

    def __new__(cls, compressed_bw: int = 4, decompressed_bw: int = 8) -> "LPBQProcessor":
        """Validate LPBQ configuration parameters."""
        if compressed_bw <= 0 or decompressed_bw <= 0:
            msg = f"Bitwidths cannot be 0 or negative (compressed_bitwidth={compressed_bw}, "
            msg += f"decompressed_bitwdith={decompressed_bw})"
            raise ValueError(msg)

        if compressed_bw >= decompressed_bw:
            msg = "Compressed bitwidth cannot be larger than decompressed bitwidth "
            msg += f"(compressed_bitwidth={compressed_bw}, "
            msg += f"decompressed_bitwdith={decompressed_bw})"
            raise ValueError(msg)

        if compressed_bw > 8:
            msg = f"Compressed bitwidth can be max 8, got compressed_bitwidth={compressed_bw}"
            raise ValueError(msg)

        if decompressed_bw > 32:
            msg = (
                f"Decompressed bitwidth can be max 32, got decompressed_bitwidth={decompressed_bw}"
            )
            raise ValueError(msg)

        instance = super().__new__(cls)
        return instance

    def can_export_as_lpbq(self, processed_params: ProcessedQuantParams) -> bool:
        """Determine if the quantization parameters can be exported as LPBQ.

        For the transformation from PerBlock to LPBQ the following criteria must
        be met:
            - LPBQ must be enabled
            - PerBlock granularity with specific patterns must be used
            - The quantization must be symmetric
            - Original bitwidth must match the compressed_bw setting
        """
        granularity = granularity_from_sizes(
            processed_params.data_shape, processed_params.tile_size
        )

        return (
            isinstance(granularity, ff.PerBlock)
            and len(granularity.block_dims) == 1
            and len(granularity.per_channel_dims) == 1
            and processed_params.is_symmetric is True
            and processed_params.bitwidth == self.compressed_bw
        )

    def generate_lpbq_encoding(
        self,
        tensor_name: str,
        processed_params: ProcessedQuantParams,
    ) -> dict[str, Any]:
        """Generate LPBQ encoding from suitable quantization parameters."""
        if not self.can_export_as_lpbq(processed_params):
            msg = f"Parameters for {tensor_name} not suitable for LPBQ"
            raise ValueError(msg)

        scale = processed_params.scale
        granularity = granularity_from_sizes(
            processed_params.data_shape, processed_params.tile_size
        )

        scale_2d_shape = self._infer_scale_2d_shape(
            scale.nelement(), processed_params.data_shape, granularity
        )

        scale_2d = scale.reshape(scale_2d_shape)

        block_grouping = self._create_block_grouping(granularity)

        decompressed_bw = self.decompressed_bw
        compressed_bw = self.compressed_bw
        block_size = cast(ff.PerBlock, granularity).block_sizes[0]

        per_block_int_scale, per_channel_float_scale = self.grouped_dynamic_quantize(
            scale_2d, block_grouping, compressed_bw
        )

        return {
            "name": tensor_name,
            "dtype": "INT",
            "enc_type": "LPBQ",
            "is_sym": True,
            "compressed_bw": compressed_bw,
            "bw": decompressed_bw,
            "block_size": block_size,
            "per_block_int_scale": per_block_int_scale.to(torch.uint32).flatten().tolist(),
            "scale": per_channel_float_scale.flatten().tolist(),
            "offset": [float(-(2 ** (decompressed_bw - 1)))]
            * len(per_channel_float_scale.flatten()),
        }

    def grouped_dynamic_quantize(
        self, input_array: torch.Tensor, grouping: list[int], bitwidth: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply grouped dynamic quantization to compress scale values.

        Splits the input into groups, computes per-group scale factor, and
        quantizes each group to the specified bitwidth.

        Args:
            input_array: 2D array of scale values to compress
            grouping: List defining how to group dimensions (has only two values,
                -1 for grouping, 1 for not grouping)
            bitwidth: Target bitwidth for compressed values
        Returns:
            Tuple of the quantized grouped scale (int) and the original grouped scale (float)
        """
        dynamic_scale = self.get_per_group_scale_factor(input_array, grouping, bitwidth)
        grouped_scale = self.split_blocks(input_array, grouping)

        quantized_input = torch.clamp(
            torch.round(grouped_scale / dynamic_scale), 1, 2**bitwidth
        ).to(torch.uint32)

        return quantized_input.reshape(input_array.shape), dynamic_scale

    def split_blocks(self, encoding: torch.Tensor, block_grouping: list[int]) -> torch.Tensor:
        """Split encoding array to expose block structure."""
        expanded_shape = []

        for idx, block_group in enumerate(block_grouping):
            if block_group == -1:
                expanded_shape.extend([1, encoding.shape[idx]])
            else:
                expanded_shape.extend([encoding.shape[idx] // block_group, block_group])
        return encoding.reshape(expanded_shape)

    def get_per_group_scale_factor(
        self, scale: torch.Tensor, block_grouping: list[int], scale_bitwidth: int
    ) -> torch.Tensor:
        """Compute per-group scale factor.

        Finds the maximum value in each group and computes the scale factor
        needed to fit the group's range into the target bitwidth.
        """
        grouped_scale = self.split_blocks(scale, block_grouping)
        group_axes = tuple(range(1, len(grouped_scale.shape), 2))
        max_scale = torch.amax(grouped_scale, dim=group_axes, keepdim=True)
        result = max_scale / torch.tensor(
            2**scale_bitwidth, dtype=max_scale.dtype, device=max_scale.device
        )

        return result

    def _infer_scale_2d_shape(
        self,
        scale_1d_size: int,
        data_shape: tuple[int, ...],
        granularity: ff.granularity.Granularity,
    ) -> tuple[int, int]:
        """Infer 2D shape for reshaping 1D scale array based on granularity pattern.

        Determines how to reshape the flattened scales into 2D arrays based on data tensor
        dimensions, block size/dimension and per-channel dimension.
        """
        match granularity:
            case ff.PerBlock(block_dims=(1,), per_channel_dims=(0,)):
                out_channels = data_shape[0]
                block_size = granularity.block_sizes[0]
                blocks_per_channel = data_shape[1] // block_size
                expected_size = out_channels * blocks_per_channel

                if expected_size == scale_1d_size:
                    return (out_channels, blocks_per_channel)
                else:
                    msg = f"LPBQ scale size mismatch for supported pattern: expected {expected_size}, got {scale_1d_size}."
                    raise ValueError(msg)

            case ff.PerBlock(block_dims=(0,), per_channel_dims=(1,)):
                in_channels = data_shape[1]
                block_size = granularity.block_sizes[0]
                blocks_per_channel = data_shape[0] // block_size
                expected_size = in_channels * blocks_per_channel

                if expected_size == scale_1d_size:
                    return (blocks_per_channel, in_channels)
                else:
                    msg = f"LPBQ scale size mismatch for supported pattern: expected {expected_size}, got {scale_1d_size}."
                    raise ValueError(msg)
            case _:
                msg = "Supported LPBQ configurations include PerBlock(block_dims=(1,), per_channel_dims=(0,)) "
                msg += f"or PerBlock(block_dims=(0,), per_channel_dims=(1,)). Instead got granularity={granularity}"
                raise ValueError(msg)

    def _create_block_grouping(self, granularity: ff.granularity.Granularity) -> list[int]:
        """Create grouping pattern for 2D scale based on granularity."""
        match granularity:
            case ff.PerBlock(block_dims=(1,), per_channel_dims=(0,)):
                return [1, -1]
            case ff.PerBlock(block_dims=(0,), per_channel_dims=(1,)):
                return [-1, 1]
            case _:
                msg = "Supported LPBQ configurations include PerBlock(block_dims=(1,), per_channel_dims=(0,)) "
                msg += f"or PerBlock(block_dims=(0,), per_channel_dims=(1,)). Instead got granularity={granularity}"
                raise ValueError(msg)
