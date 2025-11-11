# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


from typing import Any

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

    def _lpbq_granularity_and_block_size(
        self, tensor_name: str, processed_params: ProcessedQuantParams
    ) -> tuple[ff.PerBlock, int]:
        """Get granularity and block size and check if the parameters can use LPBQ.

        For the transformation from PerBlock to LPBQ the following criteria must
        be met:
            - PerBlock granularity with specific patterns must be used
            - The quantization must be symmetric
            - Original bitwidth must match the compressed_bw setting
        """
        granularity = granularity_from_sizes(
            processed_params.data_shape, processed_params.tile_size
        )

        if not (
            isinstance(granularity, ff.PerBlock)
            and len(granularity.block_dims) == 1
            and len(granularity.per_channel_dims) == 1
            and processed_params.is_symmetric is True
            and processed_params.bitwidth == self.compressed_bw
        ):
            msg = f"Parameters for {tensor_name} not suitable for LPBQ"
            raise ValueError(msg)

        block_size = granularity.block_sizes[0]
        return granularity, block_size

    def generate_lpbq_encoding(
        self,
        tensor_name: str,
        processed_params: ProcessedQuantParams,
    ) -> dict[str, Any]:
        """Generate LPBQ encoding from suitable quantization parameters."""
        granularity, block_size = self._lpbq_granularity_and_block_size(
            tensor_name, processed_params
        )

        static_encoding = {
            "name": tensor_name,
            "dtype": "INT",
            "enc_type": "LPBQ",
            "is_sym": True,
        }

        # Extract basic parameters
        scale = processed_params.scale

        # Retrieve bitwidth configuration
        decompressed_bw = self.decompressed_bw
        compressed_bw = self.compressed_bw

        # Reshape scale tensor to 2D for processing
        scale_2d_shape = self._infer_scale_2d_shape(processed_params.data_shape, granularity)

        scale_2d = scale.reshape(scale_2d_shape)

        # Determine how to group blocks based on quantization. Grouping will
        # take place in ghe group dimension (signified by a value -1) and not
        # the channel dimension (signified by a value of 1).
        match granularity:
            case ff.PerBlock(block_dims=(1,), per_channel_dims=(0,)):
                block_grouping = [1, -1]
            case ff.PerBlock(block_dims=(0,), per_channel_dims=(1,)):
                block_grouping = [-1, 1]

        # Apply grouped dynamic quantization to compress scales
        per_block_int_scale, per_channel_float_scale = self.grouped_dynamic_quantize(
            scale_2d, block_grouping, compressed_bw
        )

        # Convert quantized/float scales to appopriate encoding format (flatten and list)
        quantized_scales_list = per_block_int_scale.to(torch.uint32).flatten().tolist()
        float_scales_list = per_channel_float_scale.flatten().tolist()

        # Compute offset values and construct expected format (flatten list, matching the scales)
        offset_value = float(-(2 ** (decompressed_bw - 1)))
        offset_list = [offset_value] * len(float_scales_list)

        dynamic_encoding = {
            "compressed_bw": compressed_bw,
            "bw": decompressed_bw,
            "block_size": block_size,
            "per_block_int_scale": quantized_scales_list,
            "scale": float_scales_list,
            "offset": offset_list,
        }

        return {**static_encoding, **dynamic_encoding}

    def grouped_dynamic_quantize(
        self, scale_2d: torch.Tensor, block_grouping: list[int], bitwidth: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply grouped dynamic quantization to compress scale values.

        Splits the input into groups, computes per-group scale factor, and
        quantizes each group to the specified bitwidth.

        Args:
            scale_2d: 2D array of scale values to compress
            block_grouping: List defining how to group dimensions (has only two values,
                -1 for grouping, 1 for not grouping)
            bitwidth: Target bitwidth for compressed values
        Returns:
            Tuple of the quantized grouped scale (int) and the original grouped scale (float)
        """
        expanded_shape = []

        # Create expanded shape for block grouping
        for idx, block_group in enumerate(block_grouping):
            if block_group == -1:
                expanded_shape.extend([1, scale_2d.shape[idx]])
            else:
                expanded_shape.extend([scale_2d.shape[idx] // block_group, block_group])

        grouped_scale = scale_2d.reshape(expanded_shape)

        # Compute per-group scale factor
        group_axes = tuple(range(1, len(grouped_scale.shape), 2))
        max_scale = torch.amax(grouped_scale, dim=group_axes, keepdim=True)
        dynamic_scale = max_scale / torch.tensor(
            2**bitwidth, dtype=max_scale.dtype, device=max_scale.device
        )

        # Quantize the grouped scale
        quantized_input = torch.clamp(
            torch.round(grouped_scale / dynamic_scale), 1, 2**bitwidth
        ).to(torch.uint32)

        return quantized_input.reshape(scale_2d.shape), dynamic_scale

    def _infer_scale_2d_shape(
        self,
        data_shape: tuple[int, ...],
        granularity: ff.PerBlock,
    ) -> tuple[int, int]:
        """Infer 2D shape for reshaping 1D scale array based on granularity pattern.

        Determines how to reshape the flattened scales into 2D arrays based on data tensor
        dimensions, block size/dimension and per-channel dimension.
        """
        block_size = granularity.block_sizes[0]

        match granularity:
            case ff.PerBlock(block_dims=(1,), per_channel_dims=(0,)):
                out_channels, block_dims = data_shape[0], data_shape[1]
                shape = (out_channels, block_dims // block_size)

            case ff.PerBlock(block_dims=(0,), per_channel_dims=(1,)):
                in_channels, block_dims = data_shape[1], data_shape[0]
                shape = (block_dims // block_size, in_channels)

        return shape
