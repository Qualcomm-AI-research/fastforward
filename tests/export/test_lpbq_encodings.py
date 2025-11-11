# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
from typing import Any

import pytest
import torch

from fastforward.export._export_schemas import V1SchemaHandler
from fastforward.export._export_types import ProcessedQuantParams, QuantParametersDict
from fastforward.export._lpbq import LPBQProcessor


@pytest.mark.parametrize(
    "lpbq_overrides",
    [
        {"compressed_bw": 8},  # 8 >= 8
        {"compressed_bw": 16, "decompressed_bw": 8},  # 16 > 8
        {"compressed_bw": 0},  # Invalid bitwidth
        {"decompressed_bw": 0},  # Invalid bitwidth
    ],
)
def test_lpbq_invalid_lpbq_processor_creation(lpbq_overrides: dict[str, Any]) -> None:
    """Test that invalid settings raise ValueError at creation."""
    # GIVEN some invalid LPBQ settings
    lpbq_settings: dict[str, Any] = {
        "compressed_bw": 4,
        "decompressed_bw": 8,
    }
    lpbq_settings.update(lpbq_overrides)

    # WHEN creating a LPBQ Processor instance
    # THEN an error should be raised.
    with pytest.raises(ValueError):
        LPBQProcessor(**lpbq_settings)


@pytest.mark.parametrize(
    "data_shape,tile_size,lpbq_overrides,bitwidth,should_fail",
    [
        # Cases that should fail at eligibility check (valid settings)
        ((64, 128), (1, 1), {}, 4, True),  # Wrong granularity pattern
        ((64, 128), (4, 1), {"compressed_bw": 8, "decompressed_bw": 16}, 4, True),  # Wrong bitwidth
        ((64, 128), (4, 128), {}, 4, True),  # Creates PerBlock without per_channel_dims
        # Cases that should pass
        ((16, 32), (4, 1), {}, 4, False),
        ((64, 128), (4, 1), {}, 4, False),
    ],
)
def test_lpbq_eligibility_check(
    data_shape: tuple[int, ...],
    tile_size: tuple[int, ...],
    lpbq_overrides: dict[str, Any],
    bitwidth: int,
    should_fail: bool,
) -> None:
    """Test LPBQ eligibility logic with valid settings."""
    # GIVEN a combination of LPBQ settings
    lpbq_settings: dict[str, Any] = {
        "compressed_bw": 4,
        "decompressed_bw": 8,
    }

    lpbq_settings.update(lpbq_overrides)

    # WHEN creating the LPBQ processor.
    processor = LPBQProcessor(**lpbq_settings)  # Should not raise ValueError

    block_count = data_shape[0] // tile_size[0]
    channel_count = data_shape[1] // tile_size[1]
    scales = torch.rand(block_count, channel_count) * 0.1 + 0.001

    processed_params = _create_test_params(
        scales, torch.Size(data_shape), torch.Size(tile_size), bitwidth
    )

    # THEN the incorrect cases should raise an error.
    if should_fail:
        with pytest.raises(ValueError):
            processor.generate_lpbq_encoding("test", processed_params)
    # THEN the correct cases should create a new LPBQ encoding.
    else:
        encoding = processor.generate_lpbq_encoding("test", processed_params)
        assert encoding["enc_type"] == "LPBQ"


def test_lpbq_scale_decomposition_correctness() -> None:
    """Test that per_block_int_scale * per_channel_float_scale == original_scale."""
    # GIVEN a LPBQ processor and the original scale values
    processor = LPBQProcessor(compressed_bw=4, decompressed_bw=8)

    base_pattern = torch.tensor([0.01, 0.02, 0.015, 0.025] * 32, dtype=torch.float32)
    block_multipliers = 1.0 + torch.arange(16, dtype=torch.float32) * 0.1

    original_scale = base_pattern.unsqueeze(0) * block_multipliers.unsqueeze(1)

    data_shape, tile_size = (64, 128), (4, 1)
    bitwidth = 4

    processed_params = _create_test_params(
        original_scale, torch.Size(data_shape), torch.Size(tile_size), bitwidth
    )

    # WHEN generating the LPBQ encodings
    encoding = processor.generate_lpbq_encoding("test_weight", processed_params)

    # THEN the encodings should be created
    per_block_int_scale = torch.tensor(encoding["per_block_int_scale"])
    per_channel_float_scale = torch.tensor(encoding["scale"])

    per_block_int_scale_2d = per_block_int_scale.reshape(16, 128)

    per_channel_float_scale_2d = per_channel_float_scale.reshape(1, 128).expand(16, 128)

    # THEN it should be possible to reconstruct the original scales (with an error tolerance)
    reconstructed_scale = per_block_int_scale_2d * per_channel_float_scale_2d

    torch.testing.assert_close(
        reconstructed_scale.flatten(),
        original_scale.flatten(),
        rtol=0.15,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "compressed_bw,decompressed_bw,should_be_valid",
    [
        # Valid combinations
        (2, 8, True),
        (3, 8, True),
        (4, 8, True),
        (4, 16, True),
        (3, 16, True),
        (2, 16, True),
        (4, 32, True),
        (1, 8, True),
        # Invalid combinations - should fail at instance creation
        (8, 8, False),
        (16, 8, False),
        (8, 4, False),
        (0, 8, False),
        (4, 0, False),
        (-1, 8, False),
        (4, -1, False),
        (10, 16, False),
        (4, 64, False),
    ],
)
def test_lpbq_bitwidth_combinations(
    compressed_bw: int, decompressed_bw: int, should_be_valid: bool
) -> None:
    """Test various compressed/decompressed bitwidth combinations."""
    # GIVEN an incorrectly setup LPBQProcessor
    # WHEN attempting to create LPBQProcessor instance
    # THEN a ValueError is returned.
    if not should_be_valid:
        # Invalid settings should fail at creation
        with pytest.raises(ValueError):
            processor = LPBQProcessor(
                compressed_bw=compressed_bw,
                decompressed_bw=decompressed_bw,
            )
    else:
        # GIVEN valid LPBQ settings
        processor = LPBQProcessor(
            compressed_bw=compressed_bw,
            decompressed_bw=decompressed_bw,
        )

        # WHEN generating LPBQ encodings
        scales = torch.rand(16, 128) * 0.1 + 0.001
        processed_params = _create_test_params(
            scales, torch.Size((64, 128)), torch.Size((4, 1)), compressed_bw
        )

        # THEN LPBQ encodings can be created
        encoding = processor.generate_lpbq_encoding("test_valid", processed_params)
        assert encoding["enc_type"] == "LPBQ"


@pytest.mark.parametrize(
    "compressed_bw,decompressed_bw",
    [
        (2, 8),
        (3, 8),
        (4, 8),
        (4, 16),
        (3, 16),
        (2, 16),
        (4, 32),
        (1, 8),
    ],
)
def test_lpbq_compression_decompression(compressed_bw: int, decompressed_bw: int) -> None:
    # GIVEN a LPBQ Processor
    processor = LPBQProcessor(
        compressed_bw=compressed_bw,
        decompressed_bw=decompressed_bw,
    )

    # WHEN generating LPBQ encodings
    scales = torch.rand(16, 128) * 0.1 + 0.001
    processed_params = _create_test_params(
        scales, torch.Size((64, 128)), torch.Size((4, 1)), compressed_bw
    )

    encoding = processor.generate_lpbq_encoding("test_valid", processed_params)

    # THEN the correct bitwidths are stored
    assert encoding["compressed_bw"] == compressed_bw
    assert encoding["bw"] == decompressed_bw

    # THEN the values of the per_block_int_scale are within the appropriate ranges
    per_block_int_scale = torch.tensor(encoding["per_block_int_scale"])
    max_int_value = 2**compressed_bw
    assert torch.all(per_block_int_scale >= 1)
    assert torch.all(per_block_int_scale <= max_int_value)


def test_v1_schema_handler_perblock_to_lpbq_conversion() -> None:
    """Test that V1SchemaHandler automatically converts eligible PerBlock to LPBQ."""
    # GIVEN a V1SchemaHandler with LPBQ enabled
    lpbq_processor = LPBQProcessor(compressed_bw=4, decompressed_bw=8)

    handler = V1SchemaHandler(lpbq_processor=lpbq_processor)

    # GIVEN PerBlock quantization parameters that meet LPBQ criteria
    data_shape = (64, 128)
    tile_size = (4, 1)

    block_count = data_shape[0] // tile_size[0]
    channel_count = data_shape[1] // tile_size[1]

    # Create realistic scale values
    scales = torch.rand(block_count * channel_count) * 0.1 + 0.001

    encoding_dict: QuantParametersDict = {
        "scale": scales,
        "offset": torch.zeros_like(scales),
        "num_bits": 4,
        "tile_size": tile_size,
        "data_shape": data_shape,
    }

    # WHEN adding the encoding to the handler
    handler.add_encoding("test_weight", encoding_dict, is_param=True)

    # THEN the encoding should be converted to LPBQ format
    result = handler.build_encodings_dictionary()

    param_encoding = next(e for e in result["param_encodings"])
    assert param_encoding["name"] == "test_weight"

    # THEN it should be LPBQ format, not standard PER_BLOCK
    assert param_encoding["enc_type"] == "LPBQ"
    assert param_encoding["compressed_bw"] == 4
    assert param_encoding["bw"] == 8
    assert "per_block_int_scale" in param_encoding
    assert "scale" in param_encoding
    assert param_encoding["block_size"] == 4


def test_v1_schema_handler_perblock_fallback_when_lpbq_disabled() -> None:
    """Test that PerBlock stays as PER_BLOCK when LPBQ is disabled."""
    # GIVEN a V1SchemaHandler with LPBQ disabled (default value for
    # lpbq_processor is None)
    handler = V1SchemaHandler()

    # GIVEN the same PerBlock parameters as above
    encoding_dict: QuantParametersDict = {
        "scale": torch.rand(16 * 128) * 0.1 + 0.001,
        "offset": torch.zeros(16 * 128),
        "num_bits": 4,
        "tile_size": (4, 1),
        "data_shape": (64, 128),
    }

    # WHEN adding the encoding
    handler.add_encoding("test_weight", encoding_dict, is_param=True)

    # THEN it should remain as standard PER_BLOCK
    result = handler.build_encodings_dictionary()
    param_encoding = next(e for e in result["param_encodings"])

    assert param_encoding["enc_type"] == "PER_BLOCK"
    assert param_encoding["block_size"] == 4
    assert "compressed_bw" not in param_encoding
    assert "per_block_int_scale" not in param_encoding


def _create_test_params(
    scales: torch.Tensor, data_shape: torch.Size, tile_size: torch.Size, bitwidth: int
) -> ProcessedQuantParams:
    offset = torch.zeros_like(scales)
    qnn_offset = torch.full_like(scales, -(2 ** (bitwidth - 1)))

    return ProcessedQuantParams(
        scale=scales,
        offset=offset,
        qnn_offset=qnn_offset,
        bitwidth=bitwidth,
        is_symmetric=True,
        data_shape=data_shape,
        tile_size=tile_size,
    )
