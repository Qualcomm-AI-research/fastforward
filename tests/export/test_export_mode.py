# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import torch

import fastforward as ff

from fastforward.quantization.affine.static import quantize_by_tile
from fastforward.quantized_tensor import QuantizedTensor


def test_export_mode() -> None:
    # GIVEN a torch tensor and some quantization parameters
    a = torch.randn(2, 2)

    tile_size = a.size()
    scale = torch.tensor([1.0])
    offset = torch.tensor([0.0])
    num_bits = 8

    # WHEN quantizing the input tensor NOT on export mode
    quant_a = quantize_by_tile(a, scale, offset, tile_size, num_bits)

    # THEN the output type should be a QuantizedTensor object
    assert not ff.get_export_mode()
    assert isinstance(quant_a, QuantizedTensor)

    # WHEN quantizing the input tensor on export mode
    with ff.export_mode(True):
        fake_quant_a = quantize_by_tile(a, scale, offset, tile_size, num_bits)
        # THEN the output type should not be a QuantizedTensor
        assert ff.get_export_mode()
        assert not isinstance(fake_quant_a, QuantizedTensor)

    assert not ff.get_export_mode()  # type: ignore[unreachable]
