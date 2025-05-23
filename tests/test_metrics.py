# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from fastforward.quantization.random import random_quantized
from fastforward.testing.metrics import sqnr


def test_sqnr() -> None:
    # GIVEN a random torch tensor.
    quantized_tensor = random_quantized((100, 100), num_bits=16)
    dequantized_tensor = quantized_tensor.dequantize()

    # WHEN calculating the SQNR between the dequantized tensor and itself, and the quantized tensor and itself.
    sqnr_score = sqnr(dequantized_tensor, dequantized_tensor)

    # THEN the SQNR score in db should be close to 150 db, given the default eps of 1e-15.
    # For test stability we set the bar to 100.
    assert sqnr_score > 100.0

    # THEN the SQNR score of the dequantized and itself should be the same for any combination of dequantized/quantized.
    assert sqnr_score == sqnr(quantized_tensor, quantized_tensor)
    assert sqnr_score == sqnr(dequantized_tensor, quantized_tensor)
    assert sqnr_score == sqnr(quantized_tensor, dequantized_tensor)

    # WHEN calculating the SQNR between the dequantized tensor and itself reversed.
    sqnr_reversed_score = sqnr(dequantized_tensor, -dequantized_tensor, in_db=True)

    # THEN the SQNR score in db should be close to -6db. For test stability we set the bar to 0.
    assert sqnr_reversed_score < 0.0
