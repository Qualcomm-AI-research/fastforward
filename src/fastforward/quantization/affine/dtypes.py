# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import functools
import logging

import torch


@functools.lru_cache(maxsize=None)
def can_support_bitwidth(dtype: torch.dtype, num_bits: float) -> bool:
    """Check if dtype can store a quantized value without precision loss."""
    available_precision_bits: float
    if dtype.is_complex or dtype.is_floating_point:
        match torch.finfo(dtype).dtype:
            case "bfloat16":
                available_precision_bits = 7
            case "float16":
                available_precision_bits = 10
            case "float32":
                available_precision_bits = 23
            case "float64":
                available_precision_bits = 52
            case "float8_e4m3fn" | "float8_e4m3fnuz":
                available_precision_bits = 3
            case "float8_e5m2" | "float8_e5m2fnuz":
                available_precision_bits = 2
            case _:
                available_precision_bits = num_bits
                logging.getLogger(__name__).warning(
                    f"Unknown mantissa size for {dtype}; precision loss possible."
                )
    else:
        available_precision_bits = torch.iinfo(dtype).bits

    # The first integer that cannot be exactly represented using a floating
    # point representation is (2 ** (mantissa_bits + 1)) + 1. Since fastforward
    # uses a signed quantization representation, we can also leverage the sign
    # bit, providing an extra bit of precision.
    representable_bitwidth = available_precision_bits + 2
    return representable_bitwidth >= num_bits
