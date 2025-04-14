# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import pytest
import torch

from fastforward.quantization.affine.range import parameters_for_range


@pytest.mark.parametrize("num_bits", (4, 8, 12, 16))
@pytest.mark.parametrize("symmetric", (True, False))
@pytest.mark.parametrize("allow_one_sided", (True, False))
def test_parameters_for_range_with_different_datatypes(
    num_bits: int, symmetric: bool, allow_one_sided: bool
) -> None:
    # GIVEN a lower and upper threshold for a quantization range
    min_range = torch.tensor([-494], dtype=torch.bfloat16)
    max_range = torch.tensor([544], dtype=torch.bfloat16)

    # WHEN the affine quantization parameters are computed using bfloat16 input
    scale_16, offset_16 = parameters_for_range(
        min_range=min_range,
        max_range=max_range,
        num_bits=num_bits,
        symmetric=symmetric,
        allow_one_sided=allow_one_sided,
    )

    # WHEN the affine quantization parameters are computed using float32 input
    scale_32, offset_32 = parameters_for_range(
        min_range=min_range.to(torch.float32),
        max_range=max_range.to(torch.float32),
        num_bits=num_bits,
        symmetric=symmetric,
        allow_one_sided=allow_one_sided,
    )

    # THEN the quantization parameters must match exactly
    torch.testing.assert_close(scale_16, scale_32, rtol=0.0, atol=0.0)
    torch.testing.assert_close(offset_16, offset_32, rtol=0.0, atol=0.0)
