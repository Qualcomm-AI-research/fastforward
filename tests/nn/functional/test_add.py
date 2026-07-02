# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import fastforward.nn.functional as ff
import pytest
import torch


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_add_two_int_tensors(
    device: torch.device | str,
) -> None:
    # GIVEN: two int32 tensors
    a = torch.arange(0, 100, dtype=torch.int32).to(device)
    b = torch.arange(100, 0, -1, dtype=torch.int32).to(device)

    # WHEN: I try to add the two tensors with fastforward functional add:
    c = ff.add(a, b, strict_quantization=False)

    # THEN:
    # - ff.add can be executed without runtime errors.
    # - Output dtype is int32.
    # - Output is exactly the same from torch.add(a, b) and (a+b).
    assert c.dtype == torch.int32
    assert torch.all(c == (a + b))
    assert torch.all(c == torch.add(a, b))
