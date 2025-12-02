# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Tests pertaining to the environment."""

import os

import pytest
import torch

_CUDA_VERSION_ENV_VAR = "VER_CUDA"


def test_environment() -> None:
    """Tests that the CUDA version set by CI matches the version reported by PyTorch.

    For a while, there was an undetected version mismatch, thus this test was added.
    """
    if (cuda_version := os.getenv(_CUDA_VERSION_ENV_VAR)) is None:
        pytest.skip("Environment variable holding CUDA version is not set.")
    assert cuda_version == torch.version.cuda
