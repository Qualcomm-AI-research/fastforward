#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Executed in CI to generate a matrix that parameterizes tests."""

import json
import os

# Versions
torch_cuda_python_versions = [
    ("2.4.1", "12.1", "3.10"),
    ("2.5.1", "12.1", "3.10"),
    ("2.6.0", "12.4", "3.10"),
    ("2.6.0", "12.4", "3.12"),
    # ("2.7.1", "12.6", "3.10"),  # currently broken, cf. #453
    # ("2.7.1", "12.6", "3.12"),  # currently broken, cf. #453
    # ("2.9.0", "12.6", "3.10"),  # currently broken, cf. #453, #454
    # ("2.9.0", "12.6", "3.12"),  # currently broken, cf. #453, #454
    # ("2.9.0", "12.6", "3.14"),  # currently broken, cf. #451
]
docker_registry = os.environ["DOCKER_REGISTRY"]
docker_image = os.environ["DOCKER_IMAGE"]
docker_tag = os.environ["IMAGE_TAG"]

# Generate matrix
matrix = []
for torch, cuda, python in torch_cuda_python_versions:
    image_name = f"{docker_registry}/{docker_image}-py{python}-pt{torch}-cu{cuda}"
    matrix.append({
        "VER_PYTHON": python,
        "VER_TORCH": torch,
        "VER_CUDA": cuda,
        "IMAGE_NAME": image_name,
        "IMAGE_TAG": docker_tag,
    })

# Write to GITHUB_OUTPUT
with open(os.environ["GITHUB_OUTPUT"], "a") as f:
    f.write(f"matrix={json.dumps(matrix)}\n")
