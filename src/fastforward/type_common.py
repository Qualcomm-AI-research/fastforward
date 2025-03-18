# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import TypeAlias, TypeVar

import torch

T = TypeVar("T")

ScalarOrTuple: TypeAlias = T | tuple[T, ...]
ScalarOr2Tuple: TypeAlias = T | tuple[T, T]


SizeT = torch.Size | tuple[int, ...]

IntOrTuple: TypeAlias = ScalarOrTuple[int]
IntOr2Tuple: TypeAlias = ScalarOr2Tuple[int]

Tuple3: TypeAlias = tuple[T, T, T]
