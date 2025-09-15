# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from itertools import islice
from typing import Iterable, Sized

import torch

from tqdm.notebook import tqdm


def _sliced_tqdm_iterator(iterator: Sized, limit: int | None = None, **kwargs) -> Iterable:
    numel = limit if limit else len(iterator)
    return tqdm(islice(iterator, limit), total=numel, **kwargs)  # type:ignore[call-overload]


def _str_to_torch_dtype(dtype: str) -> torch.dtype:
    dtype_obj = getattr(torch, dtype, None)
    if not isinstance(dtype_obj, torch.dtype):
        msg = f"Unknown dtype {dtype}"
        raise RuntimeError(msg)
    return dtype_obj


def generate_attention_mask(seq_length: int, lowest_value: int = -100):
    attention_mask = torch.zeros(seq_length, seq_length) + lowest_value
    attention_mask = torch.triu(attention_mask, diagonal=1)
    attention_mask = attention_mask.reshape(1, 1, seq_length, seq_length)

    return attention_mask
