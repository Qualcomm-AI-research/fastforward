# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Utility functions for autoquantregression tests."""

from itertools import islice
from typing import Any

import torch

from torch.utils.data import DataLoader


def _prepare_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "labels": batch["labels"].to(torch.long).to(device),
    }


@torch.no_grad()
def evaluate_perplexity(
    model: torch.nn.Module,
    valid_loader: DataLoader[Any],
    device: torch.device,
    limit: int | None = None,
) -> float:
    """Compute perplexity of `model` over the first `limit` batches of `valid_loader`."""
    model.eval()
    losses: list[torch.Tensor] = []
    for batch in islice(valid_loader, limit):
        prepared = _prepare_batch(batch, device)
        outputs = model(**prepared)
        losses.append(outputs.loss)
    eval_loss = torch.stack(losses).mean()
    return float(torch.exp(eval_loss))
