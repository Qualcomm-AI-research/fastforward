# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import math

import torch

from doc_helpers.export.benchmark.util import _sliced_tqdm_iterator, generate_attention_mask


def evaluate_perplexity_metrics(
    model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, limit: int | None = None
):
    losses = []
    token_count = 0

    with torch.no_grad():
        for batch in _sliced_tqdm_iterator(dataloader, limit):
            seq_length = batch["input_ids"].shape[1]
            attention_mask = generate_attention_mask(seq_length)
            batch["attention_mask"] = attention_mask.cuda()

            output = model(**{k: v.cuda() for k, v in batch.items()})
            losses.append(output.loss)
            token_count += batch["input_ids"].numel()

    losses = torch.stack(losses)
    try:
        eval_loss = torch.mean(losses).item()
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")

    return dict(loss=eval_loss, per_token_perplexity=perplexity, token_count=token_count)
