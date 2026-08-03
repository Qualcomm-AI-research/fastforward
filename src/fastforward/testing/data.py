# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#
# Code adapted from https://github.com/huggingface/transformers/blob/v5.9.0/examples/pytorch/language-modeling/run_clm.py
# Copyright 2018- The Hugging Face team. All rights reserved. Licensed under the Apache License, Version 2.0
# License is provided for attribution purposes only, Not a Contribution

from __future__ import annotations

from itertools import chain, islice
from typing import TYPE_CHECKING, Any, Sized

from tqdm import tqdm

if TYPE_CHECKING:
    import datasets  # type: ignore[import-untyped]
    import tokenizers  # type: ignore[import-untyped]


def tokenize_dataset(
    dataset: datasets.Dataset, tokenizer: tokenizers.Tokenizer, sequence_length: int
) -> datasets.Dataset:
    """Tokenize a text dataset and group tokens into fixed-length chunks.

    Text entries are joined with double newlines before tokenization, then the
    resulting token stream is concatenated and split into contiguous chunks of
    `sequence_length`. Any remainder that does not fill a full chunk is
    dropped. A `label` column is added as a copy of `input_ids`.

    Args:
        dataset: A HuggingFace `datasets.Dataset` containing a `text` column.
        tokenizer: A HuggingFace tokenizer used to encode the text.
        sequence_length: The length of each tokenized chunk.

    Returns:
        The tokenized and chunked dataset.
    """

    # Define Tokenization function and Grouping function
    def _tokenize_function_join_nn(examples: dict[str, list[str]]) -> Any:
        return tokenizer(["\n\n".join(examples["text"])])

    def _group_texts(examples: dict[str, list[str]]) -> dict[str, Any]:
        """Concatenate all texts from our dataset and generate chunks of max_seq_length."""
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # Drop the small remainder, we could add padding if the model supported it instead of this drop.
        # You can customize this part to your needs.
        if total_length >= sequence_length:
            total_length = (total_length // sequence_length) * sequence_length
        else:
            total_length = 0

        # Split by chunks of max_len.
        result = {
            k: [t[i : i + sequence_length] for i in range(0, total_length, sequence_length)]
            for k, t in concatenated_examples.items()
        }
        result["label"] = result["input_ids"].copy()
        return result

    tokenized_datasets = dataset.map(
        _tokenize_function_join_nn,
        batched=True,
        batch_size=None,
        writer_batch_size=None,
        num_proc=1,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on datasets",
    )
    tokenized_datasets = tokenized_datasets.map(
        _group_texts,
        batched=True,
        batch_size=None,
        num_proc=1,
        desc=f"Grouping texts in chunks of {sequence_length}",
    )
    return tokenized_datasets


def sliced_tqdm(iterator: Sized, limit: int | None = None, **kwargs: Any) -> tqdm[Any]:
    """Wrap an iterator in a tqdm progress bar, optionally limited to `limit` items.

    When `limit` is provided, iteration stops after `limit` items and the
    progress bar total reflects that limit. When `limit` is `None`, the full
    length of `iterator` is used as the total.

    Args:
        iterator: A sized iterable to iterate over.
        limit: The maximum number of items to yield. If `None`, iterate over all items.
        **kwargs: Additional keyword arguments forwarded to `tqdm`.

    Returns:
        A `tqdm` progress bar wrapping the (optionally sliced) iterator.
    """
    numel = limit if limit else len(iterator)
    return tqdm(islice(iterator, limit), total=numel, **kwargs)  # type:ignore[call-overload]
