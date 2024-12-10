# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
#
# Code adapted from https://github.com/huggingface/transformers
# Copyright 2018- The Hugging Face team. All rights reserved. Licensed under the Apache License, Version 2.0
# License is provided for attribution purposes only, Not a Contribution

from itertools import chain


def tokenize_dataset(dataset, tokenizer, sequence_length):
    # Define Tokenization function and Grouping function
    def _tokenize_function_join_nn(examples):
        return tokenizer(["\n\n".join(examples["text"])])

    def _group_texts(examples):
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
