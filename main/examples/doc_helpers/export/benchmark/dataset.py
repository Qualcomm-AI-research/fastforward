# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from itertools import chain

import datasets
import torch.utils.data

from transformers import GPT2Tokenizer


def preprocess_dataset(
    dataset: datasets.Dataset,
    tokenizer: GPT2Tokenizer,
    sequence_length: int,
    num_workers: int = 4,
    batch_size: int = 1000,
    sequence_collate_function: str = "line_by_line",
) -> torch.utils.data.Dataset:
    # In distributed training, the load_dataset function guarantee that only one local process can
    # concurrently download the dataset.
    block_size = sequence_length

    # only keep the 'text' column
    dataset = dataset.remove_columns([c for c in dataset.column_names if c != "text"])

    # Preprocessing the datasets.
    # Check sequence length
    if block_size > tokenizer.model_max_length:
        print(
            f"The block_size passed ({block_size}) is larger than the maximum "
            f"length for the model ({tokenizer.model_max_length}). Using "
            f"block_size={tokenizer.model_max_length}."
        )
    block_size = min(block_size, tokenizer.model_max_length)

    # Tokenize all the texts.
    column_names = dataset.column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # Define tokenization function
    def tokenize_function_line_by_line(examples):
        return tokenizer(examples[text_column_name])

    def tokenize_function_join(examples):
        return tokenizer(["".join(examples[text_column_name])])

    def tokenize_function_join_nn(examples):
        return tokenizer(["\n\n".join(examples[text_column_name])])

    tokenize_functions = dict(
        line_by_line=tokenize_function_line_by_line,
        join=tokenize_function_join,
        join_nn=tokenize_function_join_nn,
    )

    tokenize_function = tokenize_functions[sequence_collate_function]
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=batch_size,
        writer_batch_size=batch_size,
        num_proc=num_workers,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    # Main data processing function that will concatenate all texts from our dataset and generate
    # chunks of max_seq_length.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of
        # this drop, you can customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        else:
            total_length = 0
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts
    # throws away a remainder for each of those groups of 1,000 texts. You can adjust that
    # batch_size here but a higher value might be slower to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method
    # for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    tokenized_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        batch_size=batch_size,
        num_proc=num_workers,
        # load_from_cache_file=not config.data.overwrite_cache,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    return tokenized_dataset
