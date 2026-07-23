# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Write a HuggingFace tokenizer's vocabulary into a GGUF writer.

Kept separate from the stage module because it touches the tokenizer's on-disk
files (for BPE merges), which the pure extraction/packing helpers deliberately
avoid.
"""

import contextlib
import pathlib
import tempfile

from typing import Any

from gguf import GGUFWriter, SpecialVocab, TokenType

from fastforward.export.stages.gguf._config import GgufSourceConfig
from fastforward.export.stages.gguf.adapter import ArchAdapter


def write_vocab(
    writer: GGUFWriter,
    tokenizer: Any,
    config: GgufSourceConfig,
    adapter: ArchAdapter,
) -> None:
    """Write a BPE vocabulary and special tokens to ``writer``.

    Args:
        writer: An open ``gguf.GGUFWriter``.
        tokenizer: A HuggingFace fast tokenizer.
        config: The source model config (used for ``vocab_size``).
        adapter: Architecture adapter carrying the llama.cpp tokenizer-model
            and pre-tokenizer discriminators for the target architecture.
    """
    vocab_size = config.vocab_size

    vocab = tokenizer.get_vocab()
    added_vocab = tokenizer.get_added_vocab()

    special_token_ids: set[int] = set()
    for token_id, token_obj in getattr(tokenizer, "added_tokens_decoder", {}).items():
        if getattr(token_obj, "special", False):
            special_token_ids.add(token_id)

    id_to_token = {token_id: token for token, token_id in vocab.items()}
    tokens: list[str] = []
    token_types: list[int] = []
    for token_id in range(vocab_size):
        if token_id in id_to_token:
            token = id_to_token[token_id]
            if token_id in special_token_ids:
                token_types.append(int(TokenType.CONTROL))
            elif token in added_vocab:
                token_types.append(int(TokenType.USER_DEFINED))
            else:
                token_types.append(int(TokenType.NORMAL))
            tokens.append(token)
        else:
            tokens.append(f"[PAD{token_id}]")
            token_types.append(int(TokenType.UNUSED))

    writer.add_tokenizer_model(adapter.tokenizer_model)
    writer.add_token_list(tokens)
    writer.add_token_types(token_types)
    writer.add_tokenizer_pre(adapter.tokenizer_pre)

    # ``SpecialVocab`` reads merges from disk. If the tokenizer isn't backed by a
    # real directory (in-memory tokenizer), materialize it in a temp dir that is
    # cleaned up only after ``add_to_gguf`` has finished reading from it.
    raw_path = tokenizer.name_or_path
    model_dir = pathlib.Path(raw_path) if raw_path else None
    with contextlib.ExitStack() as stack:
        if model_dir is None or not model_dir.exists():
            td = stack.enter_context(tempfile.TemporaryDirectory())
            model_dir = pathlib.Path(td)
            tokenizer.save_pretrained(model_dir)

        special_vocab = SpecialVocab(model_dir, load_merges=True)
        special_vocab.add_to_gguf(writer)
