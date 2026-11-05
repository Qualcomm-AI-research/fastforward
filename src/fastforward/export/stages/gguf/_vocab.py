# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Write a HuggingFace tokenizer's vocabulary into a GGUF writer.

Kept separate from the stage module because it touches the tokenizer's on-disk
files (for BPE merges), which the pure extraction/packing helpers deliberately
avoid.

Two writers are provided: :func:`write_bpe_vocab` for GPT-2-style BPE
tokenizers (the default), and :func:`make_llama_hf_vocab_writer` for
byte-fallback BPE tokenizers converted from SentencePiece (e.g. Gemma). An
:class:`~fastforward.export.stages.gguf.adapter.ArchAdapter` selects one through
its ``write_vocab`` field; custom writers can reuse :func:`tokenizer_dir`.
"""

from __future__ import annotations

import contextlib
import pathlib
import shutil
import tempfile

from collections.abc import Collection, Iterator
from typing import TYPE_CHECKING, Any

from gguf import GGUFWriter, LlamaHfVocab, SpecialVocab, TokenType

from fastforward.exceptions import ExportError
from fastforward.export.stages.gguf._config import GgufSourceConfig

if TYPE_CHECKING:
    from fastforward.export.stages.gguf.adapter import ArchAdapter, WriteVocabT

# llama.cpp's converter gives every token of a byte-fallback BPE vocabulary this score.
_LLAMA_HF_TOKEN_SCORE = -1000.0


@contextlib.contextmanager
def tokenizer_dir(tokenizer: Any) -> Iterator[pathlib.Path]:
    """Yield a directory holding ``tokenizer``'s files, for writers that read from disk.

    The ``gguf`` vocab readers (``SpecialVocab``, ``LlamaHfVocab``, ...) load
    from disk. The tokenizer is always saved with ``save_pretrained`` to a
    temporary directory that lives until the context exits, so in-memory
    changes (e.g. added tokens) are reflected even when the tokenizer was
    loaded from a local directory. The source directory's ``config.json``, if
    any, is copied along because ``SpecialVocab`` falls back to its
    ``*_token_id`` entries for special tokens the tokenizer files don't define.

    Example::

        def write_my_vocab(writer, tokenizer, config, adapter):
            with tokenizer_dir(tokenizer) as model_dir:
                SpecialVocab(model_dir, load_merges=True).add_to_gguf(writer)

    Args:
        tokenizer: A tokenizer with ``save_pretrained``, e.g. a HuggingFace
            tokenizer.

    Yields:
        The directory holding the tokenizer's files.
    """
    with tempfile.TemporaryDirectory() as td:
        tokenizer.save_pretrained(td)
        model_dir = pathlib.Path(td)
        raw_path = getattr(tokenizer, "name_or_path", None)
        source_config = pathlib.Path(raw_path) / "config.json" if raw_path else None
        if source_config is not None and source_config.is_file():
            shutil.copyfile(source_config, model_dir / "config.json")
        yield model_dir


def _write_tokenizer_discriminators(writer: GGUFWriter, adapter: ArchAdapter) -> None:
    writer.add_tokenizer_model(adapter.tokenizer_model)
    if adapter.tokenizer_pre is not None:
        writer.add_tokenizer_pre(adapter.tokenizer_pre)


def write_bpe_vocab(
    writer: GGUFWriter,
    tokenizer: Any,
    config: GgufSourceConfig,
    adapter: ArchAdapter,
) -> None:
    """Write a BPE vocabulary and special tokens to ``writer``.

    This is the default ``ArchAdapter.write_vocab``. Token types come from the
    tokenizer's special and added tokens; ids missing from the tokenizer are
    padded up to ``config.vocab_size`` as ``UNUSED`` placeholders.

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

    _write_tokenizer_discriminators(writer, adapter)
    writer.add_token_list(tokens)
    writer.add_token_types(token_types)

    with tokenizer_dir(tokenizer) as model_dir:
        SpecialVocab(model_dir, load_merges=True).add_to_gguf(writer)


def make_llama_hf_vocab_writer(
    *,
    user_defined_tokens: Collection[str] = (),
    add_bos_token: bool | None = None,
    add_space_prefix: bool | None = None,
) -> WriteVocabT:
    """Build a vocab writer for byte-fallback BPE tokenizers (SentencePiece-derived).

    The writer reads the tokenizer through ``gguf.LlamaHfVocab``, as llama.cpp's
    converter does for Gemma-family models, so token scores and ``BYTE`` types
    for ``<0xNN>`` tokens match the converter's output. The tokenizer must be
    backed by a ``tokenizer.json`` whose BPE model has ``byte_fallback``
    enabled.

    Args:
        user_defined_tokens: Token texts written as ``USER_DEFINED`` regardless
            of their type in the tokenizer, so llama.cpp always renders them
            (e.g. chat-template markers its output parsers read).
        add_bos_token: Written as ``tokenizer.ggml.add_bos_token`` when not
            ``None``, overriding any value from ``tokenizer_config.json``.
        add_space_prefix: Written as ``tokenizer.ggml.add_space_prefix`` when
            not ``None``.

    Returns:
        A function usable as ``ArchAdapter.write_vocab``.
    """
    forced_user_defined = frozenset(user_defined_tokens)

    def write_llama_hf_vocab(
        writer: GGUFWriter,
        tokenizer: Any,
        config: GgufSourceConfig,
        adapter: ArchAdapter,
    ) -> None:
        with tokenizer_dir(tokenizer) as model_dir:
            vocab = LlamaHfVocab(model_dir)
            tokens: list[bytes] = []
            scores: list[float] = []
            token_types: list[int] = []
            try:
                for text, score, token_type in vocab.all_tokens():
                    tokens.append(text)
                    scores.append(score)
                    if text.decode("utf-8") in forced_user_defined:
                        token_type = TokenType.USER_DEFINED
                    token_types.append(int(token_type))
            except KeyError as e:
                # LlamaHfVocab indexes the base vocabulary by consecutive id.
                msg = f"The tokenizer's base vocabulary has no token with id {e.args[0]}"
                raise ExportError(msg) from e

            # A token's id is its position in the list, so every id the tokenizer
            # assigns must land at its own position (no gaps before added tokens).
            for text, token_id in tokenizer.get_vocab().items():
                if token_id >= len(tokens) or tokens[token_id] != text.encode("utf-8"):
                    msg = (
                        f"Token {text!r} has id {token_id} in the tokenizer but would be "
                        + "exported at a different id; the vocabulary has gaps"
                    )
                    raise ExportError(msg)

            if len(tokens) > config.vocab_size:
                msg = (
                    f"Tokenizer has {len(tokens)} tokens but the model's vocab_size is "
                    + f"{config.vocab_size}"
                )
                raise ExportError(msg)
            for token_id in range(len(tokens), config.vocab_size):
                tokens.append(f"[PAD{token_id}]".encode())
                scores.append(_LLAMA_HF_TOKEN_SCORE)
                token_types.append(int(TokenType.UNUSED))

            _write_tokenizer_discriminators(writer, adapter)
            writer.add_token_list(tokens)
            writer.add_token_scores(scores)
            writer.add_token_types(token_types)
            SpecialVocab(model_dir, load_merges=True).add_to_gguf(writer)

        if add_space_prefix is not None:
            writer.add_add_space_prefix(add_space_prefix)
        if add_bos_token is not None:
            writer.add_add_bos_token(add_bos_token)

    return write_llama_hf_vocab
