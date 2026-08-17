# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Tests for writing a HuggingFace tokenizer's vocabulary into a GGUF file.

``write_vocab`` exists because the ``gguf`` package's own ``BpeVocab`` and
``LlamaHfVocab`` both require an on-disk ``tokenizer.json`` and expose no
``add_to_gguf``; only ``SpecialVocab`` (which handles merges) is reusable. These
tests drive a real ``GGUFWriter`` and read the result back with ``GGUFReader``
so the assertions cover the bytes that actually land in the file.
"""

import json
import pathlib

from typing import Any

import gguf

from fastforward.export.stages.gguf._vocab import write_vocab
from fastforward.export.stages.gguf.adapter import ArchAdapter
from gguf import GGUFWriter, TokenType


class _Config:
    """A config satisfying :class:`GgufSourceConfig`; only ``vocab_size`` is read here."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = 0
        self.num_attention_heads = 1
        self.num_hidden_layers = 0
        self.intermediate_size = 0
        self.max_position_embeddings = 0
        self.rms_norm_eps = 1e-5


class _SpecialToken:
    """Stand-in for a HuggingFace ``AddedToken`` with a ``special`` flag."""

    def __init__(self, special: bool) -> None:
        self.special = special


class _StubTokenizer:
    """A tokenizer exposing only the surface ``write_vocab`` consumes.

    Args:
        vocab: The base token-to-id mapping.
        added_vocab: Tokens registered as "added" (become ``USER_DEFINED``).
        special_ids: Token ids flagged special (become ``CONTROL``).
        name_or_path: When it points at an existing directory ``write_vocab``
            reads merges from it directly; when empty the tokenizer is
            materialized into a temporary directory instead.
    """

    def __init__(
        self,
        vocab: dict[str, int],
        added_vocab: dict[str, int] | None = None,
        special_ids: set[int] | None = None,
        name_or_path: str = "",
    ) -> None:
        self._vocab = vocab
        self._added_vocab = added_vocab or {}
        self.name_or_path = name_or_path
        self.added_tokens_decoder = {
            token_id: _SpecialToken(True) for token_id in (special_ids or set())
        }
        self.save_pretrained_calls: list[pathlib.Path] = []

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def get_added_vocab(self) -> dict[str, int]:
        return dict(self._added_vocab)

    def save_pretrained(self, directory: Any) -> None:
        """Write the minimum on disk that ``SpecialVocab`` needs to load."""
        path = pathlib.Path(directory)
        self.save_pretrained_calls.append(path)
        (path / "merges.txt").write_text("#version: 0.2\na b\n", encoding="utf-8")
        (path / "special_tokens_map.json").write_text(
            json.dumps({"bos_token": "<s>"}), encoding="utf-8"
        )


def _make_adapter(tokenizer_model: str = "gpt2", tokenizer_pre: str = "default") -> ArchAdapter:
    """A minimal adapter; ``write_vocab`` only reads the tokenizer discriminators."""
    return ArchAdapter(
        gguf_arch="test",
        name_map=lambda hf_name: hf_name,
        transforms=[],
        write_metadata=lambda writer, config: None,
        tokenizer_model=tokenizer_model,
        tokenizer_pre=tokenizer_pre,
    )


def _write_and_read(
    tmp_path: pathlib.Path,
    tokenizer: _StubTokenizer,
    config: _Config,
    adapter: ArchAdapter,
) -> gguf.GGUFReader:
    """Run ``write_vocab`` into a real GGUF file and reopen it for reading."""
    output_path = tmp_path / "vocab.gguf"
    writer = GGUFWriter(str(output_path), arch=adapter.gguf_arch)
    try:
        write_vocab(writer, tokenizer, config, adapter)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()
    return gguf.GGUFReader(str(output_path))


def _string_field(reader: gguf.GGUFReader, key: str) -> str:
    field = reader.get_field(key)
    assert field is not None, f"missing field {key}"
    return bytes(field.parts[field.data[-1]]).decode("utf-8")


def _token_list(reader: gguf.GGUFReader) -> list[str]:
    field = reader.get_field("tokenizer.ggml.tokens")
    assert field is not None
    return [bytes(field.parts[index]).decode("utf-8") for index in field.data]


def _token_types(reader: gguf.GGUFReader) -> list[int]:
    field = reader.get_field("tokenizer.ggml.token_type")
    assert field is not None
    return [int(field.parts[index][0]) for index in field.data]


def test_write_vocab_classifies_token_types(tmp_path: pathlib.Path) -> None:
    # GIVEN: a tokenizer with a normal, a special, and an added token.
    tokenizer = _StubTokenizer(
        vocab={"a": 0, "b": 1, "<s>": 2, "custom": 3},
        added_vocab={"custom": 3, "<s>": 2},
        special_ids={2},
    )

    # WHEN: writing its vocabulary, with vocab_size matching the token count.
    reader = _write_and_read(tmp_path, tokenizer, _Config(vocab_size=4), _make_adapter())

    # THEN: tokens land in id order.
    assert _token_list(reader) == ["a", "b", "<s>", "custom"]

    # THEN: special -> CONTROL, added -> USER_DEFINED, remainder -> NORMAL. The
    # special check precedes the added check, so "<s>" (both) is CONTROL.
    assert _token_types(reader) == [
        int(TokenType.NORMAL),
        int(TokenType.NORMAL),
        int(TokenType.CONTROL),
        int(TokenType.USER_DEFINED),
    ]


def test_write_vocab_pads_up_to_config_vocab_size(tmp_path: pathlib.Path) -> None:
    # GIVEN: a config claiming a larger vocab than the tokenizer actually has.
    tokenizer = _StubTokenizer(vocab={"a": 0, "b": 1})

    # WHEN: writing the vocabulary.
    reader = _write_and_read(tmp_path, tokenizer, _Config(vocab_size=5), _make_adapter())

    # THEN: the gap is filled with placeholders so the token list matches the
    # embedding-matrix row count llama.cpp expects.
    assert _token_list(reader) == ["a", "b", "[PAD2]", "[PAD3]", "[PAD4]"]
    assert _token_types(reader) == [
        int(TokenType.NORMAL),
        int(TokenType.NORMAL),
        int(TokenType.UNUSED),
        int(TokenType.UNUSED),
        int(TokenType.UNUSED),
    ]


def test_write_vocab_skips_ids_beyond_config_vocab_size(tmp_path: pathlib.Path) -> None:
    # GIVEN: a tokenizer holding more tokens than the config's vocab_size.
    tokenizer = _StubTokenizer(vocab={"a": 0, "b": 1, "c": 2, "d": 3})

    # WHEN: writing the vocabulary.
    reader = _write_and_read(tmp_path, tokenizer, _Config(vocab_size=2), _make_adapter())

    # THEN: only ids below vocab_size are written; the surplus is dropped.
    assert _token_list(reader) == ["a", "b"]


def test_write_vocab_writes_adapter_tokenizer_discriminators(tmp_path: pathlib.Path) -> None:
    # GIVEN: an adapter carrying non-default tokenizer discriminators.
    adapter = _make_adapter(tokenizer_model="gpt2", tokenizer_pre="llama-bpe")
    tokenizer = _StubTokenizer(vocab={"a": 0})

    # WHEN: writing the vocabulary.
    reader = _write_and_read(tmp_path, tokenizer, _Config(vocab_size=1), adapter)

    # THEN: llama.cpp's tokenizer-model and pre-tokenizer selectors come from the
    # adapter, not from the tokenizer.
    assert _string_field(reader, "tokenizer.ggml.model") == "gpt2"
    assert _string_field(reader, "tokenizer.ggml.pre") == "llama-bpe"


def test_write_vocab_materializes_in_memory_tokenizer_for_merges(
    tmp_path: pathlib.Path,
) -> None:
    # GIVEN: a tokenizer not backed by a real directory.
    tokenizer = _StubTokenizer(vocab={"a": 0, "b": 1}, name_or_path="")

    # WHEN: writing the vocabulary.
    reader = _write_and_read(tmp_path, tokenizer, _Config(vocab_size=2), _make_adapter())

    # THEN: it was saved to a temporary directory so SpecialVocab could read
    # merges from disk, and that directory outlived the read.
    assert len(tokenizer.save_pretrained_calls) == 1
    assert reader.get_field("tokenizer.ggml.merges") is not None

    # THEN: the temporary directory was cleaned up afterwards.
    assert not tokenizer.save_pretrained_calls[0].exists()


def test_write_vocab_reads_merges_from_existing_tokenizer_directory(
    tmp_path: pathlib.Path,
) -> None:
    # GIVEN: a tokenizer already backed by a directory containing merges.
    model_dir = tmp_path / "tokenizer"
    model_dir.mkdir()
    (model_dir / "merges.txt").write_text("#version: 0.2\na b\n", encoding="utf-8")
    tokenizer = _StubTokenizer(vocab={"a": 0, "b": 1}, name_or_path=str(model_dir))

    # WHEN: writing the vocabulary.
    reader = _write_and_read(tmp_path, tokenizer, _Config(vocab_size=2), _make_adapter())

    # THEN: the existing directory is used as-is, without re-saving the tokenizer.
    assert tokenizer.save_pretrained_calls == []
    assert reader.get_field("tokenizer.ggml.merges") is not None
