# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Tests for writing a HuggingFace tokenizer's vocabulary into a GGUF file.

``write_bpe_vocab`` builds the token list itself from the tokenizer's in-memory
vocabulary; ``make_llama_hf_vocab_writer`` reads it through ``gguf.LlamaHfVocab``
like llama.cpp's converter does. These tests drive a real ``GGUFWriter`` and
read the result back with ``GGUFReader`` so the assertions cover the bytes that
actually land in the file.
"""

import json
import pathlib

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import fastforward as ff
import gguf
import pytest

from fastforward.export.stages.gguf import (
    ArchAdapter,
    WriteVocabT,
    make_llama_hf_vocab_writer,
    tokenizer_dir,
)
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
    """A tokenizer exposing only the surface ``write_bpe_vocab`` consumes.

    Args:
        vocab: The base token-to-id mapping.
        added_vocab: Tokens registered as "added" (become ``USER_DEFINED``).
        special_ids: Token ids flagged special (become ``CONTROL``).
        name_or_path: The path the tokenizer claims to be loaded from. Writers
            ignore it and always materialize the tokenizer into a temporary
            directory.
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


def _make_adapter(
    tokenizer_model: str = "gpt2",
    tokenizer_pre: str | None = "default",
    write_vocab: WriteVocabT | None = None,
) -> ArchAdapter:
    """A minimal adapter; vocab writers only read the tokenizer discriminators."""
    adapter = ArchAdapter(
        gguf_arch="test",
        tensor_rules=(),
        write_metadata=lambda writer, config: None,
        tokenizer_model=tokenizer_model,
        tokenizer_pre=tokenizer_pre,
    )
    return adapter if write_vocab is None else replace(adapter, write_vocab=write_vocab)


def _write_and_read(
    tmp_path: pathlib.Path,
    tokenizer: Any,
    config: _Config,
    adapter: ArchAdapter,
) -> gguf.GGUFReader:
    """Run the adapter's vocab writer into a real GGUF file and reopen it for reading."""
    output_path = tmp_path / "vocab.gguf"
    writer = GGUFWriter(str(output_path), arch=adapter.gguf_arch)
    try:
        adapter.write_vocab(writer, tokenizer, config, adapter)
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


def _token_scores(reader: gguf.GGUFReader) -> list[float]:
    field = reader.get_field("tokenizer.ggml.scores")
    assert field is not None
    return [float(field.parts[index][0]) for index in field.data]


def _bool_field(reader: gguf.GGUFReader, key: str) -> bool:
    field = reader.get_field(key)
    assert field is not None, f"missing field {key}"
    return bool(field.parts[field.data[-1]][0])


def test_write_bpe_vocab_classifies_token_types(tmp_path: pathlib.Path) -> None:
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


def test_write_bpe_vocab_pads_up_to_config_vocab_size(tmp_path: pathlib.Path) -> None:
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


def test_write_bpe_vocab_skips_ids_beyond_config_vocab_size(tmp_path: pathlib.Path) -> None:
    # GIVEN: a tokenizer holding more tokens than the config's vocab_size.
    tokenizer = _StubTokenizer(vocab={"a": 0, "b": 1, "c": 2, "d": 3})

    # WHEN: writing the vocabulary.
    reader = _write_and_read(tmp_path, tokenizer, _Config(vocab_size=2), _make_adapter())

    # THEN: only ids below vocab_size are written; the surplus is dropped.
    assert _token_list(reader) == ["a", "b"]


def test_write_bpe_vocab_writes_adapter_tokenizer_discriminators(tmp_path: pathlib.Path) -> None:
    # GIVEN: an adapter carrying non-default tokenizer discriminators.
    adapter = _make_adapter(tokenizer_model="gpt2", tokenizer_pre="llama-bpe")
    tokenizer = _StubTokenizer(vocab={"a": 0})

    # WHEN: writing the vocabulary.
    reader = _write_and_read(tmp_path, tokenizer, _Config(vocab_size=1), adapter)

    # THEN: llama.cpp's tokenizer-model and pre-tokenizer selectors come from the
    # adapter, not from the tokenizer.
    assert _string_field(reader, "tokenizer.ggml.model") == "gpt2"
    assert _string_field(reader, "tokenizer.ggml.pre") == "llama-bpe"


def test_write_bpe_vocab_materializes_in_memory_tokenizer_for_merges(
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


def test_write_bpe_vocab_skips_pre_tokenizer_when_none(tmp_path: pathlib.Path) -> None:
    # GIVEN: an adapter without a pre-tokenizer discriminator.
    adapter = _make_adapter(tokenizer_pre=None)
    tokenizer = _StubTokenizer(vocab={"a": 0})

    # WHEN: writing the vocabulary.
    reader = _write_and_read(tmp_path, tokenizer, _Config(vocab_size=1), adapter)

    # THEN: the tokenizer model is written but no pre-tokenizer field.
    assert _string_field(reader, "tokenizer.ggml.model") == "gpt2"
    assert reader.get_field("tokenizer.ggml.pre") is None


def test_tokenizer_dir_saves_even_when_backed_by_existing_directory(tmp_path: pathlib.Path) -> None:
    # GIVEN: a tokenizer backed by an existing directory.
    tokenizer = _StubTokenizer(vocab={"a": 0}, name_or_path=str(tmp_path))

    # WHEN: entering tokenizer_dir.
    with tokenizer_dir(tokenizer) as model_dir:
        # THEN: the tokenizer is saved to a separate directory, so in-memory
        # changes are not lost.
        assert model_dir != tmp_path
    assert len(tokenizer.save_pretrained_calls) == 1


def test_tokenizer_dir_copies_config_json_from_source_directory(tmp_path: pathlib.Path) -> None:
    # GIVEN: a tokenizer backed by a directory holding a config.json.
    (tmp_path / "config.json").write_text('{"bos_token_id": 1}', encoding="utf-8")
    tokenizer = _StubTokenizer(vocab={"a": 0}, name_or_path=str(tmp_path))

    # WHEN: entering tokenizer_dir.
    with tokenizer_dir(tokenizer) as model_dir:
        # THEN: the config.json is available for SpecialVocab's fallback.
        assert (model_dir / "config.json").read_text(encoding="utf-8") == '{"bos_token_id": 1}'


def test_tokenizer_dir_saves_in_memory_tokenizer_to_temporary_directory() -> None:
    # GIVEN: a tokenizer not backed by a directory.
    tokenizer = _StubTokenizer(vocab={"a": 0}, name_or_path="")

    # WHEN: entering tokenizer_dir.
    with tokenizer_dir(tokenizer) as model_dir:
        # THEN: the tokenizer is saved into the yielded directory.
        assert (model_dir / "merges.txt").exists()

    # THEN: the directory is removed on exit.
    assert not model_dir.exists()


def test_adapter_accepts_custom_vocab_writer(tmp_path: pathlib.Path) -> None:
    # GIVEN: a user-defined vocab writer built on tokenizer_dir.
    def write_custom_vocab(
        writer: GGUFWriter, tokenizer: Any, config: Any, adapter: ArchAdapter
    ) -> None:
        writer.add_tokenizer_model(adapter.tokenizer_model)
        writer.add_token_list([f"tok{i}" for i in range(config.vocab_size)])
        with tokenizer_dir(tokenizer) as model_dir:
            gguf.SpecialVocab(model_dir, load_merges=True).add_to_gguf(writer)

    adapter = _make_adapter(
        tokenizer_model="custom", tokenizer_pre=None, write_vocab=write_custom_vocab
    )

    # WHEN: writing the vocabulary through the adapter.
    reader = _write_and_read(tmp_path, _StubTokenizer(vocab={}), _Config(vocab_size=2), adapter)

    # THEN: the custom writer's output lands in the file.
    assert _string_field(reader, "tokenizer.ggml.model") == "custom"
    assert _token_list(reader) == ["tok0", "tok1"]
    assert reader.get_field("tokenizer.ggml.merges") is not None


# Base vocabulary of the byte-fallback tokenizer below; two tokens are added on top.
_SPM_VOCAB = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "<0x41>": 4, "a": 5, "b": 6, "ab": 7}
_SPM_TOKENS = [*_SPM_VOCAB, "<|channel>", "<turn>"]


@pytest.fixture
def spm_tokenizer(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """A tiny SentencePiece-style tokenizer (byte-fallback BPE) loaded from disk.

    ``<|channel>`` is an added non-special token and ``<turn>`` an added special
    token, mirroring the kinds of added tokens in Gemma's vocabulary.
    """
    # tokenizers and transformers are not part of the CI `test` dependency group.
    tokenizers = pytest.importorskip("tokenizers")
    PreTrainedTokenizerFast = pytest.importorskip("transformers").PreTrainedTokenizerFast  # noqa: N806
    models, decoders = tokenizers.models, tokenizers.decoders
    bpe = models.BPE(vocab=_SPM_VOCAB, merges=[("a", "b")], unk_token="<unk>", byte_fallback=True)
    backend = tokenizers.Tokenizer(bpe)
    backend.decoder = decoders.Sequence([
        decoders.Replace("\u2581", " "),
        decoders.ByteFallback(),
        decoders.Fuse(),
    ])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="<eos>",
        bos_token="<bos>",
        unk_token="<unk>",
    )
    tokenizer.add_tokens(["<|channel>"])
    tokenizer.add_special_tokens({"additional_special_tokens": ["<turn>"]})
    model_dir = tmp_path_factory.mktemp("spm_tokenizer")
    tokenizer.save_pretrained(model_dir)
    return PreTrainedTokenizerFast.from_pretrained(model_dir)


def test_llama_hf_vocab_writer_matches_converter_conventions(
    tmp_path: pathlib.Path, spm_tokenizer: Any
) -> None:
    # GIVEN: an adapter using the llama-hf vocab writer and no pre-tokenizer.
    adapter = _make_adapter(
        tokenizer_model="gemma4", tokenizer_pre=None, write_vocab=make_llama_hf_vocab_writer()
    )

    # WHEN: writing the vocabulary.
    reader = _write_and_read(tmp_path, spm_tokenizer, _Config(vocab_size=10), adapter)

    # THEN: tokens land in id order, all scored -1000 as llama.cpp's converter does.
    assert _token_list(reader) == _SPM_TOKENS
    assert _token_scores(reader) == [-1000.0] * 10

    # THEN: specials -> CONTROL, <0xNN> -> BYTE, added non-special -> USER_DEFINED.
    control, byte = int(TokenType.CONTROL), int(TokenType.BYTE)
    normal, user_defined = int(TokenType.NORMAL), int(TokenType.USER_DEFINED)
    assert _token_types(reader) == [control] * 4 + [byte] + [normal] * 3 + [user_defined, control]

    # THEN: the adapter's tokenizer model is written, no pre-tokenizer, and merges
    # and special-token ids come from the tokenizer files.
    assert _string_field(reader, "tokenizer.ggml.model") == "gemma4"
    assert reader.get_field("tokenizer.ggml.pre") is None
    assert reader.get_field("tokenizer.ggml.merges") is not None
    bos_field = reader.get_field("tokenizer.ggml.bos_token_id")
    assert bos_field is not None
    assert int(bos_field.parts[bos_field.data[-1]][0]) == 2


def test_llama_hf_vocab_writer_forces_user_defined_tokens(
    tmp_path: pathlib.Path, spm_tokenizer: Any
) -> None:
    # GIVEN: a writer forcing the special "<turn>" token to USER_DEFINED.
    adapter = _make_adapter(
        tokenizer_pre=None, write_vocab=make_llama_hf_vocab_writer(user_defined_tokens={"<turn>"})
    )

    # WHEN: writing the vocabulary.
    reader = _write_and_read(tmp_path, spm_tokenizer, _Config(vocab_size=10), adapter)

    # THEN: only that token's type changes.
    types = _token_types(reader)
    assert types[_SPM_TOKENS.index("<turn>")] == int(TokenType.USER_DEFINED)
    assert types[_SPM_TOKENS.index("<bos>")] == int(TokenType.CONTROL)


@pytest.mark.parametrize(
    ("flags", "expected_space_prefix"),
    [({"add_bos_token": True, "add_space_prefix": False}, False), ({}, None)],
    ids=["requested", "unset"],
)
def test_llama_hf_vocab_writer_writes_only_requested_flags(
    tmp_path: pathlib.Path,
    spm_tokenizer: Any,
    flags: dict[str, Any],
    expected_space_prefix: bool | None,
) -> None:
    # GIVEN: a writer requesting some flags, or none.
    adapter = _make_adapter(tokenizer_pre=None, write_vocab=make_llama_hf_vocab_writer(**flags))

    # WHEN: writing the vocabulary.
    reader = _write_and_read(tmp_path, spm_tokenizer, _Config(vocab_size=10), adapter)

    # THEN: requested flags are written with their values; unset ones are omitted.
    if expected_space_prefix is None:
        assert reader.get_field("tokenizer.ggml.add_space_prefix") is None
    else:
        assert _bool_field(reader, "tokenizer.ggml.add_space_prefix") is expected_space_prefix
        assert _bool_field(reader, "tokenizer.ggml.add_bos_token") is True


def test_llama_hf_vocab_writer_pads_up_to_config_vocab_size(
    tmp_path: pathlib.Path, spm_tokenizer: Any
) -> None:
    # GIVEN: a config claiming two more tokens than the tokenizer has.
    adapter = _make_adapter(tokenizer_pre=None, write_vocab=make_llama_hf_vocab_writer())

    # WHEN: writing the vocabulary.
    reader = _write_and_read(tmp_path, spm_tokenizer, _Config(vocab_size=12), adapter)

    # THEN: the gap is filled with UNUSED placeholders.
    assert _token_list(reader)[10:] == ["[PAD10]", "[PAD11]"]
    assert _token_types(reader)[10:] == [int(TokenType.UNUSED)] * 2
    assert len(_token_scores(reader)) == 12


def test_llama_hf_vocab_writer_rejects_tokenizer_larger_than_vocab(
    tmp_path: pathlib.Path, spm_tokenizer: Any
) -> None:
    # GIVEN: a config with fewer tokens than the tokenizer.
    adapter = _make_adapter(tokenizer_pre=None, write_vocab=make_llama_hf_vocab_writer())

    # WHEN/THEN: writing the vocabulary raises instead of truncating it.
    with pytest.raises(ff.exceptions.ExportError, match="10 tokens"):
        _write_and_read(tmp_path, spm_tokenizer, _Config(vocab_size=8), adapter)


def test_llama_hf_vocab_writer_reflects_in_memory_added_tokens(
    tmp_path: pathlib.Path, spm_tokenizer: Any
) -> None:
    # GIVEN: a tokenizer loaded from disk that gains a token in memory.
    spm_tokenizer.add_tokens(["NEW_TOKEN"])
    adapter = _make_adapter(tokenizer_pre=None, write_vocab=make_llama_hf_vocab_writer())

    # WHEN: writing the vocabulary.
    reader = _write_and_read(tmp_path, spm_tokenizer, _Config(vocab_size=12), adapter)

    # THEN: the new token is exported, not a placeholder.
    tokens = _token_list(reader)
    assert "NEW_TOKEN" in tokens
    assert "[PAD10]" not in tokens


def test_llama_hf_vocab_writer_rejects_gaps_in_token_ids(
    tmp_path: pathlib.Path, spm_tokenizer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # GIVEN: a tokenizer whose vocab has an id beyond the exported token list.
    vocab = {**spm_tokenizer.get_vocab(), "GAP_TOKEN": 12}
    monkeypatch.setattr(spm_tokenizer, "get_vocab", lambda: vocab)
    adapter = _make_adapter(tokenizer_pre=None, write_vocab=make_llama_hf_vocab_writer())

    # WHEN/THEN: writing the vocabulary raises instead of shifting ids.
    with pytest.raises(ff.exceptions.ExportError, match="GAP_TOKEN"):
        _write_and_read(tmp_path, spm_tokenizer, _Config(vocab_size=16), adapter)


def _reload_with_edited_tokenizer_json(
    tokenizer: Any, directory: pathlib.Path, edit: Callable[[dict[str, Any]], None]
) -> Any:
    """Save ``tokenizer``, apply ``edit`` to its ``tokenizer.json``, and load it back."""
    tokenizer.save_pretrained(directory)
    path = directory / "tokenizer.json"
    content = json.loads(path.read_text(encoding="utf-8"))
    edit(content)
    path.write_text(json.dumps(content), encoding="utf-8")
    return type(tokenizer).from_pretrained(directory)


def test_llama_hf_vocab_writer_rejects_hole_in_base_vocabulary(
    tmp_path: pathlib.Path, spm_tokenizer: Any
) -> None:
    # GIVEN: a base vocabulary with no token at id 7 (the last token jumps to id 8).
    def move_last_base_token(content: dict[str, Any]) -> None:
        content["model"]["vocab"]["ab"] = 8

    tokenizer = _reload_with_edited_tokenizer_json(
        spm_tokenizer, tmp_path / "holey", move_last_base_token
    )
    adapter = _make_adapter(tokenizer_pre=None, write_vocab=make_llama_hf_vocab_writer())

    # WHEN/THEN: writing the vocabulary raises a clear error, not a bare KeyError.
    with pytest.raises(ff.exceptions.ExportError, match="no token with id 7"):
        _write_and_read(tmp_path, tokenizer, _Config(vocab_size=16), adapter)


def test_llama_hf_vocab_writer_keeps_ids_when_added_tokens_are_stored_out_of_order(
    tmp_path: pathlib.Path, spm_tokenizer: Any
) -> None:
    # GIVEN: a tokenizer.json listing its added tokens in reverse order.
    tokenizer = _reload_with_edited_tokenizer_json(
        spm_tokenizer, tmp_path / "reversed", lambda content: content["added_tokens"].reverse()
    )
    adapter = _make_adapter(tokenizer_pre=None, write_vocab=make_llama_hf_vocab_writer())

    # WHEN: writing the vocabulary.
    reader = _write_and_read(tmp_path, tokenizer, _Config(vocab_size=10), adapter)

    # THEN: every token is exported at the id the tokenizer assigns it.
    exported = _token_list(reader)
    assert {text: exported.index(text) for text in tokenizer.get_vocab()} == tokenizer.get_vocab()
