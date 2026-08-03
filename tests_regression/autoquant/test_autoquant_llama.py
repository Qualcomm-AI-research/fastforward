# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import logging
import pathlib

from typing import Any, Iterator

import fastforward as ff
import pytest
import torch

from datasets import load_dataset  # type: ignore[import-untyped]
from fastforward.testing.data import tokenize_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    PreTrainedTokenizerBase,
    default_data_collator,
)

from tests_regression.autoquant._utils import evaluate_perplexity

_logger = logging.getLogger(__name__)


_SMALLEST_LLAMA_MODELS = [
    "meta-llama/Llama-3.2-1B",
]


# === Fixtures ===
_SEQ_LEN = 1024
_BATCH_SIZE = 1
_NB_BATCHES = 8
_DEVICE = "cuda"


@pytest.fixture
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the autoquant regression tests.")
    return torch.device(_DEVICE)


@pytest.fixture
def tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_name)


@pytest.fixture
def wikitext_valid_loader(tokenizer: PreTrainedTokenizerBase) -> DataLoader[Any]:
    raw_validset = load_dataset("wikitext", "wikitext-2-v1", split="validation")
    tokenized_validset = tokenize_dataset(raw_validset, tokenizer, _SEQ_LEN)
    return DataLoader(tokenized_validset, batch_size=_BATCH_SIZE, collate_fn=default_data_collator)


@pytest.fixture(scope="module", params=_SMALLEST_LLAMA_MODELS)
def model_name(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture
def model(model_name: str, device: torch.device) -> Iterator[LlamaForCausalLM]:
    m = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    m.to(device=device)  # type: ignore [call-arg]
    m.eval()
    yield m
    del m
    torch.cuda.empty_cache()


# === Tests ===


def test_autoquant_llama_matches_fp_perplexity(
    model: LlamaForCausalLM,
    model_name: str,
    wikitext_valid_loader: DataLoader[Any],
    device: torch.device,
    tmp_path: pathlib.Path,
) -> None:
    # GIVEN the wikitext-validation perplexity of a full-precision Llama model
    fp_perplexity = evaluate_perplexity(model, wikitext_valid_loader, device, limit=_NB_BATCHES)

    # WHEN the model is converted to a quantization-ready model via autoquant, with no
    # quantizer stubs materialized (they remain pass-through under strict_quantization(False))
    ff.autoquantize(
        model,
        output_path=tmp_path / "_autoquantized_llama.py",
        force_overwrite=True,
        auto_import=True,
    )
    ff.quantize_model(model, skip_quantized_modules=True)
    model.to(device=device)  # type: ignore [call-arg]

    with ff.strict_quantization(False):
        q_perplexity = evaluate_perplexity(model, wikitext_valid_loader, device, limit=_NB_BATCHES)

    report = (
        f"[{model_name}] FP perplexity: {fp_perplexity:.6f} | "
        f"Quant-ready perplexity: {q_perplexity:.6f}"
    )
    _logger.info(report)
    print(report)

    # THEN the quantization-ready model's perplexity should closely match the FP model's.
    assert q_perplexity == pytest.approx(fp_perplexity, rel=0.05)
