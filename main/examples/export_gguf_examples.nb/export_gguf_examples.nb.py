# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# # Exporting Quantized Models to GGUF
#
# > ⚠️ **WARNING**: GGUF export is experimental and its API may change.
#
# This tutorial exports **Qwen3-0.6B** with weight-only, symmetric 4-bit quantization to GGUF. It
# compares min/max calibration with GPTQ using FastForward's built-in Qwen3 architecture adapter.
#
# Both approaches load WikiText-2, measure floating-point perplexity, select transformer-block
# projections with MPath, calibrate Q4_0-compatible weight quantizers, measure quantized perplexity,
# export the model and tokenizer, and inspect the GGUF files.
#
# The notebook requires CUDA, Hugging Face model/dataset access (or populated caches), the `gguf`
# package, and llama.cpp's `llama-perplexity` executable. Install the pinned CPU build before
# running the notebook locally:
#
# ```bash
# export PATH="$(./scripts/install-llama-cpp.sh):$PATH"
# ```
#
# Q4_0 stores 32 values per block, so every quantized weight row must contain a multiple of 32
# values.

# %% [markdown]
# ## Setup

# %%
import functools
import gc
import os
import re
import shutil
import subprocess

from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

# Avoid runtime Triton compilation on environments without a system C compiler.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import fastforward as ff
import gguf
import torch

from datasets import load_dataset
from fastforward.export.pipeline import (
    ExportRequest,
    GgufLlamaCppOptions,
    export_with_pipeline,
)
from fastforward.export.stages.gguf import GGUF_Q4_0, QWEN3_ADAPTER, ArchAdapter
from fastforward.testing.data import sliced_tqdm, tokenize_dataset
from IPython.display import Markdown, display
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator

if not torch.cuda.is_available():
    msg = "This tutorial requires a CUDA-enabled PyTorch installation and GPU."
    raise RuntimeError(msg)

device = torch.device("cuda")
model_dtype = torch.float16
sequence_length = 1024
batch_size = 1
evaluation_steps = 8
calibration_steps = 1
gptq_calibration_steps = 8
llama_cpp_threads = 8
llama_cpp_timeout_seconds = 20 * 60
perplexity_relative_tolerance = 0.05

output_dir = Path("gguf_exports")
output_dir.mkdir(parents=True, exist_ok=True)

q4_0_granularity = ff.granularity.PerBlock(
    block_dims=1,
    block_sizes=GGUF_Q4_0.block_size,
    per_channel_dims=0,
)

# %% [markdown]
# Tokenize WikiText with Qwen's tokenizer for calibration and perplexity evaluation.

# %%
raw_calibration_set = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train")
raw_validation_set = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="validation")


def make_dataloaders(tokenizer: Any) -> tuple[DataLoader, DataLoader]:
    """Tokenize WikiText and create calibration and validation loaders."""
    calibration_set = tokenize_dataset(raw_calibration_set, tokenizer, sequence_length)
    validation_set = tokenize_dataset(raw_validation_set, tokenizer, sequence_length)
    calibration_loader = DataLoader(
        calibration_set,
        batch_size=batch_size,
        collate_fn=default_data_collator,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=batch_size,
        collate_fn=default_data_collator,
    )
    return calibration_loader, validation_loader


def prepare_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Move a tokenized language-model batch to CUDA."""
    return {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "labels": batch["labels"].to(device=device, dtype=torch.long),
    }


def prepare_perplexity_batch(
    batch: dict[str, torch.Tensor], tokenizer: Any
) -> dict[str, torch.Tensor]:
    """Prepare labels matching llama.cpp's half-window perplexity protocol."""
    input_ids = batch["input_ids"].to(device).clone()
    if getattr(tokenizer, "add_bos_token", False):
        if tokenizer.bos_token_id is None:
            msg = "Tokenizer enables BOS insertion but has no BOS token ID"
            raise RuntimeError(msg)
        input_ids[:, 0] = tokenizer.bos_token_id

    labels = input_ids.clone()
    first_scored_logit = input_ids.shape[1] // 2
    labels[:, : first_scored_logit + 1] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": batch["attention_mask"].to(device),
        "labels": labels,
    }


@torch.no_grad()
def evaluate_perplexity(
    model: torch.nn.Module,
    data_loader: DataLoader,
    tokenizer: Any,
) -> float:
    """Evaluate perplexity using llama.cpp's half-window scoring protocol."""
    model.eval()
    negative_log_likelihood = torch.zeros((), dtype=torch.float64, device=device)
    token_count = 0
    for batch in sliced_tqdm(data_loader, evaluation_steps):
        prepared_batch = prepare_perplexity_batch(batch, tokenizer)
        outputs = model(**prepared_batch)
        scored_tokens = int((prepared_batch["labels"] != -100).sum())
        negative_log_likelihood += outputs.loss.double() * scored_tokens
        token_count += scored_tokens
    return float(torch.exp(negative_log_likelihood / token_count))


def release_cuda_memory() -> None:
    """Release Python and CUDA memory between the two model runs."""
    gc.collect()
    torch.cuda.empty_cache()


_LLAMA_PPL_PATTERN = re.compile(
    r"Final estimate: PPL = (?P<ppl>[0-9.eE+-]+) \+/- (?P<uncertainty>[0-9.eE+-]+)"
)


def run_llama_perplexity(model_path: Path, corpus_path: Path) -> tuple[float, float]:
    """Run llama.cpp perplexity on one GGUF model and parse its final estimate."""
    executable = shutil.which("llama-perplexity")
    if executable is None:
        msg = (
            "llama-perplexity was not found on PATH. Run "
            '`export PATH="$(./scripts/install-llama-cpp.sh):$PATH"` before this tutorial.'
        )
        raise RuntimeError(msg)

    command = [
        executable,
        "--model",
        str(model_path),
        "--file",
        str(corpus_path),
        "--ctx-size",
        str(sequence_length),
        "--chunks",
        str(evaluation_steps),
        "--batch-size",
        str(sequence_length),
        "--threads",
        str(llama_cpp_threads),
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=llama_cpp_timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        msg = f"llama-perplexity timed out after {llama_cpp_timeout_seconds} seconds"
        raise RuntimeError(msg) from error

    if result.returncode != 0:
        msg = f"llama-perplexity failed with exit code {result.returncode}:\n{result.stdout}"
        raise RuntimeError(msg)

    match = _LLAMA_PPL_PATTERN.search(result.stdout)
    if match is None:
        msg = f"Could not parse llama-perplexity output:\n{result.stdout}"
        raise RuntimeError(msg)
    return float(match.group("ppl")), float(match.group("uncertainty"))


def validate_with_llama_cpp(
    minmax_path: Path,
    gptq_path: Path,
    validation_text: Sequence[str],
) -> tuple[float, float, float, float]:
    """Compare both exports with llama.cpp on the shared validation corpus."""
    validation_corpus_path = output_dir / "wikitext-2-validation.txt"
    validation_corpus_path.write_text("\n\n".join(validation_text), encoding="utf8")

    executable = shutil.which("llama-perplexity")
    if executable is None:
        msg = "llama-perplexity was not found; install it before running this tutorial."
        raise RuntimeError(msg)

    llama_version = subprocess.run(
        [executable, "--version"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
    ).stdout.strip()
    print(llama_version)

    minmax_ppl, minmax_uncertainty = run_llama_perplexity(minmax_path, validation_corpus_path)
    gptq_ppl, gptq_uncertainty = run_llama_perplexity(gptq_path, validation_corpus_path)
    return minmax_ppl, minmax_uncertainty, gptq_ppl, gptq_uncertainty


def calibrate_weights(model: torch.nn.Module, data_loader: DataLoader) -> None:
    """Estimate weight ranges using real WikiText inputs.

    One batch is sufficient for min/max weight calibration because every forward pass uses the
    same weights. The Qwen example below compares this baseline with activation-aware GPTQ.
    """
    model.eval()
    with (
        torch.no_grad(),
        ff.strict_quantization(False),
        ff.estimate_ranges(model, ff.range_setting.running_minmax),
    ):
        for batch in sliced_tqdm(data_loader, calibration_steps):
            model(**prepare_batch(batch))


def optimize_with_gptq(
    model: torch.nn.Module,
    calibration_inputs: Sequence[torch.Tensor],
) -> None:
    """Apply the tutorial's Q4_0-compatible GPTQ configuration."""
    targets = ff.mpath.query("**/layers/**/[cls:ff.nn.QuantizedLinear]")
    gptq_fn = functools.partial(ff.algorithms.gptq, perc_damp=0.05)
    with torch.no_grad(), ff.strict_quantization(False):
        ff.layerwise_optimize(
            model,
            calibration_inputs,
            gptq_fn,
            targets=targets,
            sample_args=(calibration_inputs[0],),
        )


def initialize_q4_0_weights(model: torch.nn.Module, pattern: str) -> Any:
    """Initialize selected weight quantizers for GGUF Q4_0 export."""
    weight_quantizers = ff.find_quantizers(model, pattern)
    if not weight_quantizers:
        msg = f"No weight quantizers matched {pattern!r}"
        raise RuntimeError(msg)
    weight_quantizers.initialize(
        ff.nn.LinearQuantizer,
        num_bits=GGUF_Q4_0.num_bits,
        symmetric=True,
        granularity=q4_0_granularity,
        device=next(model.parameters()).device,
    )
    names = [result.full_name for result in weight_quantizers]
    print(f"Initialized {len(names)} Q4_0 weight quantizers")
    print("First matches:", names[:5])
    return weight_quantizers


def export_gguf(
    model: torch.nn.Module,
    tokenizer: Any,
    adapter: ArchAdapter,
    model_name: str,
) -> Path:
    """Export a calibrated model through the registered GGUF pipeline."""
    options = GgufLlamaCppOptions(
        arch_adapter=adapter,
        quant_format=GGUF_Q4_0,
        model_config=model.config,
        tokenizer=tokenizer,
    )
    artifacts = export_with_pipeline(
        ExportRequest(
            model=model,
            sample_inputs=[],
            output_dir=output_dir,
            model_name=model_name,
            target="gguf",
            format="llama_cpp",
            options=options.to_context(),
        )
    )
    return artifacts.stage_outputs["write_gguf"]


def inspect_gguf(path: Path) -> tuple[gguf.GGUFReader, dict[str, int | float | str]]:
    """Open a GGUF file and summarize its tensor types."""
    reader = gguf.GGUFReader(str(path))
    q4_count = sum(
        tensor.tensor_type == gguf.GGMLQuantizationType.Q4_0 for tensor in reader.tensors
    )
    summary: dict[str, int | float | str] = {
        "path": str(path),
        "size_mib": round(path.stat().st_size / (1024**2), 2),
        "q4_0_tensors": q4_count,
        "float_tensors": len(reader.tensors) - q4_count,
    }
    print(summary)
    return reader, summary


def tensor_type(reader: gguf.GGUFReader, name: str) -> gguf.GGMLQuantizationType:
    """Return a named tensor's GGML type."""
    tensor = next(tensor for tensor in reader.tensors if tensor.name == name)
    return tensor.tensor_type


# %% [markdown]
# ## Qwen3-0.6B with min/max calibration
#
# Weight-only export does not require the whole Hugging Face model to be autoquantized. It is enough
# to convert the standard `torch.nn.Linear` projections whose weights we want to quantize. Here
# those are the attention and MLP projections inside each Qwen decoder layer. Token embeddings, the
# language-model head, and normalization parameters remain floating point.

# %%
qwen_model_id = "Qwen/Qwen3-0.6B"

qwen_model = AutoModelForCausalLM.from_pretrained(
    qwen_model_id,
    dtype=model_dtype,
    attn_implementation="eager",
).to(device)
qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_id)
qwen_calibration_loader, qwen_validation_loader = make_dataloaders(qwen_tokenizer)

qwen_fp_perplexity = evaluate_perplexity(
    qwen_model,
    qwen_validation_loader,
    qwen_tokenizer,
)
print(f"Qwen FP perplexity: {qwen_fp_perplexity:.4f}")

# %% [markdown]
# MPath selects the decoder-layer attention and MLP linears without encoding the model's complete
# module path in Python. Calling `ff.quantize_model` on each selected leaf with `recursive=False`
# uses FastForward's public `Linear -> QuantizedLinear` mapping without generating model-specific
# source code.

# %%
qwen_projections = ff.mpath.search(
    "**/layers/*/{self_attn,mlp}/[cls:torch.nn.Linear]",
    qwen_model,
)


def convert_linear(_name: str, module: torch.nn.Module) -> None:
    """Convert one selected Linear module in place."""
    ff.quantize_model(module, recursive=False)


qwen_projections.apply(convert_linear)
qwen_projection_names = [name for name, _ in qwen_projections.named_modules()]

if not qwen_projection_names:
    msg = "No Qwen attention or MLP projections were converted"
    raise RuntimeError(msg)

qwen_weight_quantizers = initialize_q4_0_weights(
    qwen_model,
    "**/layers/**/[quantizer:parameter/weight]",
)
assert len(qwen_weight_quantizers) == len(qwen_projection_names)
assert not ff.find_quantizers(qwen_model, "**/embed_tokens/[quantizer:parameter/weight]")
assert not ff.find_quantizers(qwen_model, "**/lm_head/[quantizer:parameter/weight]")

calibrate_weights(qwen_model, qwen_calibration_loader)
with ff.strict_quantization(False):
    qwen_q4_perplexity = evaluate_perplexity(
        qwen_model,
        qwen_validation_loader,
        qwen_tokenizer,
    )
print(f"Qwen Q4_0 perplexity: {qwen_q4_perplexity:.4f}")

# %% [markdown]
# The built-in adapter supplies Qwen3 metadata and tensor names. We derive a precision-policy
# variant that writes ordinary floating-point tensors as F16 while retaining normalization tensors
# and biases in F32.

# %%
qwen_adapter = replace(
    QWEN3_ADAPTER,
    float_type="F16",
    float_type_overrides={
        r".*norm.*": "F32",
        r".*\.bias": "F32",
    },
)
qwen_gguf_path = export_gguf(
    qwen_model,
    qwen_tokenizer,
    qwen_adapter,
    "qwen3-0.6b-q4_0",
)
qwen_reader, qwen_summary = inspect_gguf(qwen_gguf_path)

assert qwen_reader.get_field("general.architecture") is not None
assert tensor_type(qwen_reader, "blk.0.attn_q.weight") == gguf.GGMLQuantizationType.Q4_0
assert tensor_type(qwen_reader, "blk.0.ffn_up.weight") == gguf.GGMLQuantizationType.Q4_0
assert tensor_type(qwen_reader, "token_embd.weight") == gguf.GGMLQuantizationType.F16
if any(tensor.name == "output.weight" for tensor in qwen_reader.tensors):
    assert tensor_type(qwen_reader, "output.weight") == gguf.GGMLQuantizationType.F16

qwen_results = {
    "fp_perplexity": qwen_fp_perplexity,
    "q4_0_perplexity": qwen_q4_perplexity,
    **qwen_summary,
}
qwen_results

# %% [markdown]
# ## Qwen3-0.6B with GPTQ
#
# Min/max chooses quantization ranges from the weights alone. GPTQ additionally uses model inputs
# to approximate each projection's Hessian and update its weights to compensate for quantization
# error. Start from a fresh floating-point checkpoint so GPTQ is compared against the same baseline
# rather than the already quantized min/max model.

# %%
# MPath collections retain the model, so release them before clearing CUDA memory.
del qwen_model, qwen_projections, qwen_weight_quantizers, qwen_reader
release_cuda_memory()

qwen_gptq_model = AutoModelForCausalLM.from_pretrained(
    qwen_model_id,
    dtype=model_dtype,
    attn_implementation="eager",
    use_cache=False,
)

qwen_gptq_projections = ff.mpath.search(
    "**/layers/*/{self_attn,mlp}/[cls:torch.nn.Linear]",
    qwen_gptq_model,
)
qwen_gptq_projections.apply(convert_linear)
qwen_gptq_projection_names = [name for name, _ in qwen_gptq_projections.named_modules()]
if not qwen_gptq_projection_names:
    msg = "No Qwen attention or MLP projections were converted for GPTQ"
    raise RuntimeError(msg)

qwen_gptq_weight_quantizers = initialize_q4_0_weights(
    qwen_gptq_model,
    "**/layers/**/[quantizer:parameter/weight]",
)
assert len(qwen_gptq_weight_quantizers) == len(qwen_gptq_projection_names)

# Trace and optimize on CUDA. Qwen3-0.6B and its layer-local GPTQ working state fit on the device.
qwen_gptq_model.to(device)

# GPTQ only observes projection inputs, so labels and padding masks are unnecessary here.
qwen_gptq_calibration_set = [
    batch["input_ids"].to(device)
    for batch in sliced_tqdm(qwen_calibration_loader, gptq_calibration_steps)
]
optimize_with_gptq(qwen_gptq_model, qwen_gptq_calibration_set)

qwen_gptq_model.to(device)
with ff.strict_quantization(False):
    qwen_gptq_perplexity = evaluate_perplexity(
        qwen_gptq_model,
        qwen_validation_loader,
        qwen_tokenizer,
    )
print(f"Qwen GPTQ Q4_0 perplexity: {qwen_gptq_perplexity:.4f}")

qwen_gptq_gguf_path = export_gguf(
    qwen_gptq_model,
    qwen_tokenizer,
    qwen_adapter,
    "qwen3-0.6b-gptq-q4_0",
)
qwen_gptq_reader, qwen_gptq_summary = inspect_gguf(qwen_gptq_gguf_path)

assert qwen_gptq_reader.get_field("general.architecture") is not None
assert tensor_type(qwen_gptq_reader, "blk.0.attn_q.weight") == gguf.GGMLQuantizationType.Q4_0
assert tensor_type(qwen_gptq_reader, "blk.0.ffn_up.weight") == gguf.GGMLQuantizationType.Q4_0
assert tensor_type(qwen_gptq_reader, "token_embd.weight") == gguf.GGMLQuantizationType.F16

qwen_gptq_results = {
    "perplexity": qwen_gptq_perplexity,
    **qwen_gptq_summary,
}
qwen_gptq_results

# %% [markdown]
# ## Validate the exports with llama.cpp
#
# llama.cpp computes perplexity over the second half of each context window, using the first half
# only as context. The FastForward measurements above use the same scoring protocol. Both runtimes
# consume the same WikiText validation text and evaluate eight 1024-token chunks.

# %%
del qwen_gptq_model, qwen_gptq_projections, qwen_gptq_weight_quantizers, qwen_gptq_reader
release_cuda_memory()

(
    qwen_llama_ppl,
    qwen_llama_ppl_uncertainty,
    qwen_gptq_llama_ppl,
    qwen_gptq_llama_ppl_uncertainty,
) = validate_with_llama_cpp(
    qwen_gguf_path,
    qwen_gptq_gguf_path,
    raw_validation_set["text"],
)

qwen_ppl_delta = abs(qwen_q4_perplexity - qwen_llama_ppl) / qwen_q4_perplexity
qwen_gptq_ppl_delta = abs(qwen_gptq_perplexity - qwen_gptq_llama_ppl) / qwen_gptq_perplexity

comparison_table = f"""
| Variant | FastForward PPL | llama.cpp PPL | Relative delta |
| --- | ---: | ---: | ---: |
| FP16 | {qwen_fp_perplexity:.4f} | — | — |
| Min/max Q4_0 | {qwen_q4_perplexity:.4f} | {qwen_llama_ppl:.4f} ± {qwen_llama_ppl_uncertainty:.5f} | {qwen_ppl_delta:.2%} |
| GPTQ Q4_0 | {qwen_gptq_perplexity:.4f} | {qwen_gptq_llama_ppl:.4f} ± {qwen_gptq_llama_ppl_uncertainty:.5f} | {qwen_gptq_ppl_delta:.2%} |
"""
display(Markdown(comparison_table))

mismatches = []
if qwen_ppl_delta > perplexity_relative_tolerance:
    mismatches.append(f"min/max delta is {qwen_ppl_delta:.2%}")
if qwen_gptq_ppl_delta > perplexity_relative_tolerance:
    mismatches.append(f"GPTQ delta is {qwen_gptq_ppl_delta:.2%}")
if mismatches:
    details = "; ".join(mismatches)
    display(
        Markdown(
            f"> ⚠️ FastForward and llama.cpp perplexity differ by more than "
            f"{perplexity_relative_tolerance:.0%}: {details}."
        )
    )

qwen_results["llama_cpp_perplexity"] = qwen_llama_ppl
qwen_results["perplexity_relative_delta"] = qwen_ppl_delta
qwen_gptq_results["llama_cpp_perplexity"] = qwen_gptq_llama_ppl
qwen_gptq_results["perplexity_relative_delta"] = qwen_gptq_ppl_delta

# %% [markdown]
# `llama-perplexity` performs real CPU inference directly from each exported GGUF file. Small
# differences remain possible because FastForward evaluates with CUDA FP16 kernels while llama.cpp
# uses its CPU kernels. A relative delta above 5% is highlighted without failing documentation
# generation.

# %% [markdown]
# ## Results

# %%
print("Qwen3-0.6B:", qwen_results)
print("Qwen3-0.6B GPTQ:", qwen_gptq_results)

# %% [markdown]
# Both files contain Q4_0 transformer projections while retaining embeddings, norms, biases, and
# untargeted weights in floating point, and llama.cpp can load and evaluate both exports.

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
