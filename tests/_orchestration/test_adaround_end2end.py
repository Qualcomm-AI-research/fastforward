# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""End-to-end AdaRound through the orchestration framework.

The unit tests in `tests/algorithms/test_adaround.py` call `adaround` directly. This
module checks the other half: that the orchestration framework hands the algorithm the
activations it declares, layer after layer.

The first tests use a toy model and run on every commit. The last test is the
representative workload: it quantizes a pre-trained Llama to 4 bits and measures
perplexity, with the same recipe as
`test_gptq_end2end.py::test_gptq_layerwise_optimize_perplexity`. It carries the
`benchmark` mark, so it is deselected unless you give `--include-benchmark`.
"""

import functools
import importlib.util
import logging
import math
import random

from typing import Callable, Protocol, cast

import fastforward as ff
import pytest
import torch

from fastforward._orchestration.data_flow import InputActivations
from fastforward._orchestration.instruction_engine import OffloadEverything
from fastforward._orchestration.registry import AlgorithmSpec, MPathSelector
from fastforward._orchestration.trace import _MIN_TORCH_VERSION, trace
from packaging.version import Version

from ._models import ToyLlama

skip_without_datasets_and_transformers = pytest.mark.skipif(
    importlib.util.find_spec("datasets") is None
    or importlib.util.find_spec("transformers") is None,
    reason="requires `datasets` and `transformers` (install with `[docs]` extra)",
)


class _TokenizedData(Protocol):
    input_ids: torch.Tensor


def _get_c4(model_name: str, sequence_length: int, seed: int) -> list[torch.Tensor]:
    """Load a subset of C4 as a calibration set."""
    import datasets  # type: ignore[import-untyped]

    from transformers import AutoTokenizer

    traindata = datasets.load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    random.seed(seed)
    loader = []
    for _ in range(128):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tmp: _TokenizedData = tokenizer(
                traindata[i]["text"],
                return_tensors="pt",
                truncation=True,
                max_length=sequence_length * 2,
            )
            if tmp.input_ids.shape[1] >= sequence_length:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - sequence_length - 1)
        j = i + sequence_length
        loader.append(tmp.input_ids[:, i:j])
    return loader


def _get_wikitext2(
    model_name: str, nsamples: int, sequence_length: int, seed: int
) -> list[torch.Tensor]:
    """Build a WikiText-2 validation set for perplexity evaluation."""
    import datasets

    from transformers import AutoTokenizer

    testdata = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    testenc: _TokenizedData = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    random.seed(seed)
    sequences: list[torch.Tensor] = []
    for _ in range(nsamples):
        i = random.randint(0, testenc.input_ids.shape[1] - sequence_length - 1)
        j = i + sequence_length
        sequences.append(testenc.input_ids[:, i:j])
    return sequences


@torch.inference_mode()
def _evaluate(model: torch.nn.Module, dataset: list[torch.Tensor], device: torch.device) -> float:
    """Compute perplexity on a dataset of token-id sequences."""
    original_device = next(model.parameters()).device
    model.to(device)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
    nll_sum = 0.0
    total_tokens = 0

    for batch in dataset:
        batch = batch.to(device)
        logits = model(batch, use_cache=False).logits
        shift_logits = logits[:, :-1].reshape(-1, logits.size(-1))
        shift_labels = batch[:, 1:].reshape(-1)
        nll_sum += loss_fct(shift_logits, shift_labels).item()
        total_tokens += batch.size(1) - 1

    model.to(original_device)
    return math.exp(nll_sum / total_tokens)


pytestmark = pytest.mark.skipif(
    Version(torch.__version__.split("+", 1)[0]) < _MIN_TORCH_VERSION,
    reason=f"requires PyTorch >= {_MIN_TORCH_VERSION}",
)

_DIM = 32
_NUM_BITS = 3
_TARGETS = "**/layers/**/[cls:ff.nn.QuantizedLinear]"
_Q_PROJ = "layers.0.self_attn.q_proj.weight"
_DOWN_PROJ = "layers.1.mlp.down_proj.weight"


def _quantize_linear_leaves(module: torch.nn.Module) -> None:
    """Replace every `nn.Linear` leaf of `module` by its quantized counterpart.

    `ToyLlama` and its containers have no quantized counterpart, so `ff.quantize_model`
    cannot convert the model as a whole. Converting only the linear leaves gives the same
    situation that `layerwise_optimize` sees on a real model: `QuantizedLinear` targets
    inside containers that the framework does not need to know about.
    """
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            setattr(module, name, ff.quantize_model(child))
        else:
            _quantize_linear_leaves(child)


def _quantized_toy_llama() -> torch.nn.Module:
    """Build a `ToyLlama` with an uncalibrated weight quantizer on every linear layer."""
    torch.manual_seed(0)
    model = ToyLlama(dim=_DIM, n_layers=2).eval()
    _quantize_linear_leaves(model)
    weight_quantizers = ff.find_quantizers(
        model, ff.mpath.query(_TARGETS) / "[quantizer:parameter/weight]"
    )
    weight_quantizers.initialize(
        ff.nn.LinearQuantizer,
        num_bits=_NUM_BITS,
        granularity=ff.PerChannel(channel_dim=0),
        symmetric=False,
    )
    return model


def _weight(model: torch.nn.Module, name: str) -> torch.Tensor:
    """Return the weight of a targeted layer, addressed by its parameter name.

    This avoids `state_dict`, which detaches every parameter and therefore raises on the
    uninitialized quantization parameters of a model that is not calibrated yet.
    """
    return model.get_parameter(name)


def _output_error(
    model: torch.nn.Module, activation: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    with ff.strict_quantization(False), torch.no_grad():
        output = cast(torch.Tensor, model(activation))
    return (output - reference).pow(2).mean()


@pytest.mark.slow
def test_layerwise_optimize_with_adaround_beats_nearest_rounding() -> None:
    # GIVEN an unquantized toy transformer, a calibration set and its unquantized output
    torch.manual_seed(0)
    reference_model = ToyLlama(dim=_DIM, n_layers=2).eval()
    calibration = [torch.randn(2, 8, _DIM) for _ in range(4)]
    evaluation = torch.randn(2, 8, _DIM)
    with torch.no_grad():
        reference = reference_model(evaluation)

    # GIVEN a min-max baseline at the same bit width and granularity
    baseline = _quantized_toy_llama()
    with (
        torch.no_grad(),
        ff.strict_quantization(False),
        ff.estimate_ranges(baseline, ff.range_setting.smoothed_minmax),
    ):
        for batch in calibration:
            baseline(batch)
    baseline_error = _output_error(baseline, evaluation, reference)

    # WHEN we run AdaRound over the same layers through the orchestration framework
    model = _quantized_toy_llama()
    with torch.no_grad(), ff.strict_quantization(False):
        ff.layerwise_optimize(
            model,
            calibration,
            functools.partial(ff.algorithms.adaround, num_iterations=400),
            targets=ff.mpath.query(_TARGETS),
            sample_args=(calibration[0],),
        )
    adaround_error = _output_error(model, evaluation, reference)

    # THEN the model output is closer to the unquantized output than min-max rounding
    assert adaround_error < 0.95 * baseline_error


@pytest.mark.slow
def test_layerwise_optimize_with_adaround_runs_under_offloading() -> None:
    # GIVEN a quantized toy transformer, a calibration set and a CPU-to-CPU offloading strategy
    model = _quantized_toy_llama()
    calibration = [torch.randn(2, 8, _DIM) for _ in range(2)]
    initial_weight = _weight(model, _Q_PROJ).clone()
    cpu = torch.device("cpu")

    # WHEN we optimize with offloading enabled
    with torch.no_grad(), ff.strict_quantization(False):
        ff.layerwise_optimize(
            model,
            calibration,
            functools.partial(ff.algorithms.adaround, num_iterations=10),
            targets=ff.mpath.query(_TARGETS),
            sample_args=(calibration[0],),
            offloading=OffloadEverything(compute_device=cpu, storage_device=cpu),
        )

    # THEN the targeted weights were replaced by their AdaRound result
    assert not torch.allclose(initial_weight, _weight(model, _Q_PROJ))


@pytest.mark.slow
def test_layerwise_optimize_with_adaround_accepts_a_prebuilt_graph() -> None:
    # GIVEN a quantized toy transformer whose graph we trace ahead of time
    model = _quantized_toy_llama()
    calibration = [torch.randn(2, 8, _DIM) for _ in range(2)]
    with ff.strict_quantization(False):
        graph = trace(model, calibration[0])
    initial_weight = _weight(model, _DOWN_PROJ).clone()

    # WHEN we optimize through the supplied graph
    algorithm: Callable[..., None] = functools.partial(ff.algorithms.adaround, num_iterations=10)
    with torch.no_grad(), ff.strict_quantization(False):
        ff.layerwise_optimize(
            model, calibration, algorithm, targets=ff.mpath.query(_TARGETS), graph=graph
        )

    # THEN every targeted layer, including the last one, was optimized
    assert not torch.allclose(initial_weight, _weight(model, _DOWN_PROJ))


def _quantized_llama(
    model_name: str, num_bits: int, granularity: ff.granularity.Granularity, symmetric: bool
) -> torch.nn.Module:
    """Load the pre-trained model and give every decoder-layer linear a weight quantizer.

    `autoquantize` generates the quantized modules from the FP model, so this needs no
    hand-written quantized attention. The quantizers are not calibrated yet: the caller
    decides how the ranges are set.
    """
    from transformers import LlamaForCausalLM

    model: torch.nn.Module = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        use_cache=False,
    )
    model.eval()
    ff.autoquantize(
        model,
        output_path="_autoquantized_llama_benchmark.py",
        force_overwrite=True,
        auto_import=True,
    )
    ff.quantize_model(model, skip_quantized_modules=True)

    w_quantizers = ff.find_quantizers(
        model, ff.mpath.query(_TARGETS) / "[quantizer:parameter/weight]"
    )
    w_quantizers.initialize(
        ff.nn.LinearQuantizer, num_bits=num_bits, granularity=granularity, symmetric=symmetric
    )
    return model


@pytest.mark.benchmark
@skip_without_datasets_and_transformers
def test_adaround_layerwise_optimize_perplexity() -> None:
    """AdaRound W4 quantization of Llama-3.2-1B-Instruct, evaluated on WikiText-2.

    Uses the same model, bit width, granularity, calibration set and validation set as
    `test_gptq_end2end.py::test_gptq_layerwise_optimize_perplexity`, so the numbers
    can be compared directly. Three stages exercise the asymmetric (Eq. 25), symmetric
    (Eq. 21), and GPTQ-style (quantized-input symmetric) objectives; all three differ only in
    their declared data flows.
    """
    # Logging is set up here and not at module level to avoid affecting the fast toy tests.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_bits = 4
    symmetric = False
    granularity: ff.granularity.Granularity = ff.PerChannel(channel_dim=0)
    num_iterations = 10_000

    validation_set = _get_wikitext2(model_name, nsamples=128, sequence_length=2048, seed=0)
    calibration_set = _get_c4(model_name, sequence_length=2048, seed=0)

    # WHEN running AdaRound with the asymmetric objective (Eq. 25)
    adaround_fn: Callable[..., None] = functools.partial(
        ff.algorithms.adaround, num_iterations=num_iterations
    )
    adaround_targets = ff.mpath.query(_TARGETS)
    adaround_model = _quantized_llama(model_name, num_bits, granularity, symmetric)
    with torch.no_grad(), ff.strict_quantization(False):
        offloading = OffloadEverything(compute_device=device, storage_device=torch.device("cpu"))
        ff.layerwise_optimize(
            adaround_model,
            calibration_set,
            adaround_fn,
            targets=adaround_targets,
            sample_args=(calibration_set[0],),
            offloading=offloading,
        )
    with ff.strict_quantization(False):
        adaround_perplexity = _evaluate(adaround_model, validation_set, device)
    del adaround_model

    # WHEN running the symmetric objective (Eq. 21) by passing the original flow twice
    adaround_sym_model = _quantized_llama(model_name, num_bits, granularity, symmetric)
    with torch.no_grad(), ff.strict_quantization(False):
        spec = AlgorithmSpec(
            fn=adaround_fn,
            selector=MPathSelector(query=ff.mpath.query(_TARGETS)),
            flows=[InputActivations("original"), InputActivations("original")],
        )
        offloading = OffloadEverything(compute_device=device, storage_device=torch.device("cpu"))
        ff.layerwise_optimize(
            adaround_sym_model,
            calibration_set,
            spec,
            sample_args=(calibration_set[0],),
            offloading=offloading,
        )
    with ff.strict_quantization(False):
        adaround_sym_perplexity = _evaluate(adaround_sym_model, validation_set, device)
    del adaround_sym_model

    # WHEN running the GPTQ-style objective with quantized activations in both flows
    adaround_gptq_model = _quantized_llama(model_name, num_bits, granularity, symmetric)
    with torch.no_grad(), ff.strict_quantization(False):
        spec = AlgorithmSpec(
            fn=adaround_fn,
            selector=MPathSelector(query=ff.mpath.query(_TARGETS)),
            flows=[InputActivations("quantized"), InputActivations("quantized")],
        )
        offloading = OffloadEverything(compute_device=device, storage_device=torch.device("cpu"))
        ff.layerwise_optimize(
            adaround_gptq_model,
            calibration_set,
            spec,
            sample_args=(calibration_set[0],),
            offloading=offloading,
        )
    with ff.strict_quantization(False):
        adaround_gptq_perplexity = _evaluate(adaround_gptq_model, validation_set, device)
    del adaround_gptq_model

    print(f"Wiki2 PPL Llama-3.2-1B-Instruct W4 AdaRound (Eq. 25):  {adaround_perplexity:.4f}")
    print(f"Wiki2 PPL Llama-3.2-1B-Instruct W4 AdaRound (Eq. 21):  {adaround_sym_perplexity:.4f}")
    print(f"Wiki2 PPL Llama-3.2-1B-Instruct W4 AdaRound (GPTQ):    {adaround_gptq_perplexity:.4f}")
