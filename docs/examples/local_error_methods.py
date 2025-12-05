# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import functools
import os

from typing import Any, Callable, Collection, Iterable, Iterator, TypeAlias

import datasets
import fastforward as ff
import torch

from datasets import load_dataset
from fastforward._orchestration.graph_module import GraphModule, LocalOptimizer, SubgraphSpec
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM, default_data_collator

from doc_helpers.quick_start.quantized_models import quantized_llama as quantized_llama
from doc_helpers.quick_start.quick_start_utils import tokenize_dataset

datasets.utils.logging.get_logger("datasets.packaged_modules.cache").setLevel("ERROR")


##########
# HF Setup:
# 1. Load LlaMA-3.2-1B-Instruct
# 2. Init a model with quantizers
# 3. Quantize the model with a simple approach as a baseline to the MSE approach.
##########


def load_model(
    model_name_or_path: str = os.environ.get(
        "FF_QUICKSTART_MODEL", "meta-llama/Llama-3.2-1B-Instruct"
    ),
    model_dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> LlamaForCausalLM:
    """Load and return a FF-quantized Llama-3.2-1B-Instruct model.

    Args:
        model_name_or_path: A Hugging Face compatible name for a LlamaForCausalLM model.
        model_dtype: Data type to load the model weights in.
        device: Torch device.

    Returns:
        A LlamaForCausalLM model.
    """
    from_tf = bool(".ckpt" in model_name_or_path)
    model = LlamaForCausalLM.from_pretrained(
        model_name_or_path, from_tf=from_tf, torch_dtype=model_dtype
    )
    model.to(device)
    return model


def quantize_model(
    model: LlamaForCausalLM,
    data_loader: Iterable[dict[str, torch.Tensor]],
    w_bits: int,
    device: torch.device,
) -> None:
    """Perform a simple quantization of the model.

    Quantizes `model` in-place.

    Args:
        model: A Hugging Face LlamaForCausalLM model.
        data_loader: An iterable that serves inputs for `model`.
        w_bits: Number of bits used to quantize the weights of `model`.
        device: Torch device.
    """
    ff.quantize_model(model)

    w_quantizers = ff.find_quantizers(model, "**/layers/*/self_attn/*/[quantizer:parameter/weight]")
    w_quantizers |= ff.find_quantizers(model, "**/layers/*/mlp/*/[quantizer:parameter/weight]")
    w_quantizers.initialize(ff.nn.LinearQuantizer, num_bits=w_bits, granularity=ff.PerChannel())
    model.to(device)

    with (
        torch.no_grad(),
        ff.strict_quantization(False),
        ff.estimate_ranges(model, ff.range_setting.running_minmax),
    ):
        x = next(iter(data_loader))
        model(**x)


def quantizer_parameters(module: torch.nn.Module) -> Iterator[torch.nn.Parameter]:
    """Return all quantizer parameters for `module`.

    Args:
        module: A torch module to extract quantizer parameters from.

    Returns:
        An iterator of quantizer parameters.
    """
    for _, quantizer in ff.nn.quantized_module.named_quantizers(module, recurse=True):
        for parameter in quantizer.parameters():
            yield parameter


##########
# GraphModule "linearlization":
# 1. Add Custom module to handle "HF overhead" in forward pass.
# 2. Transform LlaMA-3.2-1B-Instruct into a GraphModule.
##########


class PositionIDs(torch.nn.Module):
    """Generate Positional IDs for a RoPE forward pass."""

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        position_ids = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
        return position_ids.unsqueeze(dim=0)


def graph_module(model: quantized_llama.QuantizedLlamaForCausalLM) -> GraphModule:
    """Return a GraphModule representation of QuantizedLlamaForCausalLM.

    Args:
        model: A QuantizedLlamaForCausalLM model to convert.

    Returns:
        A GraphModule representation of the model.
    """
    graph = GraphModule()

    # Batch is defined below, its a dict of 'input_ids', 'attention_mask', and 'labels'.
    batch = graph.add_input("batch")
    input_ids = batch["input_ids"]

    hidden = graph.add_node("embed_tokens", model.model.embed_tokens, [input_ids])

    # small setup for RoPE Position Embeddings required.
    position_ids = graph.add_node("position_ids", PositionIDs(), [input_ids])
    position_embeddings = graph.add_node(
        "position_embeddings", model.model.rotary_emb, [hidden, position_ids]
    )
    decoder_kwargs = {"position_embeddings": position_embeddings}

    for i, layer in enumerate(model.model.layers):
        hidden = graph.add_node(f"llama_decoder_layer_{i}", layer, [hidden], decoder_kwargs)
        hidden = hidden[0]  # Retrieve the first element of the tuple output.

    norm = graph.add_node("norm", model.model.norm, [hidden])
    lm_head = graph.add_node("lm_head", model.lm_head, [norm])

    # Note: output here is raw logits, HF typically returns a struct.
    graph.add_output(lm_head)
    return graph


##########
# Optimization Spec Creation
# 1. Using mpath, select the layers we want to run a local error method on.
#    Here we search for all decoder layers and attach `fn` to optimize them.
# 2. The function we attach is a simple mean squared error between the quantized and
#    non-quantized forward pass.
##########


DatasetBatch: TypeAlias = Iterable[tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]]
LocalErrorFn: TypeAlias = Callable[[torch.nn.Module, Collection[Any]], None]


def decoder_layer_only_spec(graph: GraphModule, fn: LocalErrorFn) -> list[SubgraphSpec]:
    """Generate a spec of QuantizedLlamaDecoderLayers to optimize.

    Args:
        graph: GraphModule from QuantizedLlamaForCausalLM.
        fn: A function to apply to each subgraph.

    Returns:
        A list of subgraphspecs corresponding to decoder layers.
    """
    layers = ff.mpath.search("**/[cls:quantized_llama.QuantizedLlamaDecoderLayer]", graph)
    return [
        SubgraphSpec(graph.node_ref(layer.module), graph.node_ref(layer.module), fn=fn)
        for layer in layers
    ]


def opt_mse(
    module: torch.nn.Module, dataset: DatasetBatch, learning_rate: float, n_epochs: int
) -> None:
    """Optimize a module using Mean Squared Error between normal and quantized forward passes.

    Args:
        module: A torch module to optimize.
        dataset: An iterable of input batches and position embeddings.
        learning_rate: Learning rate for the Adam optimizer.
        n_epochs: Number of optimization epochs.
    """
    optimizer = torch.optim.Adam(quantizer_parameters(module), lr=learning_rate)
    for _ in range(n_epochs):
        for inputs, position_embeddings in dataset:
            inputs = inputs.detach()
            position_embeddings = (position_embeddings[0].detach(), position_embeddings[1].detach())

            optimizer.zero_grad()

            # Perform a quantized forward pass (default).
            with torch.no_grad():
                (outputs_quantized,) = module(inputs, position_embeddings=position_embeddings)

            # Perform a non-quantized forward pass.
            with ff.disable_quantization(module):
                (outputs_normal,) = module(inputs, position_embeddings=position_embeddings)

            loss = torch.mean((outputs_normal - outputs_quantized) ** 2)
            loss.backward()
            optimizer.step()


##########
# Utility functions
##########


@torch.no_grad()
def assert_model_and_graph_equality(
    model: quantized_llama.QuantizedLlamaForCausalLM,
    graph: GraphModule,
    data_loader: Iterable[dict[str, torch.Tensor]],
    n_iters: int = 5,
) -> None:
    """Assert that the model and graph produce the same output.

    Args:
        model: A QuantizedLlamaForCausalLM model.
        graph: A GraphModule representation of the model.
        data_loader: An iterable that serves inputs for the model.
        n_iters: Number of iterations to test equality.

    Raises:
        AssertionError if outputs don't match.
    """
    for _ in range(n_iters):
        with ff.strict_quantization(False):
            batch = next(iter(data_loader))
            model_out = model(**batch)
            graph_out = graph(batch)

        assert torch.allclose(model_out.logits, graph_out, atol=1e-5)


def prepare_batch(features: list[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    """Prepare a batch of features for model input.

    Args:
        features: A list of feature dictionaries to collate.
        device: Torch device to move the batch to.

    Returns:
        A dictionary containing input_ids, attention_mask, and labels tensors.
    """
    batch = default_data_collator(features)
    return {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "labels": batch["labels"].to(torch.long).to(device),
    }


def wikitext2_generators(
    device: torch.device,
    batch_size: int = 1,
    sequence_length: int = 1024,
    valid_percent: int | None = 20,
    train_percent: int | None = 5,
    model_name_or_path: str = os.environ.get(
        "FF_QUICKSTART_MODEL", "meta-llama/Llama-3.2-1B-Instruct"
    ),
) -> tuple[Iterable[dict[str, torch.Tensor]], Iterable[dict[str, torch.Tensor]]]:
    """Return train and validation data loaders for wikitext-2 dataset.

    Args:
        device: Torch device to move batches to.
        batch_size: Number of samples per batch.
        sequence_length: Maximum sequence length for tokenization.
        valid_percent: Percentage of validation split to use. None for full split.
        train_percent: Percentage of training split to use. None for full split.
        model_name_or_path: Model path for loading the tokenizer.

    Returns:
        A tuple of (valid_loader, train_loader) DataLoader objects.
    """
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, legacy=False, use_fast=True)

    # Load Dataset
    _valid_split = "validation" if valid_percent is None else f"validation[:{valid_percent}%]"
    _train_split = "train" if train_percent is None else f"train[:{train_percent}%]"
    validset = load_dataset("wikitext", "wikitext-2-raw-v1", split=_valid_split)
    trainset = load_dataset("wikitext", "wikitext-2-raw-v1", split=_train_split)

    # Tokenize Dataset
    tokenized_validset = tokenize_dataset(validset, tokenizer, sequence_length)
    tokenized_trainset = tokenize_dataset(trainset, tokenizer, sequence_length)

    # Create Dataloader
    valid_loader = DataLoader(
        tokenized_validset,
        batch_size,
        collate_fn=functools.partial(prepare_batch, device=device),
        shuffle=False,
    )
    train_loader = DataLoader(
        tokenized_trainset,
        batch_size,
        collate_fn=functools.partial(prepare_batch, device=device),
        shuffle=True,
    )
    return valid_loader, train_loader


def evaluate_model(
    model: torch.nn.Module,
    valid_loader: Iterable[dict[str, torch.Tensor]],
    max_num_batches: int | None = None,
) -> float:
    """Evaluate model perplexity on validation data.

    Args:
        model: A language model to evaluate.
        valid_loader: DataLoader providing validation batches.
        max_num_batches: Maximum number of batches to evaluate. None for all batches.

    Returns:
        The perplexity score as a float.
    """
    model.eval()
    losses = []
    for batch_idx, batch in enumerate(tqdm(valid_loader)):
        if max_num_batches is not None and batch_idx >= max_num_batches:
            break

        with torch.no_grad():
            outputs = model(**batch)

        losses.append(outputs.loss)

    eval_loss = torch.stack(losses).mean()
    perplexity = torch.exp(eval_loss)
    return float(perplexity)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.bfloat16
    w_bits: int = 8

    model_name_or_path = os.environ.get("FF_QUICKSTART_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
    model = load_model(model_name_or_path, device=device, model_dtype=model_dtype)
    valid_loader, train_loader = wikitext2_generators(device=device)

    orig_perplexity = evaluate_model(model, valid_loader, max_num_batches=None)
    print(
        f"Perplexity over wikitext-validation using full-precision model ({model_dtype}): {orig_perplexity:.4}"
    )

    ff.set_strict_quantization(False)

    # Perform weight quantization.
    quantize_model(model, train_loader, w_bits=w_bits, device=device)

    w_quant_perplexity = evaluate_model(model, valid_loader, max_num_batches=None)
    print("Perplexity over wikitext-validation:")
    print(f" - Original model:       {orig_perplexity:.4f}  ({model_dtype}) ")
    print(f" - W-Quantized model:    {w_quant_perplexity:.4f}  (W{w_bits})")

    graph = graph_module(model)

    # After GraphModule construction, the model and the graph still produce the same outputs.
    assert_model_and_graph_equality(model, graph, train_loader)

    specs = decoder_layer_only_spec(
        graph, fn=functools.partial(opt_mse, learning_rate=1e-3, n_epochs=5)
    )

    # Perform layerwise optimization (decoder layers).
    optimizer = LocalOptimizer(graph, specs)
    optimizer.optimize(train_loader)

    mse_quant_perplexity = evaluate_model(model, valid_loader, max_num_batches=None)
    print("Perplexity over wikitext-validation:")
    print(f" - Original model:       {orig_perplexity:.4f}  ({model_dtype}) ")
    print(f" - W-Quantized model:    {w_quant_perplexity:.4f}  (W{w_bits})")
    print(f" - MSE-Quantized model:    {mse_quant_perplexity:.4f}")
