# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import warnings

from typing import cast

import fastforward as ff
import torch

from datasets import load_dataset
from fastforward.nn.linear_quantizer import LinearQuantizer
from fastforward.nn.quantized_module import quantize_model
from fastforward.range_setting import estimate_ranges
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM, default_data_collator

from doc_helpers.export.benchmark.dataset import preprocess_dataset
from doc_helpers.export.benchmark.evaluate import evaluate_perplexity_metrics
from doc_helpers.export.benchmark.util import _sliced_tqdm_iterator, generate_attention_mask
from doc_helpers.export.llama import QuantizedLlamaForCausalLM as QuantizedLlamaForCausalLM

warnings.simplefilter(action="ignore", category=FutureWarning)


def get_llama_and_dataloaders(model_tag, device, sequence_length, from_tf=False):
    torch.random.manual_seed(324234)
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_tag,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        from_tf=from_tf,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_tag, attn_implementation="eager"
    )

    train_dataset_raw = load_dataset(path="wikitext", name="wikitext-2-raw-v1", split="train")
    val_dataset_raw = load_dataset(path="wikitext", name="wikitext-2-raw-v1", split="validation")

    train_dataset_preprocessed = preprocess_dataset(
        dataset=train_dataset_raw,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        sequence_collate_function="join_nn",
    )

    val_dataset_preprocessed = preprocess_dataset(
        dataset=val_dataset_raw,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        sequence_collate_function="join_nn",
    )

    ptq_dataloader = DataLoader(
        dataset=train_dataset_preprocessed,
        batch_size=1,
        num_workers=4,
        shuffle=True,
        collate_fn=default_data_collator,
    )

    test_dataloader = DataLoader(
        dataset=val_dataset_preprocessed,
        batch_size=1,
        num_workers=4,
        shuffle=False,
        collate_fn=default_data_collator,
    )

    quantize_llama(model, ptq_dataloader, test_dataloader, device=device)

    return model, ptq_dataloader, test_dataloader


def quantize_llama(
    model,
    ptq_dataloader,
    test_dataloader,
    ptq_dataloader_limit=16,
    test_dataloader_limit=16,
    device="cpu",
):
    quantize_model(model)
    model = cast(QuantizedLlamaForCausalLM, model)

    # Model weight quantizers
    weight_quantizers = ff.find_quantizers(model, "**/self_attn/**/[quantizer:parameter/weight]")
    weight_quantizers |= ff.find_quantizers(model, "**/mlp/**/[quantizer:parameter/weight]")

    layernorm_weight_quantizers = ff.find_quantizers(
        model, "**/input_layernorm/[quantizer:parameter/weight]"
    )
    layernorm_weight_quantizers |= ff.find_quantizers(
        model, "**/post_attention_layernorm/[quantizer:parameter/weight]"
    )
    layernorm_weight_quantizers |= ff.find_quantizers(model, "**/norm/[quantizer:parameter/weight]")

    embed_weight_quantizers = ff.find_quantizers(
        model, "**/embed_tokens/[quantizer:parameter/weight]"
    )
    lm_head_weight_quantizers = ff.find_quantizers(model, "**/lm_head/[quantizer:parameter/weight]")

    # Model activation quantizers
    activation_quantizers = ff.find_quantizers(model, "**/mlp/**/[quantizer:activation/output]")
    activation_quantizers |= ff.find_quantizers(
        model, "**/self_attn/**/[quantizer:activation/output]"
    )

    activation_quantizers |= ff.find_quantizers(
        model, "**/input_layernorm/[quantizer:activation/output]"
    )
    activation_quantizers |= ff.find_quantizers(
        model, "**/post_attention_layernorm/[quantizer:activation/output]"
    )
    activation_quantizers |= ff.find_quantizers(model, "**/norm/[quantizer:activation/output]")

    activation_quantizers |= ff.find_quantizers(model, "**/attn_res_act_quantizer")
    activation_quantizers |= ff.find_quantizers(model, "**/mlp_res_act_quantizer")
    activation_quantizers |= ff.find_quantizers(model, "**/lm_head/[quantizer:activation/output]")

    # Quantizer initialization
    weight_quantizers.initialize(
        LinearQuantizer, num_bits=8, granularity=ff.granularity.PerChannel(0)
    )
    embed_weight_quantizers.initialize(LinearQuantizer, num_bits=16, symmetric=False)
    lm_head_weight_quantizers.initialize(
        LinearQuantizer, num_bits=8, granularity=ff.granularity.PerChannel(0)
    )
    layernorm_weight_quantizers.initialize(LinearQuantizer, num_bits=16)

    activation_quantizers.initialize(LinearQuantizer, num_bits=16, symmetric=False)

    model.to(device)

    for quantizer in (
        weight_quantizers
        | activation_quantizers
        | embed_weight_quantizers
        | layernorm_weight_quantizers
        | lm_head_weight_quantizers
    ):
        quantizer.module.to(device)

    with estimate_ranges(model, ff.range_setting.running_minmax):
        for batch in _sliced_tqdm_iterator(ptq_dataloader, ptq_dataloader_limit):
            sequence_length = batch["input_ids"].shape[1]
            attention_mask = generate_attention_mask(sequence_length)
            batch["attention_mask"] = attention_mask
            model(**{k: v.to(device) for k, v in batch.items()})

    task_results = evaluate_perplexity_metrics(model, test_dataloader, test_dataloader_limit)
    # We need the quantizer parameters (weight and output) to be identical for the embedding layer.
    # If they differ, QNN will generate an unhelpful error message.
    model.model.embed_tokens.output_quantizer.scale = (
        model.model.embed_tokens.weight_quantizer.scale.clone()
    )
    model.model.embed_tokens.output_quantizer.offset = (
        model.model.embed_tokens.weight_quantizer.offset.clone()
    )

    print("LLaMA model performance:")
    print(f"---> Loss: {task_results['loss']}")
    print(f"---> Per Token PPL: {task_results['per_token_perplexity']}")
