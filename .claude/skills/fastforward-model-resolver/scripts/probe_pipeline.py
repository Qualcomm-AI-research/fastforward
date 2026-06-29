# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Probe whether transformers.pipeline works for a given model ID.

Attempts to build a pipeline and call preprocess() with a task-appropriate
dummy input. Reports tensor shapes on success, or the exception on failure.

Cache is handled automatically: when --cache-dir is given it sets HF_HOME so
the local cache is checked first and the network is only used if files are absent.

Usage:
    python probe_pipeline.py <model_id> [--pipeline-tag TAG]
        [--cache-dir DIR] [--hf-token TOKEN]

Exit codes:
    0  pipeline works — prints JSON with status=success and tensor shapes
    1  pipeline failed — prints JSON with status=failed and error details
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

from pathlib import Path


def _setup_hf_env(cache_dir: str | None) -> None:
    if not cache_dir:
        return
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(cache_dir) / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(cache_dir) / "hub"))


def _dummy_input(pipeline_tag: str | None):
    """Return a task-appropriate dummy input for pipe.preprocess()."""
    text_tags = {
        "fill-mask",
        "text-classification",
        "sentiment-analysis",
        "token-classification",
        "ner",
        "text-generation",
        "summarization",
        "translation",
        "question-answering",
        "zero-shot-classification",
        "feature-extraction",
        "text2text-generation",
    }
    vision_tags = {
        "image-classification",
        "object-detection",
        "image-segmentation",
        "depth-estimation",
        "mask-generation",
        "zero-shot-image-classification",
        "image-feature-extraction",
        "keypoint-matching",
    }
    audio_tags = {
        "automatic-speech-recognition",
        "audio-classification",
        "zero-shot-audio-classification",
        "text-to-audio",
        "text-to-speech",
    }

    tag = (pipeline_tag or "").lower()

    if tag in text_tags:
        return "dummy text", "text"
    if tag in vision_tags:
        from PIL import Image

        return Image.new("RGB", (224, 224)), "vision"
    if tag in audio_tags:
        import numpy as np

        return {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000}, "audio"
    # unknown tag — try text first
    return "dummy text", "unknown"


def _tensor_info(obj) -> dict | list | str:
    """Recursively extract shape/dtype from tensors in a nested structure."""
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return {"shape": list(obj.shape), "dtype": str(obj.dtype)}
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {k: _tensor_info(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_tensor_info(v) for v in obj]
    return str(type(obj).__name__)


def probe(
    model_id: str,
    pipeline_tag: str | None = None,
    cache_dir: str | None = None,
    token: str | None = None,
) -> dict:
    """Attempt to build a transformers pipeline and preprocess a dummy input."""
    _setup_hf_env(cache_dir)

    try:
        from transformers import pipeline as _pipeline
    except ImportError:
        return {
            "status": "failed",
            "reason": "transformers not installed",
            "error": "ImportError: transformers",
        }

    import torch

    device = 0 if torch.cuda.is_available() else -1

    kwargs: dict = {"model": model_id, "device": device}
    if pipeline_tag:
        kwargs["task"] = pipeline_tag
    if cache_dir:
        kwargs["model_kwargs"] = {"cache_dir": cache_dir}
    if token:
        kwargs["token"] = token

    try:
        pipe = _pipeline(**kwargs)
    except Exception as exc:
        return {
            "status": "failed",
            "reason": "pipeline construction failed",
            "error": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(limit=5),
        }

    resolved_tag = getattr(pipe, "task", pipeline_tag)
    dummy, modality = _dummy_input(resolved_tag)

    try:
        preprocessed = pipe.preprocess(dummy)
        tensor_shapes = _tensor_info(preprocessed)
    except Exception as exc:
        return {
            "status": "failed",
            "reason": "preprocess failed",
            "pipeline_tag": resolved_tag,
            "error": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(limit=5),
        }

    return {
        "status": "success",
        "pipeline_tag": resolved_tag,
        "modality": modality,
        "auto_class": type(pipe.model).__name__,
        "preprocessor_class": type(
            getattr(pipe, "tokenizer", None)
            or getattr(pipe, "image_processor", None)
            or getattr(pipe, "feature_extractor", None)
            or getattr(pipe, "processor", None)
        ).__name__,
        "tensor_shapes": tensor_shapes,
    }


def main() -> None:
    """CLI entry point for probing a transformers pipeline."""
    parser = argparse.ArgumentParser(description="Probe transformers.pipeline for a model")
    parser.add_argument("model_id", help="HuggingFace model ID")
    parser.add_argument("--pipeline-tag", default=None, help="Pipeline task tag (optional)")
    parser.add_argument("--cache-dir", default=None, help="Cache directory for model weights")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token for private repos")
    args = parser.parse_args()

    result = probe(
        args.model_id,
        pipeline_tag=args.pipeline_tag,
        cache_dir=args.cache_dir,
        token=args.hf_token,
    )
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["status"] == "success" else 1)


if __name__ == "__main__":
    main()
