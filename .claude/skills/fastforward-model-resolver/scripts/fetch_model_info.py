# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Fetch model metadata for hf-model-resolver.

Uses huggingface_hub to read model files.  When --model-dir or
--tokenizer-dir is given, sets HUGGINGFACE_HUB_CACHE and HF_HUB_OFFLINE=1
for that call so the local cache is used and the network is never contacted.
When no dirs are given, the library uses its defaults and contacts HF normally.

Tries model files from --model-dir, tokenizer files from --tokenizer-dir,
and falls back to the other dir if a file is missing.

Usage:
    python fetch_model_info.py <model_id>
        [--hf-token TOKEN]
        [--model-dir DIR]     # dir containing models--org--model/ entries
        [--tokenizer-dir DIR] # dir containing models--org--model/ entries
        [--checkpoint PATH]   # local .pt/.bin, passed through only

Exit codes:
    0  success
    1  unexpected error
    2  access denied or model not found
"""

from __future__ import annotations

import argparse
import json
import re
import sys

from pathlib import Path


def _auth_guidance(model_id: str, http_code: int) -> str:
    return (
        f"HTTP {http_code}: model '{model_id}' requires authentication. "
        "To verify access manually, run:\n\n"
        f"  curl -H 'Authorization: Bearer YOUR_TOKEN' "
        f"https://huggingface.co/{model_id}/resolve/main/config.json\n\n"
        "If that succeeds, re-run with --hf-token YOUR_TOKEN.\n"
        "To get a token: https://huggingface.co/settings/tokens\n"
        "For gated models (e.g. Llama) accept the licence at:\n"
        f"  https://huggingface.co/{model_id}"
    )


def _read_from_disk(model_id: str, filename: str, *cache_dirs: str | None) -> str | None:
    """Read a file directly from HF hub layout on disk — no network, no library.

    Layout: {cache_dir}/models--{org}--{model}/snapshots/{hash}/{filename}
    Tries the most recently modified snapshot first.
    """
    hub_name = "models--" + model_id.replace("/", "--")
    for base in cache_dirs:
        if not base:
            continue
        snapshots = Path(base) / hub_name / "snapshots"
        if not snapshots.is_dir():
            continue
        for snap in sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            candidate = snap / filename
            if candidate.is_file():
                return candidate.read_text(errors="replace")
    return None


def _read_from_network(model_id: str, filename: str, token: str | None) -> str | None:
    """Fetch a file from HuggingFace via hf_hub_download."""
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(repo_id=model_id, filename=filename, token=token)
        return Path(path).read_text(errors="replace")
    except Exception as exc:
        exc_name = type(exc).__name__
        msg = str(exc).lower()
        if (
            "gated" in exc_name.lower()
            or "gated" in msg
            or "401" in msg
            or "403" in msg
            or "unauthorized" in msg
            or "forbidden" in msg
        ):
            code = 403 if ("403" in msg or "gated" in msg or "forbidden" in msg) else 401
            raise PermissionError(_auth_guidance(model_id, code)) from exc
        if (
            "not found" in exc_name.lower()
            or "404" in msg
            or "entry" in exc_name.lower()
            or "repository" in exc_name.lower()
        ):
            return None
        raise


def _read_hf_file(
    model_id: str,
    filename: str,
    token: str | None,
    primary_dir: str | None,
    fallback_dir: str | None,
) -> str | None:
    """Read a file from disk cache if dirs provided, otherwise from network."""
    if primary_dir or fallback_dir:
        return _read_from_disk(model_id, filename, primary_dir, fallback_dir)
    return _read_from_network(model_id, filename, token)


def _extract_pipeline_tag(readme: str) -> str | None:
    match = re.match(r"^---\n(.*?)\n---", readme, re.DOTALL)
    if not match:
        return None
    for line in match.group(1).splitlines():
        if line.startswith("pipeline_tag:"):
            return line.split(":", 1)[1].strip()
    return None


def _extract_code_blocks(readme: str) -> list[str]:
    return re.findall(r"```python\n(.*?)```", readme, re.DOTALL)


def _relevant_config_keys(config: dict) -> dict:
    keep = {
        "model_type",
        "architectures",
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "max_position_embeddings",
        "image_size",
        "patch_size",
        "num_channels",
        "num_classes",
        "id2label",
        "problem_type",
        "torch_dtype",
    }
    return {k: v for k, v in config.items() if k in keep}


def fetch_model_info(
    model_id: str,
    token: str | None = None,
    model_dir: str | None = None,
    tokenizer_dir: str | None = None,
    checkpoint: str | None = None,
) -> dict:
    """Fetch model metadata.

    Model files (config.json, preprocessor_config.json, README.md) are read
    from model_dir first, then tokenizer_dir, then network.
    Tokenizer files (tokenizer_config.json) are read from tokenizer_dir first,
    then model_dir, then network.
    """
    result: dict = {"model_id": model_id}

    def read_model_file(filename: str) -> str | None:
        return _read_hf_file(model_id, filename, token, model_dir, tokenizer_dir)

    def read_tokenizer_file(filename: str) -> str | None:
        return _read_hf_file(model_id, filename, token, tokenizer_dir, model_dir)

    # config.json — required; first access check
    try:
        config_text = read_model_file("config.json")
    except PermissionError as exc:
        result["access"] = "denied"
        result["guidance"] = str(exc)
        return result

    if config_text is None:
        result["access"] = "not_found"
        dirs = [d for d in (model_dir, tokenizer_dir) if d]
        if dirs:
            result["guidance"] = (
                f"config.json not found in local cache for '{model_id}'. "
                f"Searched: {dirs}. "
                "Check that the cache path is correct and the model has been downloaded."
            )
        else:
            result["guidance"] = (
                f"No model found at '{model_id}' on HuggingFace. "
                "Check the model ID at https://huggingface.co/models"
            )
        return result

    result["access"] = "ok"
    try:
        config = json.loads(config_text)
    except json.JSONDecodeError:
        config = {}
    result["config"] = _relevant_config_keys(config)

    # tokenizer_config.json
    try:
        tok_text = read_tokenizer_file("tokenizer_config.json")
    except PermissionError:
        tok_text = None
    if tok_text:
        try:
            tok = json.loads(tok_text)
            result["tokenizer_class"] = tok.get("tokenizer_class")
            result["model_max_length"] = tok.get("model_max_length")
        except json.JSONDecodeError:
            result["tokenizer_class"] = None
            result["model_max_length"] = None
    else:
        result["tokenizer_class"] = None
        result["model_max_length"] = None

    # preprocessor_config.json
    try:
        pre_text = read_model_file("preprocessor_config.json")
    except PermissionError:
        pre_text = None
    if pre_text:
        try:
            pre = json.loads(pre_text)
            result["preprocessor_class"] = pre.get("feature_extractor_type") or pre.get(
                "image_processor_type"
            )
            result["image_size"] = pre.get("size") or pre.get("crop_size")
        except json.JSONDecodeError:
            result["preprocessor_class"] = None
            result["image_size"] = None
    else:
        result["preprocessor_class"] = None
        result["image_size"] = None

    # README.md
    try:
        readme_text = read_model_file("README.md")
    except PermissionError:
        readme_text = None
    result["pipeline_tag"] = _extract_pipeline_tag(readme_text) if readme_text else None
    result["readme_code_blocks"] = _extract_code_blocks(readme_text) if readme_text else []

    result["model_dir"] = model_dir
    result["tokenizer_dir"] = tokenizer_dir
    result["checkpoint"] = checkpoint

    return result


def main() -> None:
    """CLI entry point for fetching HuggingFace model metadata."""
    parser = argparse.ArgumentParser(description="Fetch HuggingFace model metadata")
    parser.add_argument("model_id", help="HuggingFace model ID, e.g. bert-base-uncased")
    parser.add_argument("--hf-token", default=None, help="Token for private/gated repos")
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Dir containing models--org--model/ entries for model files. "
        "Used offline (HF_HUB_OFFLINE=1).",
    )
    parser.add_argument(
        "--tokenizer-dir",
        default=None,
        help="Dir containing models--org--model/ entries for tokenizer files. "
        "Used offline (HF_HUB_OFFLINE=1).",
    )
    parser.add_argument(
        "--checkpoint", default=None, help="Local checkpoint .pt/.bin (passed through)"
    )
    args = parser.parse_args()

    try:
        info = fetch_model_info(
            args.model_id,
            token=args.hf_token,
            model_dir=args.model_dir,
            tokenizer_dir=args.tokenizer_dir,
            checkpoint=args.checkpoint,
        )
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        sys.exit(1)

    access = info.get("access")
    if access in ("denied", "not_found"):
        print(info.get("guidance", ""), file=sys.stderr)
        print(json.dumps(info, indent=2))
        sys.exit(2)

    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
