# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Resolve a model source (local_path or git_url) to a local directory path."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys

from pathlib import Path


def _cache_key(*parts: str) -> str:
    data = "|".join(parts).encode("utf-8", errors="ignore")
    return hashlib.sha256(data).hexdigest()[:16]


def _run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        check=False,
        text=True,
        capture_output=True,
    )


def _local_git_fallback_path(model_source: str) -> Path | None:
    source = model_source.rstrip("/")
    name = source.rsplit("/", maxsplit=1)[-1]
    if name.endswith(".git"):
        name = name[:-4]
    for candidate in [Path.cwd() / "models" / name, Path.cwd() / name]:
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()
    return None


def resolve_local(source: str) -> dict:
    """Resolve a local filesystem path to an absolute model directory."""
    path = Path(source).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        return {"error": f"Local model path does not exist or is not a directory: {path}"}
    return {"model_path": str(path), "resolved_revision": None, "error": None}


def resolve_git(model_source: str, source_ref: str | None, cache_dir: Path) -> dict:
    """Clone or update a git repo and return its local path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(model_source)
    repo_dir = cache_dir / f"git_{key}"

    if not repo_dir.exists():
        result = _run(["git", "clone", model_source, str(repo_dir)])
        if result.returncode != 0:
            fallback = _local_git_fallback_path(model_source)
            if fallback is None:
                return {
                    "error": (
                        f"Failed to clone git source.\n"
                        f"Command: git clone {model_source}\n"
                        f"stderr: {result.stderr.strip()}"
                    )
                }
            return {
                "model_path": str(fallback),
                "resolved_revision": "local-fallback",
                "error": None,
            }
    else:
        result = _run(["git", "fetch", "--all", "--tags"], cwd=repo_dir)
        if result.returncode != 0:
            return {"error": f"Failed to fetch updates. stderr: {result.stderr.strip()}"}

    if source_ref:
        result = _run(["git", "checkout", source_ref], cwd=repo_dir)
        if result.returncode != 0:
            return {
                "error": (f"Failed to checkout ref '{source_ref}'. stderr: {result.stderr.strip()}")
            }

    rev = _run(["git", "rev-parse", "HEAD"], cwd=repo_dir)
    resolved_revision = rev.stdout.strip() if rev.returncode == 0 else None
    return {"model_path": str(repo_dir), "resolved_revision": resolved_revision, "error": None}


def main() -> None:
    """CLI entry point for resolving a model source to a local directory."""
    parser = argparse.ArgumentParser(description="Resolve model source to local directory")
    parser.add_argument("--source-type", required=True, choices=["local_path", "git_url"])
    parser.add_argument("--model-source", required=True)
    parser.add_argument("--source-ref", default=None)
    parser.add_argument(
        "--cache-dir",
        default=str(Path.home() / ".cache" / "fastforward" / "model_sources"),
    )
    args = parser.parse_args()

    if args.source_type == "local_path":
        result = resolve_local(args.model_source)
    else:
        result = resolve_git(args.model_source, args.source_ref, Path(args.cache_dir))

    print(json.dumps(result))
    if result.get("error"):
        sys.exit(1)


if __name__ == "__main__":
    main()
