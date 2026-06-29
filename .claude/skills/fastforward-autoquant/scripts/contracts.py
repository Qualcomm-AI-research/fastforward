# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Input contracts for the autoquant skill runner."""

from __future__ import annotations

import typing

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Request:
    """Validated request payload for the autoquant runner.

    The package at output_dir must already have __init__.py with working
    load_model and model_inputs implementations — produced by
    fastforward-model-resolver or fastforward-model-discovery beforehand.
    """

    intent: str
    output_dir: Path
    seed: int = 0
    max_workers: int = 1
    command_prefix: str | None = None
    overwrite_output_files: bool = False
    generate_only: bool = False

    @classmethod
    def from_mapping(cls, payload: typing.Mapping[str, typing.Any]) -> "Request":
        """Construct a Request from a raw mapping, validating required fields."""
        intent = str(payload.get("intent", "autoquantize_model"))
        if intent != "autoquantize_model":
            msg = f"Unsupported intent '{intent}'. Expected 'autoquantize_model'."
            raise ValueError(msg)

        selected_output_dir = (
            payload.get("output_dir") or payload.get("artifacts_dir") or "./artifacts"
        )
        resolved_output_dir = Path(str(selected_output_dir)).expanduser().resolve()
        if not resolved_output_dir.name.isidentifier():
            msg = (
                f"output_dir basename '{resolved_output_dir.name}' is not a valid Python "
                "module name — main.py imports it directly via __import__(). Use "
                "underscores instead of dashes/dots, e.g. 'sam_3_artifacts' instead of "
                "'sam-3-artifacts'."
            )
            raise ValueError(msg)

        max_workers = int(payload.get("max_workers", 1))
        if max_workers < 1:
            raise ValueError("request.max_workers must be >= 1")

        overwrite_output_files = payload.get("overwrite_output_files", False)
        if not isinstance(overwrite_output_files, bool):
            raise ValueError("request.overwrite_output_files must be boolean when provided")

        generate_only = payload.get("generate_only", False)
        if not isinstance(generate_only, bool):
            raise ValueError("request.generate_only must be boolean when provided")

        return cls(
            intent=intent,
            output_dir=resolved_output_dir,
            seed=int(payload.get("seed", 0)),
            max_workers=max_workers,
            command_prefix=(
                str(payload.get("command_prefix"))
                if payload.get("command_prefix") is not None
                else None
            ),
            overwrite_output_files=overwrite_output_files,
            generate_only=generate_only,
        )
