# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Entrypoint for skill-local autoquant execution."""

from __future__ import annotations

import json
import sys
import typing

from pathlib import Path

try:
    from .contracts import Request
    from .runner import run
except ImportError:  # pragma: no cover
    from contracts import Request
    from runner import run

_JSON = dict[str, typing.Any]


def handle_request(payload: typing.Mapping[str, typing.Any]) -> _JSON:
    """Validate payload and execute the skill-local runner."""
    request = Request.from_mapping(payload)
    return run(request)


def _main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python -m .agents.skills.fastforward-autoquant.scripts.entrypoint <request.json|json-string>"
        )

    arg = sys.argv[1].strip()
    if arg.startswith("{"):
        payload = json.loads(arg)
    else:
        payload = json.loads(Path(arg).expanduser().resolve().read_text())
    result = handle_request(payload)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _main()
