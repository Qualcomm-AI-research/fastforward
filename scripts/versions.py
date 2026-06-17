#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Single source of truth for the (python, torch, cuda) versions used in CI and dev.

The first entry in ``VERSIONS`` is the default used for local docker builds, the
docs build, and the verify/docker-dev jobs that only run on a single image.

Subcommands:
  matrix    Write the full matrix as ``matrix=<json>`` to ``$GITHUB_OUTPUT``.
  default   Print one or all default version fields (for shell consumers).
"""

import argparse
import json
import os
import sys

VERSIONS = [
    {"VER_TORCH": "2.12.0", "VER_CUDA": "13.0", "VER_PYTHON": "3.12"},
    {"VER_TORCH": "2.11.0", "VER_CUDA": "12.6", "VER_PYTHON": "3.12"},
    {"VER_TORCH": "2.10.0", "VER_CUDA": "12.6", "VER_PYTHON": "3.12"},
    # {"VER_TORCH": "2.9.0", "VER_CUDA": "12.6", "VER_PYTHON": "3.14"},  # currently broken, cf. #451
    {"VER_TORCH": "2.8.0", "VER_CUDA": "12.6", "VER_PYTHON": "3.12"},
    {"VER_TORCH": "2.6.0", "VER_CUDA": "12.4", "VER_PYTHON": "3.10"},
    {"VER_TORCH": "2.4.1", "VER_CUDA": "12.1", "VER_PYTHON": "3.10"},
]

_FIELD_ALIASES = {
    "python": "VER_PYTHON",
    "torch": "VER_TORCH",
    "cuda": "VER_CUDA",
}


def _cmd_matrix() -> None:
    docker_registry = os.environ["DOCKER_REGISTRY"]
    docker_image = os.environ["DOCKER_IMAGE"]
    docker_tag = os.environ["IMAGE_TAG"]

    matrix = []
    for entry in VERSIONS:
        python = entry["VER_PYTHON"]
        torch = entry["VER_TORCH"]
        cuda = entry["VER_CUDA"]
        image_name = f"{docker_registry}/{docker_image}-py{python}-pt{torch}-cu{cuda}"
        matrix.append({**entry, "IMAGE_NAME": image_name, "IMAGE_TAG": docker_tag})

    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as handle:
        handle.write(f"matrix={json.dumps(matrix)}\n")


def _cmd_default(field: str | None) -> None:
    default = VERSIONS[0]
    if field is None:
        print(f"{default['VER_PYTHON']} {default['VER_TORCH']} {default['VER_CUDA']}")
        return
    print(default[_FIELD_ALIASES[field]])


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the requested subcommand."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("matrix", help="write CI matrix to $GITHUB_OUTPUT")

    default_parser = sub.add_parser("default", help="print default versions")
    default_parser.add_argument(
        "--field",
        choices=sorted(_FIELD_ALIASES),
        help="print only this field (python, torch, or cuda)",
    )

    args = parser.parse_args(argv)
    if args.command == "matrix":
        _cmd_matrix()
    elif args.command == "default":
        _cmd_default(args.field)
    return 0


if __name__ == "__main__":
    sys.exit(main())
