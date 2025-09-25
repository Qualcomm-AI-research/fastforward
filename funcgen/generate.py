# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

#
# Script to generate quantized operators
#
# usage: generate.py [-h] source destination
#
# positional arguments:
#   source       Operator source file
#   destination  Destination file, will be overwritten if already exists
#

import argparse
import importlib.resources
import pathlib
import subprocess

import fastforward._quantops

from fastforward._quantops import OperatorTable
from fastforward._quantops import generate as gen_operators


def _get_default_src() -> pathlib.Path:
    """Returns a default path to yaml file with quantized_operators."""
    res_path = importlib.resources.files(fastforward._quantops) / "quantized_operators.yaml"
    with importlib.resources.as_file(res_path) as src_path:
        return src_path


def _get_default_dst() -> pathlib.Path:
    """Returns a default path where store generated files."""
    script_path = pathlib.Path(__file__).parent
    return (script_path / ".." / "src" / "fastforward" / "_gen").resolve()


def main() -> None:
    """Entry point for code generation for FastForward."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=pathlib.Path,
        default=_get_default_src(),
        help="Operator source file (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        required=False,
        default=_get_default_dst(),
        help="Output directory (default: %(default)s), will be overwritten if already exists",
    )

    args = parser.parse_args()
    if not (args.input.is_file() and args.input.exists()):
        msg = f"The input ({args.input}) should be an existed file"
        raise RuntimeError(msg)

    args.output.mkdir(parents=True, exist_ok=True)
    if not args.output.is_dir():
        msg = f"The output ({args.output}) should be a directory"
        raise RuntimeError(msg)

    operators = OperatorTable.from_yaml(args.input, _resolve_dispatch=False)
    gen_operators.generate(operators, source=args.input, destination=args.output)
    subprocess.run(["ruff", "check", "--fix", args.output])
    subprocess.run(["ruff", "format", args.output])


if __name__ == "__main__":
    main()
