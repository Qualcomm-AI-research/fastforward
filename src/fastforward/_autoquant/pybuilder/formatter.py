# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Formatters are utility classes to format code."""

import dataclasses
import subprocess

from typing import Protocol, Sequence

from typing_extensions import override


class CodeFormatter(Protocol):
    """Defines the required public signature of a CodeFormatter."""

    def format(self, code: str) -> str:
        """Formats the code."""
        raise NotImplementedError


@dataclasses.dataclass
class SubprocessCodeFormatter(CodeFormatter):
    """Formats code via by invoking subprocesses."""

    commands: Sequence[Sequence[str]] = ()

    @override
    def format(self, code: str) -> str:
        """Formats the code via a subprocess."""
        for command in self.commands:
            cp = subprocess.run(
                command,
                input=code,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8",
            )
            if cp.returncode != 0:
                raise RuntimeError(
                    "Code formatting failed (cf. command below). This is an unexpected error, "
                    + "please report it to the FF development team.\n"
                    + f"Process details:\n{cp}"
                )
            code = cp.stdout
        return code


class RuffFormatter(SubprocessCodeFormatter):
    """Formats code with `ruff`."""

    def __init__(self) -> None:
        super().__init__(
            commands=(
                ("ruff", "format", "-"),
                (
                    "ruff",
                    "--config",
                    'lint.isort.known-third-party = ["fastforward"]',
                    "check",
                    "--fix",
                    "--select",
                    "I001",
                    "-",
                ),  # isort
            )
        )
