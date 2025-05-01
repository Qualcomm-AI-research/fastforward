# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Formatters are utility classes to format code."""

import subprocess

from typing import Protocol, Sequence

from typing_extensions import override


class CodeFormatter(Protocol):
    """Defines the required public signature of a CodeFormatter."""

    def format(self, code: str) -> str:
        """Formats the code."""
        raise NotImplementedError


class SubprocessCodeFormatter(CodeFormatter):
    """Formats code via by invoking subprocesses."""

    command: Sequence[str] = ("false",)

    @override
    def format(self, code: str) -> str:
        """Formats the code via a subprocess."""
        cp = subprocess.run(
            self.command,
            input=code,
            check=True,
            stdout=subprocess.PIPE,
            encoding="utf-8",
        )
        if cp.returncode != 0:
            raise RuntimeError(
                "Code formatting failed (cf. command below). This is an unexpected error, please report it to the FF development team."
                + f"\nProcess details:\n{cp}"
            )
        return cp.stdout


class RuffFormatter(SubprocessCodeFormatter):
    """Formats code with `ruff`."""

    command = ("ruff", "format", "-")
