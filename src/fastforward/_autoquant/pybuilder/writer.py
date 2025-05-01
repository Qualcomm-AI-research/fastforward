# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Writers are utility classes to write code."""

import logging
import pathlib
import sys

from typing import Any, Protocol, TextIO

from typing_extensions import override

logger = logging.getLogger(__name__)


class CodeWriterP(Protocol):
    def write(self, code: str) -> None: ...


class BasicCodeWriter(CodeWriterP):
    """Defines basic signature of a CodeWriter."""

    def __init__(self, module_name: str) -> None:
        self.module_name = module_name.removesuffix(".py")


class TextIOWriter(BasicCodeWriter):
    def __init__(self, *args: Any, writer: TextIO, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._writer = writer

    @override
    def write(self, code: str) -> None:
        """Writes the code to a `TextIO` instance."""
        self._writer.write(code)


class StdoutWriter(TextIOWriter):
    """A CodeWriter that writes to stdout."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, writer=sys.stdout, **kwargs)


class FileWriter(BasicCodeWriter):
    """A CodeWriter that writes to a file."""

    def __init__(
        self,
        output_path: pathlib.Path,
        force_overwrite: bool = False,
    ) -> None:
        if output_path.suffix == ".py":
            self.output_dir = output_path.parent
            module_name = output_path.with_suffix("").name
            self.output_file = output_path
        else:
            self.output_dir = output_path
            module_name = output_path.parts[-1]
            self.output_file = output_path / "__init__.py"

        self._force_overwrite = force_overwrite
        super().__init__(module_name=module_name)

    def write(self, code: str) -> None:
        """Writes the code to a file."""
        self.output_dir.mkdir(exist_ok=True, parents=True)
        if (outfile := self.output_file).exists() and not self._force_overwrite:
            raise FileExistsError(
                f"File {outfile} already exists. Use `force_overwrite=True` to ignore."
            )

        outfile.write_text(code, encoding="utf-8")
