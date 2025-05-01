# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""
!!! experimental
    Please be aware that autoquant is an experimental feature. Use it with caution and
    expect changes as we continue the development of the feature.

    We encourage you to report any issues or feature requests.
"""  # noqa: D205, D212

import pathlib

from typing import overload

import torch

from fastforward._autoquant import pybuilder
from fastforward._autoquant.autoquant import (
    AutoQuantizedCode,
    autoquant_with_defaults,
    codeformat_with_defaults,
    emit_code_of_module,
)
from fastforward._autoquant.pybuilder.importing import import_code
from fastforward._quantops import optable


@overload
def autoquantize(
    module: torch.nn.Module,
    *,
    operator_table: optable.OperatorTable | None = None,
    code_formatter: pybuilder.CodeFormatter | None = None,
    output_path: pathlib.Path | str | None = None,
    force_overwrite: bool = False,
    auto_import: bool = True,
) -> AutoQuantizedCode: ...


@overload
def autoquantize(
    module: torch.nn.Module,
    *,
    operator_table: optable.OperatorTable | None = None,
    code_formatter: pybuilder.CodeFormatter | None = None,
    code_writer: pybuilder.BasicCodeWriter | None = None,
    auto_import: bool = True,
) -> AutoQuantizedCode: ...


def autoquantize(
    module: torch.nn.Module,
    *,
    operator_table: optable.OperatorTable | None = None,
    code_formatter: pybuilder.CodeFormatter | None = None,
    output_path: pathlib.Path | str | None = None,
    force_overwrite: bool = False,
    code_writer: pybuilder.BasicCodeWriter | None = None,
    auto_import: bool = True,
) -> AutoQuantizedCode:
    """Create Python source code for quantized version of `module`.

    Args:
        module: The module to quantize.
        operator_table: The operator table that defines the non-quantized to
            quantized operator mapping.
        code_formatter: The code formatter to use for formatting the generated
            code. If not provided, the default formatter is used.
        output_path: The path to write the generated code to. Mutually exclusive with code_writer.
        force_overwrite: If True, overwrite the output file if it already exists.
        code_writer: The code writer to use for writing the generated code. Mutually exclusive with
            output_path.
        auto_import: If True, automatically import the written module.
    """
    autoquant_code = autoquant_with_defaults(module, operator_table)
    formatted_code = codeformat_with_defaults(autoquant_code, code_formatter=code_formatter)
    pymodule_name = emit_code_of_module(
        formatted_code,
        output_path=output_path,
        force_overwrite=force_overwrite,
        code_writer=code_writer,
    )

    if auto_import:
        pymodule = import_code(code=formatted_code, pymodule_name=pymodule_name)
    else:
        pymodule = None
    return AutoQuantizedCode(
        code=formatted_code,
        pymodule=pymodule,
        pymodule_name=pymodule_name,
    )
