# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import libcst


def generated_forward(generated_code: str) -> str:
    """Extract the forward method source from autoquant-generated code.

    The full generated output contains helper functions (e.g.
    ``quantized__in_projection_packed``) that embed ``_tmp_NNN`` variable names
    and verbatim comments from called library functions — both torch-version
    sensitive.  The ``forward`` method of the generated ``Quantized*`` class
    contains only the user's own code flow and is stable across torch versions.
    """
    cst = libcst.parse_module(generated_code)
    for node in cst.body:
        if isinstance(node, libcst.ClassDef):
            for item in node.body.body:
                if isinstance(item, libcst.FunctionDef) and item.name.value == "forward":
                    return cst.code_for_node(item)
    return ""
