# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


import libcst

from .transformer import QuantizerFunctionTransformer


def introduce_input_quantizers(funcdef: libcst.FunctionDef) -> libcst.FunctionDef:
    wrapper = libcst.MetadataWrapper(libcst.Module([funcdef]))
    updated_funcdef = wrapper.visit(QuantizerFunctionTransformer()).body[0]

    assert isinstance(updated_funcdef, libcst.FunctionDef)
    return updated_funcdef
