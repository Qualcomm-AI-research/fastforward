# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import libcst
import torch

import fastforward._autoquant.autoquant as autoquant
import fastforward._autoquant.cst.filter as node_filter
import fastforward._autoquant.pybuilder as pybuilder

from fastforward._autoquant.convert import autoquantize_funcdef
from fastforward._autoquant.function_context import FunctionContext
from fastforward._autoquant.pass_manager import PassManager
from fastforward._autoquant.pybuilder.quantizer_collection import QuantizerReferenceCollection

from .string import assert_strings_match_verbose, dedent_strip


def autoquantize_str(
    src: str, *, as_module: bool = False, as_cst: bool = False, use_type_inference: bool = False
) -> str | libcst.FunctionDef | libcst.ClassDef:
    """Autoquantizes Python function string and returns the quantized code as a string.

    Args:
        src : The input Python function as a string.
        as_module: If True, returns the quantized code as a module. Defaults to False.
        as_cst: If True, return a CST instead of a string.
        use_type_inference: If True, use type inference to reduce false
            positive function rewrites in autoquant at the cost of a longer
            runtime.


    Returns:
        The quantized Python function or module as a string or CST.

    Notes:
        This function assumes the provided function is a method of module and will
        introduce function calls to members on `self`. This function will raise `ValueError`
        if `src` is not a function that has `self` as first parameter.
    """
    src_cst = libcst.parse_module(dedent_strip(src)[0])
    dst_cst = (
        _autoquantize_str_to_classdef(src_cst, use_type_inference=use_type_inference)
        if as_module
        else _autoquantize_str_to_funcdef(src_cst, use_type_inference=use_type_inference)
    )

    if as_cst:
        return dst_cst
    else:
        dst_str = libcst.Module([dst_cst]).code
        return autoquant.codeformat_with_defaults(dst_str)


class _PlaceholderModule(torch.nn.Module):
    def forward(self) -> None:
        return None


def _autoquantize_str_to_classdef(
    src_cst: libcst.Module, use_type_inference: bool
) -> libcst.ClassDef:
    funcdef = _retrieve_funcdef(src_cst)
    if len(funcdef.params.params) == 0 or funcdef.params.params[0].name.value != "self":
        msg = "Expected `src` to be a function with `self` as first parameter"
        raise ValueError(msg)

    pm = PassManager(autoquant.default_preprocessing_passes(use_type_inferece=use_type_inference))
    src_cst = pm(src_cst)

    funcdef = _retrieve_funcdef(src_cst)

    # Setup context and quantizer_ref state such that autoquant thinks it is
    # quantizing an instance method.
    quantizer_refs = QuantizerReferenceCollection()
    placeholder_mod = _PlaceholderModule
    func_ctx = FunctionContext.from_method(placeholder_mod, "forward")
    with quantizer_refs.push_context(func_ctx):
        converted_cst = autoquantize_funcdef(
            funcdef,
            autoquant.default_optable(),
            func_ctx=func_ctx,
            quantizer_refs=quantizer_refs,
        )

    dst_class = pybuilder.QuantizedModuleBuilder(
        "QuantizedTestModule",
        bases=(),
        required_imports=(),
        origin=placeholder_mod,
    )
    dst_class.add_method(
        pybuilder.QuantizedFunctionBuilder(
            converted_cst,
            required_imports=(),
            origin=func_ctx,
        )
    )

    return dst_class.build(quantizer_refs)


def _autoquantize_str_to_funcdef(
    src_cst: libcst.Module, use_type_inference: bool
) -> libcst.FunctionDef:
    original_funcdef = _retrieve_funcdef(src_cst)
    func_name = original_funcdef.name.value
    dst_cst = _autoquantize_str_to_classdef(src_cst, use_type_inference=use_type_inference)
    funcdef = _retrieve_funcdef(dst_cst, needle=func_name)
    return funcdef


def _retrieve_funcdef(
    cst: libcst.Module | libcst.ClassDef, needle: str | None = None
) -> libcst.FunctionDef:
    funcdefs = list(node_filter.filter_nodes_by_type(cst, libcst.FunctionDef))
    if needle is None:
        if len(funcdefs) == 0:
            msg = "Module does not contain a function"
            raise ValueError(msg)
        if len(funcdefs) > 1:
            msg = "Module contains multiple functions. Only 1 is expected"
            raise ValueError(msg)
        return funcdefs[0]
    else:
        for funcdef in funcdefs:
            if funcdef.name.value == needle:
                return funcdef
        else:
            msg = f"CST does not contain a function with name '{needle}'"
            raise RuntimeError(msg)


def assert_autoquantize_result(
    input: str, expected: str, as_module: bool = False, use_type_inference: bool = False
) -> None:
    """Asserts that the autoquantization of the given input string matches the expected output.

    Args:
        input: The input string to be autoquantized.
        expected: The expected output string after autoquantization.
        as_module: Whether to autoquantize as a module. Defaults to False.
        use_type_inference: If True, use type inference to reduce false
            positive function rewrites in autoquant at the cost of a longer
            runtime.
    """
    actual = autoquantize_str(
        input, as_module=as_module, as_cst=False, use_type_inference=use_type_inference
    )
    assert isinstance(actual, str)
    formatted_expected = autoquant.codeformat_with_defaults(dedent_strip(expected)[0])

    assert_strings_match_verbose(actual, formatted_expected)
