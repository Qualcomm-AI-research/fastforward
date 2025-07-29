# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import libcst

import fastforward._autoquant.autoquant as autoquant
import fastforward._autoquant.cst.filter as node_filter
import fastforward._autoquant.pybuilder as pybuilder

from fastforward._autoquant.convert import autoquantize_funcdef

from .string import assert_strings_match_verbose, dedent_strip


def autoquantize_str(
    src: str, *, as_module: bool = False, as_cst: bool = False
) -> str | libcst.FunctionDef | libcst.ClassDef:
    """Autoquantizes Python function string and returns the quantized code as a string.

    Args:
        src : The input Python function as a string.
        as_module: If True, returns the quantized code as a module. Defaults to False.
        as_cst: If True, return a CST instead of a string.

    Returns:
        The quantized Python function or module as a string or CST.

    Notes:
        This function assumes the provided function is a method of module and will
        introduce function calls to members on `self`. This function will raise `ValueError`
        if `src` is not a function that has `self` as first parameter.
    """
    src_cst = libcst.parse_statement(dedent_strip(src)[0])
    assert isinstance(src_cst, libcst.FunctionDef), "Expected function statement"
    dst_cst = (
        _autoquantize_str_to_classdef(src_cst)
        if as_module
        else _autoquantize_str_to_funcdef(src_cst)
    )

    if as_cst:
        return dst_cst
    else:
        dst_str = libcst.Module([dst_cst]).code
        return autoquant.codeformat_with_defaults(dst_str)


def _autoquantize_str_to_classdef(src_cst: libcst.FunctionDef) -> libcst.ClassDef:
    if len(src_cst.params.params) == 0 or src_cst.params.params[0].name.value != "self":
        msg = "Expected `src` to be a function with `self` as first parameter"
        raise ValueError(msg)

    for proc_pass in autoquant.default_preprocessing_passes():
        src_cst = src_cst.visit(proc_pass)  # type: ignore[assignment]
        assert isinstance(src_cst, libcst.FunctionDef), "Processing pass did not return FuncDef"

    converted_cst = autoquantize_funcdef(src_cst, autoquant.default_optable())

    dst_class = pybuilder.QuantizedModuleBuilder(
        "QuantizedTestModule", bases=(), required_imports=()
    )
    dst_class.add_method(pybuilder.QuantizedFunctionBuilder(converted_cst, required_imports=()))

    return dst_class.build()


def _autoquantize_str_to_funcdef(src_cst: libcst.FunctionDef) -> libcst.FunctionDef:
    func_name = src_cst.name.value
    dst_cst = _autoquantize_str_to_classdef(src_cst)
    for funcdef in node_filter.filter_nodes_by_type(dst_cst, libcst.FunctionDef):
        if funcdef.name.value == func_name:
            return funcdef
    else:
        raise RuntimeError("Converted CST does not contain original function")


def assert_autoquantize_result(input: str, expected: str, as_module: bool = False) -> None:
    """Asserts that the autoquantization of the given input string matches the expected output.

    Args:
        input: The input string to be autoquantized.
        expected: The expected output string after autoquantization.
        as_module: Whether to autoquantize as a module. Defaults to False.
    """
    actual = autoquantize_str(input, as_module=as_module, as_cst=False)
    assert isinstance(actual, str)
    formatted_expected = autoquant.codeformat_with_defaults(dedent_strip(expected)[0])

    assert_strings_match_verbose(actual, formatted_expected)
