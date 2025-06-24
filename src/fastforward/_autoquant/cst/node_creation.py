# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import libcst
import libcst.helpers

from fastforward._autoquant.cst import nodes
from fastforward._quantops import OperatorTable
from fastforward._quantops.operator import Operator


def get_keyword_argument_node(keyword: str, expression: libcst.BaseExpression) -> libcst.Arg:
    """Constructs a keyword argument node for `keyword` and `expression`.

    Args:
        keyword: The kwargs keyword.
        expression: The value for the argument.

    Returns:
        A keyword argument node for `keyword`.
    """
    dummy_arg = libcst.helpers.parse_template_expression(
        "dummy_fn({keyword}={expression})", keyword=libcst.Name(keyword), expression=expression
    )
    assert isinstance(dummy_arg, libcst.Call)
    quantizer_args = dummy_arg.args[-1]
    return quantizer_args


def get_quantized_function_counterpart(
    optable: OperatorTable, func_name: str
) -> tuple[libcst.Attribute, Operator]:
    """Replaces the given function name with its quantized counterpart.

    Args:
        optable: The operator table to use for replacement.
        func_name: The name of the function to replace.

    Returns:
        A tuple consisting of
        - the node of the quantized function,
        - the operator from the optable for further analysis.
    """
    operator = optable.get(func_name)
    replace_name = operator.dispatch_qualified_name()
    assert replace_name is not None
    func = libcst.parse_expression(replace_name)
    assert isinstance(func, libcst.Attribute)
    return func, operator


def _create_quantize_statement(
    name: str, quantizer_name: nodes.QuantizerReference
) -> libcst.SimpleStatementLine:
    """Create a quantize statement.

    Args:
        name: The name of the variable to be quantized.
        quantizer_name: The name of the quantizer.

    Returns:
        The quantize statement.
    """
    name_node = libcst.Name(name)
    quantize_statement = libcst.helpers.parse_template_statement(
        "{name} = self.{quantizer_name}({name})",
        name=name_node,
        quantizer_name=quantizer_name,
    )
    assert isinstance(quantize_statement, libcst.SimpleStatementLine)
    return quantize_statement
