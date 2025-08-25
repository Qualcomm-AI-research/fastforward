# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import libcst
import libcst.helpers

from fastforward._quantops import OperatorTable
from fastforward._quantops.operator import Operator


def get_keyword_argument_node(
    keyword: str | libcst.BaseExpression, expression: libcst.BaseExpression
) -> libcst.Arg:
    """Constructs a keyword argument node for `keyword` and `expression`.

    Args:
        keyword: The kwargs keyword.
        expression: The value for the argument.

    Returns:
        A keyword argument node for `keyword`.
    """
    if isinstance(keyword, str):
        keyword = libcst.Name(keyword)

    dummy_arg = libcst.helpers.parse_template_expression(
        "dummy_fn({keyword}={expression})", keyword=keyword, expression=expression
    )
    assert isinstance(dummy_arg, libcst.Call)
    quantizer_args = dummy_arg.args[-1]
    return quantizer_args


def get_parameter_node(
    name: str | libcst.Name, annotation: libcst.BaseExpression | str | None
) -> libcst.Param:
    """Constructs a parameter node for `keyword` and `annotation`.

    Args:
        name: The parameter name.
        annotation: The parameter annotation.

    Returns:
        A keyword argument node for `keyword`.
    """
    if isinstance(name, str):
        name = libcst.Name(name)
    if isinstance(annotation, str):
        annotation = libcst.parse_expression(annotation)

    if annotation is None:
        funcdef = libcst.helpers.parse_template_statement("def f({name}): ...", name=name)
    else:
        funcdef = libcst.helpers.parse_template_statement(
            "def f({name}: {annotation}): ...", name=name, annotation=annotation
        )

    assert isinstance(funcdef, libcst.FunctionDef)
    return funcdef.params.params[0]


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


def create_quantize_statement(
    *,
    target: str | libcst.Name,
    quantizer_ref: libcst.BaseExpression,
    source: str | libcst.BaseExpression | None = None,
) -> libcst.SimpleStatementLine:
    """Create a quantize statement.

    Args:
        target: Variable name to assign the quantized result to
        quantizer_ref: Expression that evaluates to the quantizer function
        source: Expression to quantize (defaults to target)

    Returns:
        The quantize statement.
    """
    if not isinstance(target, libcst.Name):
        target = libcst.Name(target)
    if source is None:
        source = target
    if isinstance(source, str):
        source = libcst.Name(source)

    quantize_statement = libcst.helpers.parse_template_statement(
        "{target} = {quantizer_ref}({source})",
        target=target,
        quantizer_ref=quantizer_ref,
        source=source,
    )
    assert isinstance(quantize_statement, libcst.SimpleStatementLine)
    return quantize_statement
