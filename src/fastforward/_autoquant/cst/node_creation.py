# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Any, Callable, Sequence

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
    optable: OperatorTable, func_key: Callable[..., Any] | str, args: Sequence[libcst.Arg]
) -> tuple[libcst.Attribute, Operator]:
    """Replaces the given function name with its quantized counterpart.

    Args:
        optable: The operator table to use for replacement.
        func_key: The name of the function or reference to a function to replace.
        args: The arguments passed to the original function

    Returns:
        A tuple consisting of
        - the node of the quantized function,
        - the operator from the optable for further analysis.
    """
    pos_args, kw_args = _args_to_pos_and_kw_args(args)
    for operator in optable.get(func_key):
        if not operator.validate_arguments(pos_args, kw_args):
            continue

        replace_name = operator.dispatch_qualified_name()
        assert replace_name is not None
        func = libcst.parse_expression(replace_name)
        assert isinstance(func, libcst.Attribute)
        return func, operator

    msg = (
        "Optable does not contain an operator that can replace an operator identified "
        + f"by {func_key} and provided args."
    )
    raise KeyError(msg)


def _args_to_pos_and_kw_args(
    args: Sequence[libcst.Arg],
) -> tuple[tuple[libcst.Arg, ...], dict[str, libcst.Arg]]:
    """Convert a list of cst `Arg` to a tuple of positional and dict of keyword arguments."""
    pos_args: list[libcst.Arg] = []
    kw_args: dict[str, libcst.Arg] = {}
    for arg in args:
        if arg.keyword is not None:
            kw_args[arg.keyword.value] = arg
        else:
            pos_args.append(arg)

    return tuple(pos_args), kw_args


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
