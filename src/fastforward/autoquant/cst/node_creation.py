# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import libcst

from fastforward._quantops import OperatorTable
from fastforward._quantops.operator import Operator


def get_output_quantizer_kwarg(quantizer_var_name: str) -> libcst.Arg:
    """Constructs a keyword argument node for a quantizer with name `quantizer_var_name`.

    Args:
        quantizer_var_name: The name of the quantizer.

    Returns:
        A keyword argument node for the output quantizer.
    """
    dummy_arg = libcst.parse_expression(
        f"dummy_fn(dummy_var, output_quantizer=self.{quantizer_var_name})"
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
