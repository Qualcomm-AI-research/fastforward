# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import libcst

from fastforward._quantops import OperatorTable
from fastforward._quantops.operator import Operator


def get_next_output_quantizer_kw_arg_and_name(
    func_name: str, current_num_quantized_vars: int
) -> tuple[libcst.Arg, str]:
    """Constructs a keyword argument node for an output quantizer related to the function name.

    For example, for `func_name=torch.sigmoid` and `current_num_quantized_vars=5`, the
    returned value is a `libcst.Arg`-node of keyword type which generates the code
    `output_quantizer=self.quantizer_sigmoid_6`.

    Args:
    func_name: The name of the function for which to generate the output quantizer.
    current_num_quantized_vars: The current number of quantized variables.

    Returns:
        A tuple containing
        - a keyword argument node for the output quantizer,
        - the name of the new quantizer variable.
    """
    quantized_var_name = f"quantizer_{func_name.split('.')[-1]}_{current_num_quantized_vars + 1}"
    dummy_arg = libcst.parse_expression(
        f"dummy_fn(dummy_var, output_quantizer=self.{quantized_var_name})"
    )
    assert isinstance(dummy_arg, libcst.Call)
    quantizer_args = dummy_arg.args[-1]
    return quantizer_args, quantized_var_name


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
