# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import TypeVar

_T = TypeVar("_T")


def ensure_type(node: object, NodeType: type[_T], ExceptionType: type[Exception] = TypeError) -> _T:
    """
    Assert that `node` is an instance of `NodeType`.

    If assertion fails, raise `ExceptionType`.

    This function is similar to `libcst.ensure_type` but does not raise
    `Exception` by default.

    Args:
        node: The node to check the type of.
        NodeType: The expected node type of `node`.
        Expcetion: The exception to raise if `node` is not an instance of
            `NodeType`.
    """
    if not isinstance(node, NodeType):
        raise ExceptionType(f"Expected {NodeType.__name__} but found {type(node).__name__}")
    return node
