from typing import TypeVar

from .exceptions import CFGError

_T = TypeVar("_T")


def ensure_type(node: object, NodeType: type[_T], ExceptionType: type[Exception] = CFGError) -> _T:
    if not isinstance(node, NodeType):
        raise ExceptionType(f"Expected {NodeType.__name__} but found {type(node).__name__}")
    return node
