# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

#
# Parser for operator schema
#

from typing import Any, TypeAlias, cast

from . import operator, symtypes

# Re-export error types
from ._parser import ParseError as ParseError
from ._parser import Parser, ParseRule, Sentinel, as_list, tokenizer
from ._parser import TokenizationError as TokenizationError


def _operator(
    identifier: str, params: list[operator.Parameter], return_type: symtypes.Type | Sentinel
) -> operator.Operator:
    """Helper function to turn parse result into operator
    """
    return_type_ = None
    if isinstance(return_type, symtypes.Type):
        return_type_ = return_type
    return operator.Operator(identifier, tuple(params), return_type_)


def _param(identifier: str, type_: symtypes.Type, default: str | Sentinel) -> operator.Parameter:
    """Helper function to turn parse result into parameter
    """
    default_value = None
    if isinstance(default, str):
        default_value = default
    return operator.Parameter(type_, identifier, default_value)


def _type(*args: Any) -> symtypes.Type | None:
    """Helper function to turn parse result into symbolic type
    """
    match args:
        case (str(),):
            try:
                return symtypes.str_to_type(args[0])
            except ValueError:
                return None
        case (symtypes.Type(),):
            return cast(symtypes.Type, args[0])
        case ("Union" | "union", sub_type):
            return symtypes.Union[tuple(sub_type)]
        case ("Optional" | "optional", sub_type):
            return symtypes.Optional[tuple(sub_type)]
        case ("Tuple" | "tuple", sub_type):
            return symtypes.Tuple[tuple(sub_type)]
        case ("List" | "list", sub_type):
            return symtypes.List[tuple(sub_type)]
        case ("Sequence", sub_type):
            return symtypes.GenericType("List", tuple(sub_type), py_repr="Sequence")
        case (base_type, sub_type):
            return symtypes.GenericType(base_type, tuple(sub_type))
        case _:
            return None


_R: TypeAlias = ParseRule


class _SpecParser(Parser):
    """Simple PEG based parser for quantizer operator spec.
    """

    rules = (
        _R("start", ["operator"]),
        #
        _R("operator", ["ident !LEFTPAREN param_list !RIGHTPAREN [return_value]"], _operator),
        #
        _R("ident", ["IDENTIFIER"], lambda r: r.source),
        _R("digit", ["DIGIT"], lambda r: r.source),
        _R("expr", ["ident", "digit"]),
        #
        _R("type", ["type_union", "ident !LEFTBRACKET sub_types !RIGHTBRACKET", "ident"], _type),
        _R("type_union", ["type_union !PIPE type", "type !PIPE type"], symtypes.Union.create),
        _R("type_list", ["type !COMMA type_list", "type_list", "type"], as_list),
        _R("sub_types", ["type_list ellipsis_trailer", "type_list"], as_list),
        _R("ellipsis_trailer", ["COMMA PERIOD PERIOD PERIOD"], lambda *_: [symtypes.EllipsisType]),
        #
        _R("param", ["ident !COLON type [default_value]"], _param),
        _R("default_value", ["!EQUALS expr"]),
        _R("param_list", ["param !COMMA param_list", "param"], as_list),
        #
        _R("return_value", ["!ARROW type"]),
    )


def parse_schema(schema_str: str) -> operator.Operator:
    """Parse schema_str of a quantized operator and create corresponding OpSpec
    object.
    """
    return _SpecParser(tokenizer(schema_str)).parse()  # type: ignore[no-any-return]
