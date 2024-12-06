# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import dataclasses

from typing import Generic, Protocol, TypeVar

_root_type: "Type | None" = None


def _get_root_type() -> "Type":
    if _root_type is None:
        raise RuntimeError("Symtypes was not properly initialized")
    return _root_type


@dataclasses.dataclass(frozen=True, slots=True)
class Type:
    name: str
    base: "Type | None" = dataclasses.field(default_factory=_get_root_type, kw_only=True)
    py_repr: str | None = dataclasses.field(default=None, kw_only=True, compare=False)

    def parameters(self) -> tuple["Type", ...]:
        return ()

    def __or__(self, other: "Type") -> "Type":
        return Union[self, other]

    def __repr__(self) -> str:
        return f"{self.name}"

    def variants(self) -> list["Type"]:
        return [self]

    def __len__(self) -> int:
        return len(self.variants())

    def __contains__(self, other: "Type") -> bool:
        return other == self

    def replace(self, old_type: "Type", new_type: "Type") -> "Type":
        if old_type == self:
            return new_type
        return self

    def pyrepr(self) -> str:
        return self.py_repr or self.name


@dataclasses.dataclass(frozen=True, slots=True)
class GenericType(Type):
    _parameters: tuple[Type, ...]

    def __post_init__(self) -> None:
        for param in self.parameters():
            if not isinstance(param, Type):
                raise TypeError(f"{self.name} parameters must be of type Type")

    def parameters(self) -> tuple["Type", ...]:
        return self._parameters

    def __repr__(self) -> str:
        params = ", ".join(repr(param) for param in self.parameters())
        return f"{self.name}[{params}]"

    def pyrepr(self) -> str:
        params = ", ".join(param.pyrepr() for param in self.parameters())
        return f"{self.py_repr or self.name}[{params}]"

    def replace(self, old_type: "Type", new_type: "Type") -> "Type":
        if old_type == self:
            return new_type
        return dataclasses.replace(
            self,
            _parameters=tuple(param.replace(old_type, new_type) for param in self._parameters),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class _GenericUnionType(GenericType):
    def extend(self, parameter: Type) -> "_GenericUnionType":
        return Union[self, parameter]

    def variants(self) -> list["Type"]:
        return list(self.parameters())

    def __repr__(self) -> str:
        params = list(sorted(self.parameters(), key=lambda param: param.name))
        if len(params) == 2:
            if params[0] == NoneType:
                return f"Optional[{params[1]}]"
            elif params[1] == NoneType:
                return f"Optional[{params[0]}]"

        param_list = ", ".join(repr(param) for param in params)
        return f"{self.name}[{param_list}]"

    def pyrepr(self) -> str:
        params = list(sorted(self.parameters(), key=lambda param: param.pyrepr()))
        if len(params) == 2:
            if params[0] == NoneType:
                return f"Optional[{params[1].pyrepr()}]"
            elif params[1] == NoneType:
                return f"Optional[{params[0].pyrepr()}]"

        param_list = ", ".join(param.pyrepr() for param in params)
        return f"{self.py_repr or self.name}[{param_list}]"

    def __contains__(self, other: Type) -> bool:
        if isinstance(other, _GenericUnionType):
            return all(param in self for param in other.parameters())
        return other in self._parameters

    def replace(self, old_type: Type, new_type: Type) -> Type:
        if old_type not in self:
            return self
        new_type = (
            Union[tuple(param for param in self._parameters if param not in old_type)] | new_type
        )
        if len(new_type) == 1:
            return new_type.parameters()[0]
        return new_type

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _GenericUnionType):
            return NotImplemented
        if self.name != other.name:
            return False
        for param in other.parameters():
            if param not in self:
                return False
        return True


@dataclasses.dataclass(frozen=True, slots=True)
class _GenericTupleType(GenericType):
    def __repr__(self) -> str:
        params = ", ".join(repr(param) for param in self.parameters())
        return f"{self.name}[{params}]"


_G = TypeVar("_G", bound=GenericType, covariant=True)


class _GenericTypeFactory(Protocol[_G]):
    def __call__(self, *args: Type) -> _G: ...
    @property
    def __name__(self) -> str: ...


class UnboundGenericType(Generic[_G]):
    def __init__(self, type_factory: _GenericTypeFactory[_G]) -> None:
        self._type_factory = type_factory
        self._name = type_factory.__name__

    def __getitem__(self, parameters: Type | tuple[Type, ...]) -> _G:
        parameters = parameters if isinstance(parameters, tuple) else (parameters,)
        return self._type_factory(*parameters)

    def create(self, *parameters: Type) -> _G:
        return self[parameters]


@UnboundGenericType
def Union(*parameters: Type) -> _GenericUnionType:
    params = set([variant for param in parameters for variant in param.variants()])
    return _GenericUnionType("Union", tuple(params))


@UnboundGenericType
def Optional(*parameters: Type) -> _GenericUnionType:
    if len(parameters) != 1:
        raise ValueError("Optional takes exactly one type parameter")
    return Union[parameters[0], NoneType]


def unwrap_optional(parameter: Type) -> Type:
    if not (isinstance(parameter, _GenericUnionType) and parameter.name == "Union"):
        return parameter
    if NoneType not in parameter:
        return parameter
    params = tuple(param for param in parameter.parameters() if param != NoneType)
    if len(params) == 1:
        return params[0]
    return _GenericUnionType("Union", params)


@UnboundGenericType
def Tuple(*parameters: Type) -> _GenericTupleType:
    return _GenericTupleType("Tuple", parameters, py_repr="tuple")


@UnboundGenericType
def List(*parameters: Type) -> GenericType:
    if len(parameters) != 1:
        raise ValueError("List takes exactly one type parameter")
    return GenericType("List", parameters, py_repr="list")


Object = _root_type = Type("Object", base=None, py_repr="object")
NoneType = Type("NoneType", py_repr="None")
Tensor = Type("Tensor", py_repr="torch.Tensor")
QuantizedTensor = Type("QuantizedTensor")
Int = Type("Int", py_repr="int")
Float = Type("Float", py_repr="float")
String = Type("String", py_repr="str")
Bool = Type("Bool", py_repr="bool")
Number = Int | Float
MaybeQuantized = Tensor | QuantizedTensor
EllipsisType = Type("EllipsisType", py_repr="...")
Size = Type("torch.Size") | Tuple[Int, EllipsisType]
Quantizer = Type("Quantizer")


def str_to_type(name: str) -> Type:
    match name:
        case "object":
            return Object
        case "None":
            return NoneType
        case "Tensor" | "torch.Tensor":
            return Tensor
        case "QuantizedTensor" | "Quantized":
            return QuantizedTensor
        case "MaybeQuantized":
            return MaybeQuantized
        case "int" | "Int":
            return Int
        case "float" | "Float":
            return Float
        case "bool" | "Bool":
            return Bool
        case "str" | "String" | "string":
            return String
        case "number" | "Number":
            return Number
        case "size" | "Size":
            return Size
        case "..." | "Ellipsis":
            return EllipsisType
    raise ValueError(f"No known type with name '{name}'")


def type_to_python_str(symtype: Type) -> str:
    return symtype.pyrepr()
