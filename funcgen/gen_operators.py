# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import os
import pathlib
import textwrap

from typing import Any, Optional, Protocol, Sequence, cast

import libcst
import libcst.helpers

from fastforward._quantops import OperatorTable, operator, symtypes

STRICT_QUANTIZATION_PARAM = "strict_quantization"


module_header_raw = """
#
# Warning: you should not make changes to this file directly.
# This file is generated based on '{src_file}'.
#

"""


class Writer(Protocol):
    """Writer for code generation.

    Implementations of the writer protocol are used to 'export' the generated
    code.
    """

    def write(self, __data: str) -> Any:
        """Write `__data` to output.

        Args:
            __data: data to write to output.
        """
        raise NotImplementedError


class _ModuleGenerator:
    def __init__(self, src_file: pathlib.Path):
        self._module = libcst.parse_module(
            module_header_raw.format(src_file=src_file.relative_to(os.getcwd()))
        )
        self._imports: list[libcst.SimpleStatementLine | libcst.BaseCompoundStatement] = []
        self._header_parts: list[
            Sequence[libcst.SimpleStatementLine | libcst.BaseCompoundStatement]
        ] = []
        self._ops: list[libcst.FunctionDef] = []

    @property
    def code(self) -> str:
        all_ops = ", ".join([f'"{op.name.value}"' for op in self._ops])
        all_statement = f"__all__ = [{all_ops}]"
        parts = [statement for part in self._header_parts for statement in part]
        return self._module.with_changes(
            body=[
                *self._module.body,
                *self._imports,
                *parts,
                _statement(all_statement),
                *self._ops,
            ]
        ).code

    @property
    def config(self) -> libcst.PartialParserConfig:
        return self._module.config_for_parsing

    def append_import(self, import_module: str, from_module: str | None = None):
        if from_module is not None:
            import_str = f"from {from_module} import {import_module}"
        else:
            import_str = f"import {import_module}"
        self._imports.append(libcst.parse_statement(import_str, config=self.config))

    def append_op(self, op_funcdef: libcst.FunctionDef) -> None:
        self._ops.append(op_funcdef)

    def append_raw(self, raw_source: str) -> None:
        source_cst = libcst.parse_module(textwrap.dedent(raw_source))
        self._header_parts.append(source_cst.body)


class _ParameterList:
    def __init__(self) -> None:
        self._params: list[tuple[str, str]] = []  # name, value

    def append(self, name: str, value: str):
        self._params.append((name, value))

    def __repr__(self) -> str:
        return ", ".join((f"{name}={value}" for name, value in self._params))


BASE_FUNCTION_DEF = cast(libcst.FunctionDef, libcst.parse_module("def name():\n    pass").body[0])
OptQuantizer = symtypes.Optional[symtypes.Type('"Quantizer"')]
OptBool = symtypes.Optional[symtypes.Bool]


def _type_expression(symtype_or_typestr: symtypes.Type | str | None) -> libcst.BaseExpression:
    if symtype_or_typestr is None:
        return libcst.parse_expression("None")
    elif isinstance(symtype_or_typestr, symtypes.Type):
        SizeT = symtypes.Type("Size")
        symtype = symtype_or_typestr.replace(symtypes.QuantizedTensor, symtypes.Tensor)
        symtype = symtype.replace(symtypes.Size, SizeT)
        return libcst.parse_expression(symtypes.type_to_python_str(symtype))
    else:
        return libcst.parse_expression(symtype_or_typestr)


def _parameter(param: operator.Parameter) -> libcst.Param:
    default: libcst.BaseExpression | None = None
    if param.default_value is not None:
        default = libcst.parse_expression(param.default_value)
    return libcst.Param(
        name=libcst.Name(value=param.name),
        annotation=libcst.Annotation(_type_expression(param.param_type)),
        default=default,
    )


def _expression(exp: str | libcst.BaseExpression) -> libcst.BaseExpression:
    if isinstance(exp, str):
        return libcst.parse_expression(exp)
    return exp


def _statement(stmt: str | libcst.BaseStatement) -> libcst.BaseStatement:
    if isinstance(stmt, str):
        return libcst.parse_statement(stmt)
    return stmt


def _operator_function_stub(
    op: operator.Operator, extra_params: Sequence[libcst.Param] = ()
) -> libcst.FunctionDef:
    """Returns FunctionDef with correct signature for op and empty body.

    Args:
        op: Operator to create function for
        extra_params: extra params that are not specified by op. Each is added
            to function signature as kwonly parameter.
    """
    if not op.metadata or not op.metadata.specification_file:
        raw_warning = "This function was automatically generated"
    else:
        raw_warning = (
            "Automatically generated based on "
            f"{op.metadata.specification_file.relative_to(os.getcwd())}:"
            f"{op.metadata.line_number}"
        )
    comments = (libcst.EmptyLine(comment=libcst.Comment(f"# {raw_warning}")),)
    func_def = BASE_FUNCTION_DEF.with_changes(
        name=libcst.Name(op.identifier),
        leading_lines=comments,
    )

    params = [_parameter(param) for param in op.parameters]
    func_def = func_def.with_deep_changes(
        func_def.params, params=params, kwonly_params=extra_params
    )
    func_def = func_def.with_changes(returns=libcst.Annotation(_type_expression(op.return_type)))

    return func_def


def _simple_if(
    test_exp: str | libcst.BaseExpression,
    body_stmt: str | libcst.BaseStatement | libcst.IndentedBlock,
    orelse_stmt: str | libcst.BaseStatement | None = None,
    leading_empty_line: bool = True,
) -> libcst.If:
    orelse: libcst.If | libcst.Else | None
    if orelse_stmt is not None:
        if isinstance(orelse_stmt, (libcst.Else, libcst.If)):
            orelse = orelse_stmt
        else:
            orelse = libcst.Else(libcst.IndentedBlock([_statement(orelse_stmt)]))
    else:
        orelse = None
    if not isinstance(body_stmt, libcst.IndentedBlock):
        body = libcst.IndentedBlock([_statement(body_stmt)])
    else:
        body = body_stmt

    return libcst.If(
        test=_expression(test_exp),
        body=body,
        orelse=orelse,
        leading_lines=[libcst.EmptyLine()] if leading_empty_line else [],
    )


def _param_quantization_guard(param: operator.Parameter) -> list[libcst.BaseStatement]:
    TQuantList = symtypes.List[symtypes.MaybeQuantized] | symtypes.List[symtypes.QuantizedTensor]
    if param.param_type in TQuantList:
        return _quantizer_list_param_quantization_guard(param)
    return _single_param_quantization_guard(param)


def _quantizer_list_param_quantization_guard(
    param: operator.Parameter,
) -> list[libcst.BaseStatement]:
    list_type = param.param_type.parameters()[0]

    elem_name = "elem__"
    loop_param = operator.Parameter(list_type, elem_name, None)
    loop_body = _single_param_quantization_guard(loop_param)
    loop_body.append(libcst.parse_statement(f"elems__.append({elem_name})"))
    loop_node: libcst.For = libcst.parse_statement(f"for {elem_name} in {param.name}: pass")  # type: ignore[assignment]
    loop_node = loop_node.with_changes(body=libcst.IndentedBlock(loop_body))

    result_list = libcst.parse_statement("elems__: list[torch.Tensor] = []")
    return [
        result_list.with_changes(leading_lines=[libcst.EmptyLine()]),
        loop_node,
        libcst.parse_statement(f"{param.name} = elems__"),
    ]


def _single_param_quantization_guard(param: operator.Parameter) -> list[libcst.BaseStatement]:
    is_optional_param = symtypes.NoneType in param.param_type and len(param.param_type) > 1

    param_type = symtypes.unwrap_optional(param.param_type)
    always_quantized = param_type == symtypes.QuantizedTensor
    maybe_quantized = symtypes.QuantizedTensor in param_type
    always_quantized_if_tensor = symtypes.Tensor not in param_type and maybe_quantized
    check_quantized_if_strict = maybe_quantized and symtypes.Tensor not in param_type

    if not maybe_quantized:
        return []

    quant_check: list[libcst.BaseStatement] = []
    if check_quantized_if_strict:
        if always_quantized:
            cond = f"not isinstance({param.name}, QuantizedTensor)"
        elif always_quantized_if_tensor:
            cond = f"isinstance({param.name}, torch.Tensor) and not isinstance({param.name}, QuantizedTensor)"
        else:
            raise NotImplementedError(f"QuantiationGuard for {param.param_type} is not implemented")

        quant_check.append(
            _simple_if(
                f"{STRICT_QUANTIZATION_PARAM} and {cond}",
                (
                    "raise QuantizationError("
                    f"\"Expected '{param.name}' to be an instance of 'QuantizedTensor' \""
                    '"because strict_quantization=True.")'
                ),
            )
        )

    quant_check.append(
        _simple_if(
            f"isinstance({param.name}, QuantizedTensor)",
            f"{param.name} = {param.name}.dequantize()",
        )
    )

    if is_optional_param:
        return [_simple_if(f"{param.name} is not None", libcst.IndentedBlock(quant_check))]
    else:
        return quant_check


def _fallback_op(op: operator.Operator) -> libcst.FunctionDef:
    if op.metadata is None:
        raise ValueError("Cannot create fallback op without metadata")

    return_type = op.return_type
    has_output_quantizer = return_type is not None and symtypes.QuantizedTensor in return_type

    extra_params: list[libcst.Param] = []
    if has_output_quantizer:
        quantizer_param = _parameter(operator.Parameter(OptQuantizer, "output_quantizer", "None"))
        extra_params.append(quantizer_param)
    extra_params.append(
        _parameter(operator.Parameter(symtypes.Bool, STRICT_QUANTIZATION_PARAM, "True"))
    )

    func_def = _operator_function_stub(op, extra_params=extra_params)
    body: list[libcst.BaseStatement] = []

    # Check if output quantizer is given when required
    if has_output_quantizer:
        body.append(
            _simple_if(
                f"{STRICT_QUANTIZATION_PARAM} and output_quantizer is None",
                f"raise QuantizationError(\"'output_quantizer' must be provided "
                f'if {STRICT_QUANTIZATION_PARAM}=True")',
            )
        )

    # Ensure all parameters are properly quantized
    params = _ParameterList()
    for param in op.parameters:
        # The parameter names of the fallback function and the argument names
        # are the same; We can use param.name for both
        params.append(param.name, param.name)
        body += _param_quantization_guard(param)

    # fallback call
    body.append(
        _statement(f"output = {op.metadata.fallback}({params})").with_changes(
            leading_lines=[libcst.EmptyLine()]
        )
    )

    # Apply quantizer if provided
    body.append(
        _simple_if(
            "output_quantizer is not None",
            "output = output_quantizer(output)",
            leading_empty_line=False,
        )
    )
    if op.metadata.cast_output is not None:
        body.append(_statement(f"return cast({op.metadata.cast_output}, output)"))
    else:
        body.append(_statement("return output"))

    return func_def.with_changes(body=libcst.IndentedBlock(body))


def _dispatch_op(op: operator.Operator) -> libcst.FunctionDef:
    if op.metadata is None:
        raise ValueError("Cannot create dispatch op without metadata")

    return_type = op.return_type
    has_output_quantizer = return_type is not None and symtypes.QuantizedTensor in return_type

    extra_params: list[libcst.Param] = []
    if has_output_quantizer:
        quantizer_param = _parameter(operator.Parameter(OptQuantizer, "output_quantizer", "None"))
        extra_params.append(quantizer_param)
    extra_params.append(_parameter(operator.Parameter(OptBool, STRICT_QUANTIZATION_PARAM, "None")))

    func_def = _operator_function_stub(op, extra_params=extra_params)
    body: list[libcst.BaseStatement] = []

    strict_param = STRICT_QUANTIZATION_PARAM
    body.append(
        _simple_if(
            f"{strict_param} is None", f"{strict_param} = fastforward.get_strict_quantization()"
        )
    )

    params = _ParameterList()
    for param in op.parameters:
        params.append(param.name, param.name)
    if has_output_quantizer:
        params.append("output_quantizer", "output_quantizer")
    params.append("strict_quantization", "strict_quantization")

    op_ident = op.identifier
    body.append(
        _statement(f'dispatch_op = dispatch("{op_ident}", {params})').with_changes(
            leading_lines=[libcst.EmptyLine()]
        )
    )
    body.append(_statement(f"selected_op = dispatch_op or fallback.{op_ident}"))
    body.append(_statement(f"return selected_op({params})"))
    return func_def.with_changes(body=libcst.IndentedBlock(body))


def _add_imports(module: _ModuleGenerator) -> _ModuleGenerator:
    module.append_import(
        from_module="fastforward.quantized_tensor", import_module="QuantizedTensor"
    )
    module.append_import(from_module="typing", import_module="TypeAlias")
    module.append_import(from_module="typing", import_module="Union")
    module.append_import(from_module="typing", import_module="Sequence")
    module.append_import(import_module="torch")
    module.append_import(from_module="typing", import_module="cast")
    module.append_import(from_module="typing", import_module="Optional")
    module.append_import(from_module="typing", import_module="TYPE_CHECKING")
    module.append_import(from_module="fastforward.exceptions", import_module="QuantizationError")
    module.append_import(from_module="fastforward.dispatcher", import_module="dispatch")
    module.append_import(from_module=".", import_module="fallback")
    module.append_import(import_module="fastforward")

    return module


def _generate_fallback(operators: OperatorTable, source: pathlib.Path, writer: Writer) -> None:
    module = _add_imports(_ModuleGenerator(source))

    module.append_raw("""
        if TYPE_CHECKING:
            from fastforward.nn.quantizer import Quantizer
        Size: TypeAlias = Union[torch.Size, list[int], tuple[int, ...]]
    """)

    for op in operators.operators():
        module.append_op(_fallback_op(op))

    writer.write(module.code)


def _generate_operators(operators: OperatorTable, source: pathlib.Path, writer: Writer) -> None:
    module = _add_imports(_ModuleGenerator(source))
    module.append_import(from_module="fastforward.nn.quantizer", import_module="Quantizer")

    module.append_raw("""
        Size: TypeAlias = Union[torch.Size, list[int], tuple[int, ...]]
    """)

    for op in operators.operators():
        module.append_op(_dispatch_op(op))

    writer.write(module.code)


def generate(operators: OperatorTable, source: pathlib.Path, destination: pathlib.Path) -> None:
    """Generate "fallback.py" and "operators.py" from operators.

    Args:
        operators: The `OperatorTable` to generate files from.
        source: The source locationfrom which the files are created.
        destination: The directory to write the newly created files to.
    """
    with (destination / "fallback.py").open("w") as dest_file:
        _generate_fallback(operators, source=source, writer=dest_file)
    with (destination / "operators.py").open("w") as dest_file:
        _generate_operators(operators, source=source, writer=dest_file)
