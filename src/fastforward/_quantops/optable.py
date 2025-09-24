# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import dataclasses
import pathlib

from collections.abc import Sequence
from typing import Any, Callable, Hashable, Iterator, TypeAlias

import libcst
import torch
import yaml

from typing_extensions import Self

from fastforward._import import QualifiedNameReference, fully_qualified_name
from fastforward._quantops.operator import Operator

_PyOp: TypeAlias = Callable[..., Any]


def _resolve_name(py_op: str | _PyOp) -> str:
    return py_op if isinstance(py_op, str) else fully_qualified_name(py_op)


def _default_yaml_file() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "quantized_operators.yaml"


class _SafeLoaderWithLines(yaml.loader.SafeLoader):
    """Load YAML file while also storing line numbers using __line__ key."""

    def construct_mapping(
        self, node: yaml.nodes.MappingNode, *args: Any, **kwargs: Any
    ) -> dict[Hashable, Any]:
        mapping = super().construct_mapping(node, *args, **kwargs)
        mapping["__line__"] = node.start_mark.line + 1
        return mapping


def _fallback_alias(op: Operator) -> str | None:
    if metadata := op.metadata:
        return metadata.fallback
    return None


def _functional_alias(op: Operator) -> str | None:
    if not (metadata := op.metadata):
        return None
    if not (dispatch_op := metadata.dispatch_op):
        return None

    if func := getattr(torch.nn.functional, op.identifier, None):
        try:
            fallback = QualifiedNameReference(metadata.fallback).import_()
        except ImportError:
            return None
        if func is fallback:
            return f"F.{dispatch_op.__name__}"
    return None


STR_ALIASES_EXTENSIONS = (_fallback_alias, _functional_alias)


class OperatorTable:
    """Lookup table for quantized operators.

    In-memory representation of quantized_operators.yaml that can be extended
    at runtime. The table maintains a registry of quantized operator specifications
    indexed by their fallback (unquantized) operations.

    Operator Overloading:
        Multiple operator specifications can be registered for the same fallback
        operation, creating an overload set. When looking up operators via `get()`
        or `__getitem__`, the most recently added operator is returned first,
        allowing runtime customization and extension of quantization behavior.

    Example:
        >>> table = OperatorTable.from_yaml()
        >>> # Add custom quantized conv2d variant
        >>> table.add(
        ...     "conv2d(Tensor, Quantizer, Tensor, Quantized) -> Quantized",
        ...     dispatch_op=my_conv2d_dispatch,
        ...     fallback_op=torch.nn.functional.conv2d
        ... )
        >>> # This new spec will be returned first when looking up conv2d
        >>> ops = list(table.get(torch.nn.functional.conv2d))
        >>> # ops[0] is the custom spec, ops[1:] are the original specs

    Aliases:
        String aliases can be registered to reference operators by name instead
        of by function reference. This is useful for configuration and dynamic
        lookup scenarios.

    Args:
        alias_extensions: Sequence of callables that generate additional aliases
            for operators as they are added. Each callable receives an `Operator`
            and returns an optional alias string.
        _resolve_dispatch: If True, automatically resolve dispatch operations from
            `fastforward.nn.functional` based on operator identifiers.
    """

    def __init__(
        self,
        *,
        alias_extensions: Sequence[Callable[[Operator], str | None]] = (),
        _resolve_dispatch: bool = True,
    ) -> None:
        self._operator_specs: list[list[Operator]] = []
        self._py_op_index: dict[_PyOp, int] = {}
        self._py_op_aliases: dict[str, _PyOp] = {}
        self._resolve_dispatch = _resolve_dispatch

        self._alias_extensions = list(alias_extensions)

    def append_operator(self, operator: Operator) -> None:
        """Add a new operator to the table.

        If an operator with the same fallback already exists, this creates an
        overload. The new operator will be returned first during lookup, allowing
        users to override or extend existing quantization specifications.

        The operator _must_ have a non-empty metadata field.

        If the `operator.metadata.dispatch_op` field is None, a dispatch_op is
        inferred from `fastforward.nn.functional` based on the operator
        identifier. An error is raised if this operator does not exist.

        A default lookup alias is added based on the `operator.metadata.fallback`.

        Args:
            operator: `Operator` to add to table. Must have metadata populated.

        Raises:
            ValueError: If operator has no metadata or if dispatch_op cannot be
                resolved.
        """
        if not (metadata := operator.metadata):
            raise ValueError("Cannot add operator without metadata")

        py_op = QualifiedNameReference(metadata.fallback).import_()

        if not operator.metadata.dispatch_op and self._resolve_dispatch:
            dispatch_op = self._dispatch_op(operator.identifier)
            new_metadata = dataclasses.replace(operator.metadata, dispatch_op=dispatch_op)
            operator = dataclasses.replace(operator, metadata=new_metadata)

        if py_op in self._py_op_index:
            spec_idx = self._py_op_index[py_op]
            self._operator_specs[spec_idx].append(operator)
        else:
            self._py_op_index[py_op] = spec_idx = len(self._operator_specs)
            self._operator_specs.append([operator])

            assert operator.metadata  # helping mypy and friends
            for alias_ext in self._alias_extensions:
                if alias := alias_ext(operator):
                    self.add_alias(alias, py_op)

    def add(
        self,
        spec: str,
        fallback_op: str | _PyOp,
        dispatch_op: str | _PyOp | None = None,
        intermediate_quantizers: tuple[str, ...] = (),
        **kwargs: Any,
    ) -> None:
        """Add an operator specification to the table.

        This is a convenience method that constructs an `Operator` from a
        specification string and adds it to the table. Multiple calls with the
        same `fallback_op` create overloads.

        Args:
            spec: Operator specification string describing the signature
                (e.g., "conv2d(Tensor, Quantized, Tensor, Quantized) -> Quantized")
            fallback_op: The unquantized operation this spec replaces. Can be
                a string qualified name or a direct function reference.
            dispatch_op: The quantized implementation to dispatch to. If None
                and `_resolve_dispatch` is True, automatically resolved from
                `fastforward.nn.functional`.
            intermediate_quantizers: Tuple of quantizer names for intermediate
                values.
            **kwargs: Additional metadata fields to attach to the operator.

        Raises:
            ValueError: If the spec is invalid or dispatch_op cannot be resolved.
        """
        resolved_dispatch_op: _PyOp | None = None
        if isinstance(dispatch_op, str):
            resolved_dispatch_op = (
                self._dispatch_op(dispatch_op) if self._resolve_dispatch else None
            )
        else:
            resolved_dispatch_op = dispatch_op
        operator = Operator.from_spec(
            spec,
            fallback=_resolve_name(fallback_op),
            dispatch_op=resolved_dispatch_op,
            intermediate_quantizers=intermediate_quantizers,
            **kwargs,
        )
        self.append_operator(operator)

    def _dispatch_op(self, name: str) -> _PyOp:
        qualified_name = f"fastforward.nn.functional.{name}"
        try:
            return QualifiedNameReference(qualified_name).import_()  # type: ignore[no-any-return]
        except (ImportError, AttributeError):
            msg = f"No dispatch op was specified for '{name}' and '{qualified_name}' does not exist"
            raise ValueError(msg)

    def _clear_aliases(self, py_op: _PyOp) -> None:
        for alias, target in self._py_op_aliases.items():
            if py_op is target:
                del self._py_op_aliases[alias]

    @classmethod
    def from_yaml(
        cls,
        source: pathlib.Path | None = None,
        *,
        alias_extensions: Sequence[Callable[[Operator], str | None]] = (),
        _resolve_dispatch: bool = True,
    ) -> Self:
        """Create an `OperatorTable` from yaml file at `path`.

        Args:
            source: Path to the yaml file. If None, uses the default
                quantized_operators.yaml file.
            alias_extensions: Sequence of alias extensions that can alter the
                operator lookup of this table.
            _resolve_dispatch: If `True`, resolve the default dispatch function
                using standard heuristics.

        Returns:
            `OperatorTable` constructed from `path`
        """
        source = source or _default_yaml_file()
        with source.open() as f:
            raw_source = yaml.load(f, Loader=_SafeLoaderWithLines)

        table = cls(alias_extensions=alias_extensions, _resolve_dispatch=_resolve_dispatch)
        errors: list[tuple[int, str]] = []

        for op_info in raw_source:
            try:
                table.add(
                    op_info["op"],
                    specification_file=pathlib.Path(source),
                    line_number=op_info["__line__"],
                    fallback_op=op_info["fallback"],
                    cast_output=op_info.get("cast_output", None),
                )
            except ValueError as e:
                errors.append((
                    op_info["__line__"],
                    f"Unable to parse {op_info['op']} because {e.args}",
                ))

        if errors:
            error_list = "\n".join([f"  - [line: {line}] {err}" for line, err in errors])
            msg = f"Unable to parse {source} because of the following errors\n\n{error_list}"
            raise RuntimeError(msg)

        return table

    def operators(self) -> Iterator[Operator]:
        """All operators in the table.

        Returns:
            `Iterator` over all operators in the table.
        """
        for specs in self._operator_specs:
            yield from specs

    def get(self, key: _PyOp | str) -> Iterator[Operator]:
        """Lookup operators based on the fallback operation.

        Returns an iterator over all operator specifications registered for the
        given fallback operation, in reverse order of registration (most recent
        first). This allows users to override default quantization behavior by
        adding custom operators that will be matched first.

        Args:
            key: Either a reference to the operator or a string referencing
                the fallback name or alias

        Returns:
            `Iterator` over operators associated with key, most recent first

        Raises:
            KeyError: If no operator is registered for the given key

        Example:
            >>> table = OperatorTable.from_yaml()
            >>> # Get all conv2d quantization specs
            >>> for op in table.get("torch.nn.functional.conv2d"):
            ...     print(op.identifier)
        """
        alias: str | None = None
        if isinstance(key, str):
            alias = key  # used for error reporting
            key = self._resolve_alias(key)

        try:
            spec_idx = self._py_op_index[key]
        except KeyError as e:
            name = alias or str(key)
            msg = f"{type(self).__name__} contains no operator for {name}"
            raise KeyError(msg) from e

        yield from reversed(self._operator_specs[spec_idx])

    def __iter__(self) -> Iterator[Operator]:
        yield from self.operators()

    def __getitem__(self, key: str | _PyOp) -> Iterator[Operator]:
        return self.get(key)

    def __contains__(self, key: str | _PyOp) -> bool:
        try:
            self.get(key)
        except KeyError:
            return False
        return True

    def _resolve_alias(self, alias: str) -> _PyOp:
        try:
            return self._py_op_aliases[alias]
        except KeyError as e:
            msg = f"'{alias}' is not a known alias"
            raise KeyError(msg) from e

    def add_alias(self, alias: str, py_op: str | _PyOp) -> None:
        """Add a string alias for an operation.

        Both `get()` and `__getitem__` will resolve string aliases before lookup.
        This may be used to associate multiple string (qualified) names
        to a single operator, enabling flexible lookup patterns.

        Args:
            alias: The string alias to use (e.g., "my_custom_conv")
            py_op: The operator to associate the alias with. This may itself
                be an alias, which will be resolved first.

        Example:
            >>> table.add_alias("conv", torch.nn.functional.conv2d)
            >>> ops = list(table.get("conv"))  # Same as table.get(torch.nn.functional.conv2d)
        """
        if isinstance(py_op, str):
            py_op = self._resolve_alias(py_op)
        self._py_op_aliases[alias] = py_op


UNARY_OPS_LIBCST_TO_TORCH_MAPPING: dict[type[libcst.CSTNode], Callable[..., Any]] = {
    libcst.BitInvert: torch.bitwise_not,
    libcst.Minus: torch.negative,
    libcst.Plus: torch.positive,
}

BINARY_OPS_LIBCST_TO_TORCH_MAPPING: dict[type[libcst.CSTNode], Callable[..., Any]] = {
    libcst.Add: torch.add,
    libcst.BitAnd: torch.bitwise_and,
    libcst.BitOr: torch.bitwise_or,
    libcst.BitXor: torch.bitwise_xor,
    libcst.Divide: torch.div,
    libcst.FloorDivide: torch.floor_divide,
    libcst.LeftShift: torch.bitwise_left_shift,
    libcst.MatrixMultiply: torch.matmul,
    libcst.Modulo: torch.remainder,
    libcst.Multiply: torch.mul,
    libcst.Power: torch.pow,
    libcst.RightShift: torch.bitwise_right_shift,
    libcst.Subtract: torch.sub,
}

OPS_LIBCST_TO_TORCH_MAPPING = UNARY_OPS_LIBCST_TO_TORCH_MAPPING | BINARY_OPS_LIBCST_TO_TORCH_MAPPING
