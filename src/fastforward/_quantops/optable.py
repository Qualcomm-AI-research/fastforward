# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import dataclasses
import pathlib

from typing import Any, Callable, Hashable, Iterator, TypeAlias, overload

import torch
import yaml

from typing_extensions import Self

from fastforward._import import QualifiedNameReference, fully_qualified_name
from fastforward._quantops import spec_parser
from fastforward._quantops.operator import Operator, OperatorMetadata

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

    In memory representation of quantized_operators.yaml that can be extended
    at runtime.
    """

    def __init__(
        self,
        *,
        alias_extensions: Sequence[Callable[[Operator], str | None]] = (),
        _resolve_dispatch: bool = True,
    ) -> None:
        self._operator_specs: list[Operator] = []
        self._py_op_index: dict[_PyOp, int] = {}
        self._py_op_aliases: dict[str, _PyOp] = {}
        self._resolve_dispatch = _resolve_dispatch

        self._alias_extensions = list(alias_extensions)

    def append_operator(self, operator: Operator) -> None:
        """Add a new operator to the table.

        The operator _must_ have a non-empty metadata field.

        If the `operator.metadata.dispatch_op` field is None, a dispatch_op is
        inferred from `fastforward.nn.functional` based on the operator
        identifier. An error is raised if this operator does not exist.

        A default lookup alias is added based on the `operator.metadata.fallback`.

        Args:
            operator: `Operator` to add to table
        """
        if not (metadata := operator.metadata):
            raise ValueError("Cannot add operator without metadata")

        py_op = QualifiedNameReference(metadata.fallback).import_()

        if not operator.metadata.dispatch_op and self._resolve_dispatch:
            dispatch_op = self._dispatch_op(operator.identifier)
            new_metadata = dataclasses.replace(operator.metadata, dispatch_op=dispatch_op)
            operator = dataclasses.replace(operator, metadata=new_metadata)

        if spec_idx := self._py_op_index.get(py_op, None):
            self._clear_aliases(py_op)
            self._operator_specs[spec_idx] = operator
        else:
            self._py_op_index[py_op] = spec_idx = len(self._operator_specs)
            self._operator_specs.append(operator)

        assert operator.metadata  # helping mypy and friends
        for alias_ext in self._alias_extensions:
            if alias := alias_ext(operator):
                self.add_alias(alias, py_op)

    def add(
        self,
        schema: str,
        fallback_op: str | _PyOp,
        dispatch_op: str | _PyOp | None = None,
        **kwargs: Any,
    ) -> None:
        operator = spec_parser.parse_schema(schema)
        resolved_dispatch_op: _PyOp | None = None
        if self._resolve_dispatch and isinstance(dispatch_op, str):
            resolved_dispatch_op = self._dispatch_op(dispatch_op)
        operator = dataclasses.replace(
            operator,
            metadata=OperatorMetadata(
                fallback=_resolve_name(fallback_op),
                dispatch_op=resolved_dispatch_op,
                **kwargs,
            ),
        )
        self.append_operator(operator)

    def _dispatch_op(self, name: str) -> _PyOp:
        qualified_name = f"fastforward.nn.functional.{name}"
        try:
            return QualifiedNameReference(qualified_name).import_()  # type: ignore[no-any-return]
        except (ImportError, AttributeError):
            raise ValueError(
                f"No dispatch op was specified for '{name}' and '{qualified_name}' does not exist"
            )

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
            source: Path to the yaml file
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
            except spec_parser.ParseError as e:
                errors.append((
                    op_info["__line__"],
                    f"Unable to parse {op_info['op']} because {e.args}",
                ))

        if errors:
            error_list = "\n".join([f"  - [line: {line}] {err}" for line, err in errors])
            raise RuntimeError(
                f"Unable to parse {source} because of the following errors\n\n{error_list}"
            )

        return table

    def operators(self) -> Iterator[Operator]:
        """All operaotrs in the table.

        Returns:
            `Iterator` over all operators in the table.
        """
        yield from self._operator_specs

    def get(self, key: _PyOp | str) -> Operator:
        """Lookup operator based on the fallback operator.

        Args:
            key: Either a reference to the operator or a string referencing
                the fallback name or alias

        Returns:
            `Operator` associated with key
        """
        alias: str | None = None
        if isinstance(key, str):
            alias = key  # used for error reporting
            key = self._resolve_alias(key)

        try:
            spec_idx = self._py_op_index[key]
        except KeyError as e:
            name = alias or str(key)
            raise KeyError(f"{type(self).__name__} contains no operator for {name}") from e

        return self._operator_specs[spec_idx]

    def __iter__(self) -> Iterator[Operator]:
        yield from self.operators()

    def __getitem__(self, key: str | _PyOp) -> Operator:
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
            raise KeyError(f"'{alias}' is not a known alias") from e

    def add_alias(self, alias: str, py_op: str | _PyOp) -> None:
        """Add an string alias for an operation.

        Both get and __getitem__ will resolve string aliases before lookup.

        This may be used to associate multiple string (qualified) names
        to a single operator.

        Args:
            alias: The string alias to use
            py_op: The operator to associate the alias with, this may be an
                alias
        """
        if isinstance(py_op, str):
            py_op = self._resolve_alias(py_op)
        self._py_op_aliases[alias] = py_op
