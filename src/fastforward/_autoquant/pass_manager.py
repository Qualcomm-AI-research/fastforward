# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#
import dataclasses
import functools

from typing import Generic, Sequence, TypeAlias

import libcst


@dataclasses.dataclass
class MetadataTransformer:
    """A wrapper for a CST transformer that requires metadata.

    This class encapsulates a LibCST transformer that needs access to metadata
    during the transformation process. It provides configuration options for
    how the transformer should be applied to the code.

    Attributes:
        transformer: The LibCST transformer to apply to the code.
        wrap_in_module: If True, the code will be wrapped in a Module before
                        transformation and unwrapped afterward. This is useful
                        when the transformer requires metadata that is only
                        available at the module level.
    """

    transformer: libcst.CSTTransformer | libcst.CSTVisitor
    wrap_in_module: bool = False


@dataclasses.dataclass
class PassResult(Generic[libcst.CSTNodeT]):
    """Represents the result of applying a transformation pass to a CST node.

    This class stores both the resulting CST node after transformation and a flag
    indicating whether the node was actually modified during the transformation.

    Attributes:
        cst: The resulting CST node after transformation.
        altered: Boolean flag indicating whether the CST was modified during
                transformation (True) or remained unchanged (False).
    """

    cst: libcst.CSTNodeT
    altered: bool


_PassT: TypeAlias = (
    libcst.CSTTransformer
    | libcst.CSTVisitor
    | MetadataTransformer
    | type[libcst.CSTTransformer]
    | type[libcst.CSTVisitor]
)


class PassManager:
    """Manages and applies a sequence of transformation passes to a CST node.

    This class orchestrates the application of multiple transformation passes to a
    Code Syntax Tree (CST) node. It tracks whether each pass alters the CST and
    validates the results after modifications.

    The PassManager supports two types of transformers:
    1. Standard LibCST transformers
    2. MetadataTransformers that require access to metadata

    Attributes:
        _passes: A sequence of transformation passes to apply in order.
    """

    def __init__(self, passes: Sequence[_PassT]) -> None:
        self._passes = passes

    def __call__(self, cst: libcst.CSTNodeT) -> libcst.CSTNodeT:
        current = cst
        result = PassResult(cst=cst, altered=False)
        for i, pass_ in enumerate(self._passes):
            try:
                result = self._apply_pass(pass_, current)
            except Exception as e:
                msg = (
                    f"An error occurred during the application of pass {i} "
                    + f"({self._pass_repr(pass_)}) on '{type(current).__name__}'"
                )
                raise PassManagerError(msg) from e
            if result.altered:
                self._validate(result)
            current = result.cst

        return result.cst

    def _pass_repr(self, pass_: _PassT) -> str:
        match pass_:
            case libcst.CSTVisitor() | libcst.CSTTransformer():
                return type(pass_).__name__
            case MetadataTransformer():
                return (
                    "MetadataTransformer("
                    + f"{self._pass_repr(pass_.transformer)}, {pass_.wrap_in_module})"
                )
            case _:
                return str(pass_)

    def _validate(self, result: PassResult[libcst.CSTNodeT]) -> None:
        pass

    @functools.singledispatchmethod
    def _apply_pass(self, pass_: _PassT, cst: libcst.CSTNodeT) -> PassResult[libcst.CSTNodeT]:
        msg = (
            f"No implementation found for applying pass of type {type(pass_).__name__} to "
            + f"CST node of type {type(cst).__name__}"
        )
        raise NotImplementedError(msg)

    @_apply_pass.register
    def _apply_cst_transformer(
        self, pass_: libcst.CSTTransformer, cst: libcst.CSTNodeT
    ) -> PassResult[libcst.CSTNodeT]:
        if len(pass_.METADATA_DEPENDENCIES) > 0:
            return self._apply_pass(
                MetadataTransformer(transformer=pass_, wrap_in_module=True),
                cst,
            )
        else:
            result_cst = _visit_cst_same(cst, pass_)
            return PassResult(cst=result_cst, altered=result_cst is not cst)

    @_apply_pass.register
    def _apply_cst_visitor(
        self, pass_: libcst.CSTVisitor, cst: libcst.CSTNodeT
    ) -> PassResult[libcst.CSTNodeT]:
        if len(pass_.METADATA_DEPENDENCIES) > 0:
            return self._apply_pass(
                MetadataTransformer(transformer=pass_, wrap_in_module=True),
                cst,
            )
        else:
            result_cst = _visit_cst_same(cst, pass_)
            return PassResult(cst=result_cst, altered=result_cst is not cst)

    @_apply_pass.register
    def _apply_metadata_transformer(
        self, pass_: MetadataTransformer, cst: libcst.CSTNodeT
    ) -> PassResult[libcst.CSTNodeT]:
        process_cst: libcst.CSTNode = cst
        if pass_.wrap_in_module:
            process_cst = libcst.Module([process_cst])  # type: ignore[list-item]

        result_cst = _visit_cst_same(process_cst, pass_.transformer, requires_metadata=True)

        if pass_.wrap_in_module:
            if not isinstance(result_cst, libcst.Module):
                msg = f"Expected a Module after transformation, but got {type(result_cst).__name__}"
                raise ValueError(msg)
            if len(result_cst.body) == 0:
                msg = "Transformation resulted in an empty Module with no statements"
                raise ValueError(msg)
            if not isinstance(result_cst := result_cst.body[0], type(cst)):
                msg = (
                    f"Expected a {type(cst).__name__} as the first statement in Module, but got "
                    + f"{type(result_cst).__name__}"
                )
                raise ValueError(msg)
        if not isinstance(result_cst, type(cst)):
            msg = (
                f"Expected transformation to result in a {type(cst).__name__}, but got "
                + f"{type(result_cst).__name__}"
            )
            raise ValueError(msg)

        return PassResult(cst=result_cst, altered=result_cst is not cst)

    @_apply_pass.register
    def _apply_metadata_callable(
        self, pass_: type, cst: libcst.CSTNodeT
    ) -> PassResult[libcst.CSTNodeT]:
        return self._apply_pass(pass_(), cst)


class PassManagerError(Exception):
    pass


def _visit_cst_same(
    cst: libcst.CSTNodeT,
    visitor: libcst.CSTVisitor | libcst.CSTTransformer,
    requires_metadata: bool = False,
) -> libcst.CSTNodeT:
    """Apply a visitor to a CST node and ensure the result is of the same type."""
    process_cst = cst if not requires_metadata else libcst.MetadataWrapper(cst)  # type: ignore[arg-type]
    result = process_cst.visit(visitor)
    CST = type(cst)
    if not isinstance(result, CST):
        msg = (
            f"Expected application of '{type(visitor).__name__}' on '{CST.__name__}' to result in "
            + f"a '{CST.__name__}' but got '{type(result).__name__}' instead"
        )
        raise ValueError(msg)
    return result
