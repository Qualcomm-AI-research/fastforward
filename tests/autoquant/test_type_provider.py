# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Check real type inference through Autoquant's selected backend.

These checks use TypeProvider and TypeInfo predicates so FASTFORWARD_TYPE_BACKEND
can select any registered backend. The current default is native mypy; no replies
are mocked.
"""

from pathlib import Path

import libcst
import pytest

from fastforward._autoquant.cst.filter import filter_nodes_by_type
from fastforward._autoquant.tsp.context import module_context
from fastforward._autoquant.tsp_provider import TypeInfo, TypeProvider, type_inference_scope


@pytest.mark.slow
def test_assignment_targets_and_values_have_inferred_numeric_types() -> None:
    """Expose scalar types for annotated assignments and a computed expression.

    Autoquant needs metadata on both values and their assignment targets. The
    unannotated sum must also have a concrete numeric type; merely returning
    metadata or matching two opaque backend objects would not establish that.
    """
    # GIVEN two annotated float values and an unannotated sum.
    module = libcst.parse_module("""
a: float = 3.14
b: float = 2.7
c = a + b
""")

    # WHEN the selected backend resolves expression metadata.
    metadata = libcst.MetadataWrapper(module, unsafe_skip_copy=True).resolve(TypeProvider)

    # THEN every assignment target and value is a proven float scalar.
    assignments = list(filter_nodes_by_type(module, (libcst.Assign, libcst.AnnAssign)))
    assert len(assignments) == 3
    for assignment in assignments:
        assert isinstance(assignment, (libcst.Assign, libcst.AnnAssign))
        target = (
            assignment.target if isinstance(assignment, libcst.AnnAssign) else assignment.targets[0]
        )
        assert assignment.value is not None
        for expression in (target, assignment.value):
            info = metadata[expression]
            assert isinstance(info, TypeInfo)
            assert info.is_subtype("builtins.float")
            assert not info.is_subtype("builtins.int")
            assert info.is_proven_scalar()


@pytest.mark.slow
def test_scalar_facts_distinguish_numeric_unknown_and_mixed_types() -> None:
    """Require positive numeric evidence before allowing scalar operands.

    Real inference must preserve numeric unions and literals as scalars while
    rejecting optional, unknown, and text values as scalar proofs. Otherwise
    Autoquant could omit a needed quantizer or attempt to quantize a scalar.
    """
    # GIVEN numeric arguments, a numeric literal, and arguments without scalar proof.
    module = libcst.parse_module("""
from typing import Any, Literal

def values(integer: int, floating: float, boolean: bool, numeric: int | float,
           optional: int | None, unknown: Any, text: str, literal: Literal[1]):
    return integer, floating, boolean, numeric, optional, unknown, text, literal
""")

    # WHEN the selected backend provides types for the returned expressions.
    metadata = libcst.MetadataWrapper(module, unsafe_skip_copy=True).resolve(TypeProvider)
    returned = next(filter_nodes_by_type(module, libcst.Return)).value
    assert isinstance(returned, libcst.Tuple)
    proven = {
        module.code_for_node(element.value): metadata[element.value].is_proven_scalar()
        for element in returned.elements
    }

    # THEN every possible type must be numeric; Any and optional types are not proofs.
    assert proven == {
        "integer": True,
        "floating": True,
        "boolean": True,
        "numeric": True,
        "optional": False,
        "unknown": False,
        "text": False,
        "literal": True,
    }


@pytest.mark.slow
def test_len_returns_an_integer_scalar_for_tensors_lists_and_shapes() -> None:
    """Infer scalar lengths even when the argument is a tensor or its shape.

    Autoquant must not quantize a length used as a dimension or scalar operand.
    Query the unannotated calls themselves so an annotation on an assignment
    cannot hide missing return-type inference from the selected backend.
    """
    # GIVEN builtin len calls on a tensor, an ordinary list, and a tensor shape.
    module = libcst.parse_module("""
import torch

def lengths(tensor: torch.Tensor, values: list[int]):
    return len(tensor), len(values), len(tensor.shape)
""")

    # WHEN the selected backend infers each call's return type from real source.
    metadata = libcst.MetadataWrapper(module, unsafe_skip_copy=True).resolve(TypeProvider)
    returned = next(filter_nodes_by_type(module, libcst.Return)).value
    assert isinstance(returned, libcst.Tuple)

    # THEN all lengths are integer scalars, not tensors or unknown types.
    for element in returned.elements:
        expression = module.code_for_node(element.value)
        info = metadata[element.value]
        assert info.is_subtype("builtins.int"), expression
        assert not info.is_subtype("builtins.bool"), expression
        assert info.is_proven_scalar(), expression
        assert not info.is_subtype("torch.Tensor"), expression


@pytest.mark.slow
def test_indexing_a_tensor_shape_returns_an_integer_scalar() -> None:
    """Recognize individual shape dimensions as scalars after indexing.

    Shape unpacking and tensor.size(dim) do not exercise subscript inference.
    Autoquant also needs scalar metadata for tensor.shape[index], including
    negative indices and indexing a shape stored in a local variable.
    """
    # GIVEN direct shape indexing and a shape stored before indexing by a variable.
    module = libcst.parse_module("""
import torch

def dimensions(tensor: torch.Tensor, index: int):
    shape = tensor.shape
    return tensor.shape[0], tensor.shape[-1], shape[index]
""")

    # WHEN the selected backend infers the unannotated subscript expressions.
    metadata = libcst.MetadataWrapper(module, unsafe_skip_copy=True).resolve(TypeProvider)
    returned = next(filter_nodes_by_type(module, libcst.Return)).value
    assert isinstance(returned, libcst.Tuple)

    # THEN dimensions remain integer scalars instead of inheriting the tensor's type.
    for element in returned.elements:
        expression = module.code_for_node(element.value)
        info = metadata[element.value]
        assert info.is_subtype("builtins.int"), expression
        assert not info.is_subtype("builtins.bool"), expression
        assert info.is_proven_scalar(), expression
        assert not info.is_subtype("torch.Tensor"), expression


@pytest.mark.slow
def test_same_variable_name_in_different_functions_gets_its_own_type() -> None:
    """Attach inferred types to the right occurrences of a reused variable name.

    Autoquant decides how to transform expressions from their node metadata.
    Mapping by name alone would mix up an int operand and a str operand in
    separate functions. This checks real inference and source mapping without
    running quantizer insertion.
    """
    # GIVEN two occurrences of `value` with different types and source positions.
    module = libcst.parse_module("""
def count(value: int):
    return value

def label(value: str):
    return value
""")
    returns = list(filter_nodes_by_type(module, libcst.Return))
    integer, text = (node.value for node in returns)
    assert integer is not None and text is not None

    # WHEN the selected backend associates inferred types with CST nodes.
    metadata = libcst.MetadataWrapper(module, unsafe_skip_copy=True).resolve(TypeProvider)

    # THEN types are attached by source position, not just by the name `value`.
    assert metadata[integer].is_subtype("builtins.int")
    assert metadata[integer].is_proven_scalar()
    assert metadata[text].is_subtype("builtins.str")
    assert not metadata[text].is_subtype("builtins.int")
    assert not metadata[text].is_proven_scalar()


@pytest.mark.slow
def test_identical_source_in_different_packages_uses_its_own_imports(tmp_path: Path) -> None:
    """Keep inferred import types specific to the document's source path.

    Identical source in two package roots can import different definitions.
    Reusing analysis based only on source text or module name would give the
    second document stale types. This checks real imports within one call scope.
    """
    # GIVEN identical source and module names with dependencies in different roots.
    code = "from .dependency import value\nresult = value\n"
    paths = []
    for root, value in (("first", "1"), ("second", '"text"')):
        package = tmp_path / root / "cache_package"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (package / "dependency.py").write_text(f"value = {value}\n")
        path = package / "model.py"
        path.write_text(code)
        paths.append(path)

    # WHEN both sources are analyzed within the same backend's call scope.
    inferred_types = []
    with type_inference_scope():
        for path in paths:
            with module_context("cache_package.model", str(path)):
                module = libcst.parse_module(code)
                metadata = libcst.MetadataWrapper(module, unsafe_skip_copy=True).resolve(
                    TypeProvider
                )
                assignment = next(filter_nodes_by_type(module, libcst.Assign))
                inferred_types.append(metadata[assignment.value])

    # THEN each import resolves against its own root, without stale or unknown types.
    integer, text = inferred_types
    assert integer.is_subtype("builtins.int")
    assert not integer.is_subtype("builtins.str")
    assert text.is_subtype("builtins.str")
    assert not text.is_subtype("builtins.int")


@pytest.mark.slow
def test_tensor_values_and_subclasses_are_tensor_operands() -> None:
    """Recognize concrete tensors, including subclasses, as quantizable operands.

    Autoquant quantizes tensor operands and leaves scalars alone. Arguments,
    tensor arithmetic, indexing, and tensor-returning calls must be recognized
    as tensors; Parameter and user subclasses must be too. An unrelated class
    and a scalar computation must not, so the check is not just permissive Any.
    """
    # GIVEN tensor-valued expressions, tensor subclasses, an unrelated class, and a scalar.
    module = libcst.parse_module("""
import torch

class Scaled(torch.Tensor): pass
class Other: pass

def values(x: torch.Tensor, weight: torch.nn.Parameter, scaled: Scaled, other: Other, n: int):
    return x, weight, scaled, x * n, x[0], torch.zeros(3), x.sum(), other, n * 2.0
""")

    # WHEN the selected backend infers each returned expression.
    metadata = libcst.MetadataWrapper(module, unsafe_skip_copy=True).resolve(TypeProvider)
    returned = next(filter_nodes_by_type(module, libcst.Return)).value
    assert isinstance(returned, libcst.Tuple)
    is_tensor = {
        module.code_for_node(element.value): metadata[element.value].is_subtype("torch.Tensor")
        for element in returned.elements
    }

    # THEN only the tensor values are tensor operands.
    assert is_tensor == {
        "x": True,
        "weight": True,
        "scaled": True,
        "x * n": True,
        "x[0]": True,
        "torch.zeros(3)": True,
        "x.sum()": True,
        "other": False,
        "n * 2.0": False,
    }


@pytest.mark.slow
def test_tensor_sequences_are_distinguished_from_integer_sequences() -> None:
    """Match tensor lists and tuples, but not shape-like integer sequences.

    Operations such as torch.cat take a sequence of tensors, while integer lists
    and tensor shapes are dimensions that must not be quantized. A sequence is
    also not itself a tensor operand.
    """
    # GIVEN annotated and literal tensor sequences alongside integer sequences.
    module = libcst.parse_module("""
import torch

def values(x: torch.Tensor, weight: torch.nn.Parameter, n: int,
           tensors: list[torch.Tensor], pair: tuple[torch.Tensor, ...]):
    return tensors, pair, [x, weight], [n, 2], x.shape
""")

    # WHEN the selected backend infers each returned expression.
    metadata = libcst.MetadataWrapper(module, unsafe_skip_copy=True).resolve(TypeProvider)
    returned = next(filter_nodes_by_type(module, libcst.Return)).value
    assert isinstance(returned, libcst.Tuple)
    infos = {
        module.code_for_node(element.value): metadata[element.value]
        for element in returned.elements
    }

    # THEN only tensor sequences match Sequence[Tensor], and none is a single tensor.
    assert {name: info.is_sequence_of_subtype("torch.Tensor") for name, info in infos.items()} == {
        "tensors": True,
        "pair": True,
        "[x, weight]": True,
        "[n, 2]": False,
        "x.shape": False,
    }
    assert not any(info.is_subtype("torch.Tensor") for info in infos.values())
