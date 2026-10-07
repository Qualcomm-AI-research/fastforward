# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Check how TSP type descriptions inform Autoquant's quantization decisions.

Inputs are handwritten TSP response excerpts, not results from a running server.
These tests check our scalar/tensor classification rules; they do not validate
backend inference or execute the passes that insert quantizers.
"""

from typing import Any

import pytest

from fastforward._autoquant.tsp.types import TspTypeOperations, TypeInfo

# TSP 0.4 response excerpts: kind 3 is a class; kind 0 is a special built-in.
# Only fields used for classification are included here.
_INT = {"kind": 3, "declaration": {"name": "int"}}
_FLOAT = {"kind": 3, "declaration": {"name": "float"}}
_TENSOR = {"kind": 3, "declaration": {"name": "Tensor"}}
_UNKNOWN = {"kind": 0, "name": "unknown"}


def test_numeric_union_is_a_scalar_operand() -> None:
    """Keep an int-or-float scale eligible for use without a tensor quantizer.

    Every alternative in this union is numeric. Losing that information when
    decoding TSP types would prevent Autoquant from recognizing a scalar operand.
    """
    # GIVEN the response for `scale: int | float` in an expression like `x * scale`.
    response = {"kind": 4, "subTypes": [_INT, _FLOAT]}

    # WHEN Autoquant asks whether this operand is definitely scalar.
    scale = TypeInfo(response, TspTypeOperations())

    # THEN either branch is numeric, so the operand needs no tensor quantizer.
    assert scale.is_proven_scalar()


@pytest.mark.parametrize("other_type", [_TENSOR, _UNKNOWN], ids=["Tensor", "Unknown"])
def test_one_numeric_union_member_does_not_prove_the_whole_operand_is_scalar(
    other_type: dict[str, Any],
) -> None:
    """Require every union alternative to be scalar before skipping quantization.

    An int-or-Tensor operand may need quantization, and an int-or-unknown operand
    may hide a tensor too. Treating either union as proven scalar would allow
    Autoquant to omit a needed tensor quantizer.
    """
    # GIVEN an operand that may be an int, but may also be a tensor or uninferred value.
    response = {"kind": 4, "subTypes": [_INT, other_type]}

    # WHEN / THEN quantization cannot be skipped solely because one branch is numeric.
    operand = TypeInfo(response, TspTypeOperations())
    assert not operand.is_proven_scalar()


@pytest.mark.parametrize("name", ["any", "unknown"], ids=["Any", "Unknown"])
def test_missing_type_precision_keeps_tensor_operations_eligible_for_quantization(
    name: str,
) -> None:
    """Preserve possible tensor matches when the checker lacks type information.

    Any and unknown do not prove that an operand is scalar. Our conservative
    matching policy keeps tensor and tensor-sequence operations eligible for
    quantization; a positive match here is not proof of the runtime value's type.
    """
    # GIVEN an argument whose type checker cannot establish a concrete type.
    argument = TypeInfo({"kind": 0, "name": name}, TspTypeOperations())

    # WHEN / THEN it may be a tensor or tensor sequence, but is not proven scalar.
    assert argument.is_subtype("torch.Tensor")
    assert argument.is_sequence_of_subtype("torch.Tensor")
    assert not argument.is_proven_scalar()


@pytest.mark.parametrize(
    "element_type", [_TENSOR, {"kind": 0, "name": "any"}], ids=["list[Tensor]", "list[Any]"]
)
def test_tensor_sequences_are_distinguished_from_integer_sequences(
    element_type: dict[str, Any],
) -> None:
    """Distinguish tensor-list operands from integer shape lists and single tensors.

    Operations such as torch.cat take a sequence of tensors, while a shape list
    should not match that operand type. Unknown list elements remain possible
    tensors so missing annotations alone do not rule out quantization.
    """
    # GIVEN tensor lists (possibly list[Any] from untyped ops) and integer shape lists.
    tensors = TypeInfo(
        {"kind": 3, "declaration": {"name": "list"}, "typeArgs": [element_type]},
        TspTypeOperations(),
    )
    sizes = TypeInfo(
        {"kind": 3, "declaration": {"name": "list"}, "typeArgs": [_INT]}, TspTypeOperations()
    )

    # WHEN / THEN only the tensor list matches an operation taking Sequence[Tensor].
    assert tensors.is_sequence_of_subtype("torch.Tensor")
    assert not tensors.is_subtype("torch.Tensor")
    assert not sizes.is_sequence_of_subtype("torch.Tensor")
