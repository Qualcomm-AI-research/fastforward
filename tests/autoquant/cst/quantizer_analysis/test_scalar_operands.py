# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""The GR00T scalar regression, independent of a model or graph implementation."""

import textwrap

import fastforward as ff
import libcst
import pytest

from fastforward._autoquant.cst import nodes
from fastforward._autoquant.cst.passes import WrapAssignments
from fastforward._autoquant.cst.quantizer_analysis.annotator import QuantizationAnnotationProvider
from fastforward._quantops.operator import Operator
from typing_extensions import override


def annotated_targets(source: str, *, allows_scalar: bool = True) -> set[str]:
    """Supply stored scalar proofs directly, without invoking a type checker."""
    specification = "add(input: QuantizedTensor, other: QuantizedTensor"
    specification += " | float" if allows_scalar else ""
    operator = Operator.from_spec(specification + ") -> QuantizedTensor", fallback="torch.add")

    class MarkFacts(libcst.CSTTransformer):
        @override
        def leave_BinaryOperation(
            self, original_node: libcst.BinaryOperation, updated_node: libcst.BinaryOperation
        ) -> libcst.BaseExpression:
            del original_node
            # These fixture subtractions stand for expressions proven numeric by mypy.
            if isinstance(updated_node.operator, libcst.Subtract):
                return nodes.ScalarExpression(updated_node)
            return updated_node

        @override
        def leave_Call(self, original_node: libcst.Call, updated_node: libcst.Call) -> libcst.Call:
            del original_node
            if (
                isinstance(updated_node.func, libcst.Name)
                and updated_node.func.value == "quantized_add"
            ):
                return nodes.QuantizedCall(
                    **nodes.node_asdict(updated_node), original_name="torch.add", operator=operator
                )
            return updated_node

    module = (
        libcst.parse_module(textwrap.dedent(source)).visit(WrapAssignments()).visit(MarkFacts())
    )
    annotations = libcst.MetadataWrapper(module, unsafe_skip_copy=True).resolve(
        QuantizationAnnotationProvider
    )
    return {annotation.target for values in annotations.values() for annotation in values}


def test_nested_loop_scalar_demand_does_not_change_assignment_status() -> None:
    # GIVEN: Qwen's two scalar definitions, revisited during nested-loop analysis.
    source = """
    def forward(tensor, length, flag):
        for _ in range(2):
            for _ in range(2):
                text_len = length - 1
                result = quantized_add(tensor, text_len)
            if flag:
                text_len = length - 2
        return result
    """

    # WHEN: The tensor addition accepts a scalar alternative for text_len.
    targets = annotated_targets(source)

    # THEN: Analysis succeeds without promoting either scalar producer to quantized.
    assert "text_len" not in targets
    assert "tensor" in targets


@pytest.mark.parametrize("allows_scalar", [False, True])
def test_scalar_proof_requires_a_schema_alternative(allows_scalar: bool) -> None:
    # GIVEN: A numeric producer whose consuming schema may require a tensor.
    source = """
    def forward(tensor, length):
        value = length - 1
        return quantized_add(tensor, value)
    """

    # WHEN: Derive quantization requirements from the schema and stored facts.
    targets = annotated_targets(source, allows_scalar=allows_scalar)

    # THEN: Numeric proof alone does not override a tensor-only contract.
    assert ("value" in targets) is not allows_scalar


@pytest.mark.parametrize("value", ["unknown()", "tensor"])
def test_unproven_or_mixed_operand_still_requires_conversion(value: str) -> None:
    # GIVEN: One branch is proven numeric; the other has no such proof.
    source = f"""
    def forward(tensor, length, flag):
        if flag:
            value = length - 1
        else:
            value = {value}
        return quantized_add(tensor, value)
    """

    # WHEN: The operator accepts tensors and scalars.
    targets = annotated_targets(source)

    # THEN: A scalar alternative cannot suppress conversion of the other branch.
    assert "value" in targets


@pytest.mark.slow
def test_autoquant_preserves_nested_loop_scalar_operand() -> None:
    # GIVEN: The reduced Qwen pattern with real stored mypy facts and preprocessing.
    source = """
    def forward(self, tensor: torch.Tensor, length: int, flag: bool):
        for _ in range(2):
            for _ in range(2):
                text_len = length - 1
                result = tensor + text_len
            if flag:
                text_len = length - 2
        return result
    """

    # WHEN: Run the production preprocessing, annotation and code generation passes.
    code = ff.testing.autoquant.autoquantize_str(source, use_type_inference=True)

    # THEN: The addition is converted and both text lengths remain ordinary scalars.
    assert isinstance(code, str)
    assert "fastforward.nn.functional.add" in code
    assert "quantizer_text_len" not in code


@pytest.mark.slow
@pytest.mark.parametrize(
    ("expression", "operator"),
    [
        ("-0.5 * x", "mul"),
        ("x & (1 << 3)", "bitwise_and"),
        ("x & len(range(n))", "bitwise_and"),
        ("x & ((1 << 3) + 1)", "bitwise_and"),
        ("-(1 + 2) * x", "mul"),
    ],
)
def test_scalar_wrappers_preserve_literal_and_call_handling(expression: str, operator: str) -> None:
    # GIVEN: Scalar literals and calls in positions that request quantization.
    source = f"""
    def forward(self, x: torch.Tensor, n: int):
        return {expression}
    """

    # WHEN: Type-aware preprocessing wraps scalar expressions, including nested ones.
    code = ff.testing.autoquant.autoquantize_str(source, use_type_inference=True)

    # THEN: Conversion only adds quantizers for the tensor input and operator output.
    assert isinstance(code, str)
    assert "x = self.quantizer_x(x)" in code
    assert f"fastforward.nn.functional.{operator}" in code
    assert f"output_quantizer=self.quantizer_{operator}" in code
    assert code.count("self.quantizer_") == 2
