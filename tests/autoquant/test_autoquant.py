# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import libcst
import pytest
import torch

from fastforward._quantops import optable
from fastforward.autoquant import pysource
from fastforward.autoquant.autoquant import (
    _autoquant,
    _autoquant_with_defaults,
    default_source_context,
)
from fastforward.autoquant.cst import passes
from fastforward.autoquant.pysource import SourceContext
from typing_extensions import override

from tests.utils.string import assert_strings_match_verbose, dedent_strip


class _AssertNoAssignments(libcst.CSTVisitor):
    @override
    def visit_Assign(self, node: libcst.Assign) -> bool | None:
        assert False, "CST contains Assign node"

    @override
    def visit_AugAssign(self, node: libcst.AugAssign) -> bool | None:
        assert False, "CST contains AugAssign node"

    @override
    def visit_AnnAssign(self, node: libcst.AnnAssign) -> bool | None:
        assert False, "CST contains AnnAssign node"


@pytest.mark.slow
def test_default_source_context_wraps_assignment_nodes() -> None:
    # GIVEN the default source context
    source_context = default_source_context()

    # WHEN a CST is obtained through default source context
    cst = source_context.get_cst("torch.nn.modules.conv", NodeType=libcst.Module)

    # THEN, the must be no Assign, AnAssign, or AugAssign in the
    # CST anymore.
    _ = cst.visit(_AssertNoAssignments())

    # GIVEN a source context that does not remove assignment nodes
    source_context = SourceContext()

    # WHEN a CST is obtained through this source context
    cst = source_context.get_cst("torch.nn.modules.conv", NodeType=libcst.Module)

    # THEN it is expected that the CST contains assignment nodes
    with pytest.raises(AssertionError):
        _ = cst.visit(_AssertNoAssignments())


# Example module with an __init__ function and attributes
class ExampleModule1(torch.nn.Module):
    def __init__(self, z: torch.Tensor):
        super().__init__()
        self.z = z

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = torch.sigmoid(x)
        return self.z, torch.relu(y)


FLOAT_MODULE_1 = ExampleModule1(z=torch.tensor([0]))

AUTOQUANTIZED_MODULE_OUT_1 = """
class QuantizedExampleModule1(fastforward.nn.QuantizedModule, ExampleModule1):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.quantizer_sigmoid_1 = fastforward.nn.QuantizerStub()
        self.quantizer_relu_2 = fastforward.nn.QuantizerStub()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = fastforward.nn.functional.sigmoid(x, output_quantizer=self.quantizer_sigmoid_1)
        return self.z, fastforward.nn.functional.relu(y, output_quantizer=self.quantizer_relu_2)
"""


# Example module without __init__ function
class ExampleModule2(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.nn.functional.conv2d(x, x)
        return torch.nn.functional.linear(y, y)


FLOAT_MODULE_2 = ExampleModule2()

AUTOQUANTIZED_MODULE_OUT_2 = """
class QuantizedExampleModule2(fastforward.nn.QuantizedModule, ExampleModule2):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.quantizer_conv2d_1 = fastforward.nn.QuantizerStub()
        self.quantizer_linear_2 = fastforward.nn.QuantizerStub()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = fastforward.nn.functional.conv2d(x, x, output_quantizer=self.quantizer_conv2d_1)
        return fastforward.nn.functional.linear(y, y, output_quantizer=self.quantizer_linear_2)
"""


# Example module with binary operators
class ExampleModule3(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        s = x + y
        s = x | y
        s = x ^ y
        s = x / y
        s = x // y
        s = x << y
        s = x @ y
        s = x % y
        s = x * y
        s = x**y
        s = x >> y
        s = x - y
        return s


FLOAT_MODULE_3 = ExampleModule3()

AUTOQUANTIZED_MODULE_OUT_3 = """
class QuantizedExampleModule3(fastforward.nn.QuantizedModule, ExampleModule3):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.quantizer_add_1 = fastforward.nn.QuantizerStub()
        self.quantizer_bitwise_or_2 = fastforward.nn.QuantizerStub()
        self.quantizer_bitwise_xor_3 = fastforward.nn.QuantizerStub()
        self.quantizer_div_4 = fastforward.nn.QuantizerStub()
        self.quantizer_floor_divide_5 = fastforward.nn.QuantizerStub()
        self.quantizer_bitwise_left_shift_6 = fastforward.nn.QuantizerStub()
        self.quantizer_matmul_7 = fastforward.nn.QuantizerStub()
        self.quantizer_remainder_8 = fastforward.nn.QuantizerStub()
        self.quantizer_mul_9 = fastforward.nn.QuantizerStub()
        self.quantizer_pow_10 = fastforward.nn.QuantizerStub()
        self.quantizer_bitwise_right_shift_11 = fastforward.nn.QuantizerStub()
        self.quantizer_sub_12 = fastforward.nn.QuantizerStub()
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        s = fastforward.nn.functional.add(x, y, output_quantizer=self.quantizer_add_1)
        s = fastforward.nn.functional.bitwise_or(x, y, output_quantizer=self.quantizer_bitwise_or_2)
        s = fastforward.nn.functional.bitwise_xor(x, y, output_quantizer=self.quantizer_bitwise_xor_3)
        s = fastforward.nn.functional.div(x, y, output_quantizer=self.quantizer_div_4)
        s = fastforward.nn.functional.floor_divide(x, y, output_quantizer=self.quantizer_floor_divide_5)
        s = fastforward.nn.functional.bitwise_left_shift(x, y, output_quantizer=self.quantizer_bitwise_left_shift_6)
        s = fastforward.nn.functional.matmul(x, y, output_quantizer=self.quantizer_matmul_7)
        s = fastforward.nn.functional.remainder(x, y, output_quantizer=self.quantizer_remainder_8)
        s = fastforward.nn.functional.mul(x, y, output_quantizer=self.quantizer_mul_9)
        s = fastforward.nn.functional.pow(x, y, output_quantizer=self.quantizer_pow_10)
        s = fastforward.nn.functional.bitwise_right_shift(x, y, output_quantizer=self.quantizer_bitwise_right_shift_11)
        s = fastforward.nn.functional.sub(x, y, output_quantizer=self.quantizer_sub_12)
        return s
"""


# Example module with unary operators
class ExampleModule4(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = +x
        s = -x
        s = ~x
        return s


FLOAT_MODULE_4 = ExampleModule4()

AUTOQUANTIZED_MODULE_OUT_4 = """
class QuantizedExampleModule4(fastforward.nn.QuantizedModule, ExampleModule4):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.quantizer_positive_1 = fastforward.nn.QuantizerStub()
        self.quantizer_negative_2 = fastforward.nn.QuantizerStub()
        self.quantizer_bitwise_not_3 = fastforward.nn.QuantizerStub()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = fastforward.nn.functional.positive(x, output_quantizer=self.quantizer_positive_1)
        s = fastforward.nn.functional.negative(x, output_quantizer=self.quantizer_negative_2)
        s = fastforward.nn.functional.bitwise_not(x, output_quantizer=self.quantizer_bitwise_not_3)
        return s
"""


@pytest.mark.slow
@pytest.mark.parametrize(
    "input_module, expected_codegen",
    [
        (FLOAT_MODULE_1, AUTOQUANTIZED_MODULE_OUT_1),
        (FLOAT_MODULE_2, AUTOQUANTIZED_MODULE_OUT_2),
        (FLOAT_MODULE_3, AUTOQUANTIZED_MODULE_OUT_3),
        (FLOAT_MODULE_4, AUTOQUANTIZED_MODULE_OUT_4),
    ],
    ids=[f"case-{i}" for i in range(1, 5)],
)
def test_autoquant_introduces_quantization_method(
    input_module: torch.nn.Module, expected_codegen: str
) -> None:
    """Verifies autoquantization introduces the magic method and quantizers."""
    # GIVEN a torch module with a forward pass and quantizable function calls

    # GIVEN the default operator table
    operator_table = optable.OperatorTable.from_yaml(
        alias_extensions=optable.STR_ALIASES_EXTENSIONS
    )

    # GIVEN a SourceContext with a minimal set of preprocessing passes
    source_context = pysource.SourceContext(
        preprocessing_passes=[
            passes.MarkReplacementCandidates(),
        ]
    )

    # WHEN we autoquantize the example module
    actual_output = _autoquant(
        module=input_module, source_context=source_context, operator_table=operator_table
    )

    # THEN the generated code is quantized as expected
    expected_output = dedent_strip(expected_codegen)[0]
    assert_strings_match_verbose(expected_output, actual_output.strip())


class ExampleExpression(torch.nn.Module):
    def forward(self) -> None:
        print("This is not a quantized function.")


EXPECTED_OUTPUT = """
class QuantizedExampleExpression(fastforward.nn.QuantizedModule, ExampleExpression):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
    def forward(self) -> None:
        print("This is not a quantized function.")
"""


def test_expressions_not_quantized() -> None:
    """Tests that expressions are not quantized (fixes #80)."""
    actual = _autoquant_with_defaults(
        ExampleExpression(),
    )
    (expected_output,) = dedent_strip(EXPECTED_OUTPUT)
    assert_strings_match_verbose(expected_output, actual.strip())
