# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import libcst
import pytest
import torch

from fastforward._autoquant import pysource
from fastforward._autoquant.autoquant import (
    autoquant,
    autoquant_with_defaults,
    default_source_context,
)
from fastforward._autoquant.cst import passes
from fastforward._autoquant.pysource import SourceContext
from fastforward._quantops import optable
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
        self.quantizer_x_3 = fastforward.nn.QuantizerStub()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.quantizer_x_3(x)
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
        self.quantizer_x_3 = fastforward.nn.QuantizerStub()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quantizer_x_3(x)
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
        self.quantizer_x_13 = fastforward.nn.QuantizerStub()
        self.quantizer_y_14 = fastforward.nn.QuantizerStub()
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = self.quantizer_y_14(y)
        x = self.quantizer_x_13(x)
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
        self.quantizer_x_4 = fastforward.nn.QuantizerStub()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quantizer_x_4(x)
        s = fastforward.nn.functional.positive(x, output_quantizer=self.quantizer_positive_1)
        s = fastforward.nn.functional.negative(x, output_quantizer=self.quantizer_negative_2)
        s = fastforward.nn.functional.bitwise_not(x, output_quantizer=self.quantizer_bitwise_not_3)
        return s
"""


# Example with local variable re-assignment with non-quantized functions
class ExampleModule5(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(x)
        x = self.do_something(x)
        x = self.do_something(x)
        x = torch.sigmoid(x)
        return x

    def do_something(self, x: torch.Tensor) -> torch.Tensor:
        return x


FLOAT_MODULE_5 = ExampleModule5()

AUTOQUANTIZED_MODULE_OUT_5 = """
class QuantizedExampleModule5(fastforward.nn.QuantizedModule, ExampleModule5):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.quantizer_relu_1 = fastforward.nn.QuantizerStub()
        self.quantizer_sigmoid_2 = fastforward.nn.QuantizerStub()
        self.quantizer_x_3 = fastforward.nn.QuantizerStub()
        self.quantizer_x_4 = fastforward.nn.QuantizerStub()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quantizer_x_3(x)
        x = fastforward.nn.functional.relu(x, output_quantizer=self.quantizer_relu_1)
        x = self.do_something(x)
        x = self.do_something(x)
        x = self.quantizer_x_4(x)
        x = fastforward.nn.functional.sigmoid(x, output_quantizer=self.quantizer_sigmoid_2)
        return x
"""


# Example with sub- and sub-sub-modules that do not have manually quantized counterparts
class ExampleSubModule6(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.module = torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.module(x)
        return x


class ExampleModule6(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.module = ExampleSubModule6()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.module(x)
        return x


FLOAT_MODULE_6 = ExampleModule6()

AUTOQUANTIZED_MODULE_OUT_6 = """
class QuantizedExampleModule6(fastforward.nn.QuantizedModule, ExampleModule6):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.module(x)
        return x
class QuantizedIdentity(fastforward.nn.QuantizedModule, Identity):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        
    def forward(self, input: Tensor) -> Tensor:
        return input
class QuantizedExampleSubModule6(fastforward.nn.QuantizedModule, ExampleSubModule6):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.module(x)
        return x
"""


# Example with manually quantized counterparts
class ExampleModule7(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.module = torch.nn.Linear(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.module(x)
        return x


FLOAT_MODULE_7 = ExampleModule7()

AUTOQUANTIZED_MODULE_OUT_7 = """
class QuantizedExampleModule7(fastforward.nn.QuantizedModule, ExampleModule7):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.module(x)
        return x
"""


@pytest.mark.slow
@pytest.mark.parametrize(
    "input_module, expected_codegen",
    [
        (FLOAT_MODULE_1, AUTOQUANTIZED_MODULE_OUT_1),
        (FLOAT_MODULE_2, AUTOQUANTIZED_MODULE_OUT_2),
        (FLOAT_MODULE_3, AUTOQUANTIZED_MODULE_OUT_3),
        (FLOAT_MODULE_4, AUTOQUANTIZED_MODULE_OUT_4),
        (FLOAT_MODULE_5, AUTOQUANTIZED_MODULE_OUT_5),
        (FLOAT_MODULE_6, AUTOQUANTIZED_MODULE_OUT_6),
        (FLOAT_MODULE_7, AUTOQUANTIZED_MODULE_OUT_7),
    ],
    ids=[f"case-{i}" for i in range(1, 8)],
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
            passes.WrapAssignments(),
        ]
    )

    # WHEN we autoquantize the example module
    module_builder = autoquant(
        module=input_module, source_context=source_context, operator_table=operator_table
    )
    actual_output = module_builder.build().code

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
    actual = autoquant_with_defaults(ExampleExpression()).build().code
    (expected_output,) = dedent_strip(EXPECTED_OUTPUT)
    assert_strings_match_verbose(expected_output, actual.strip())
