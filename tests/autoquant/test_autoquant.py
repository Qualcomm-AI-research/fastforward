# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import libcst
import pytest
import torch

from fastforward._quantops import optable
from fastforward.autoquant import pysource
from fastforward.autoquant.autoquant import _autoquant, default_source_context
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


class ExampleModule1(torch.nn.Module):
    def __init__(self, x: torch.Tensor):
        super().__init__()
        self.x = x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.sigmoid(x)
        return torch.relu(y)


AUTOQUANTIZED_MODULE_OUT_1 = """
class QuantizedExampleModule1(fastforward.nn.QuantizedModule, ExampleModule1):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.quantizer_sigmoid_1 = fastforward.nn.QuantizerStub()
        self.quantizer_relu_2 = fastforward.nn.QuantizerStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = fastforward.nn.functional.sigmoid(x, output_quantizer=self.quantizer_sigmoid_1)
        return fastforward.nn.functional.relu(y, output_quantizer=self.quantizer_relu_2)
"""


class ExampleModule2(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.nn.functional.conv2d(x, x)
        return torch.nn.functional.linear(y, y)


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


@pytest.mark.parametrize(
    "input_module, expected_codegen",
    [
        (ExampleModule1(x=torch.tensor([0])), AUTOQUANTIZED_MODULE_OUT_1),
        (ExampleModule2(), AUTOQUANTIZED_MODULE_OUT_2),
    ],
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
