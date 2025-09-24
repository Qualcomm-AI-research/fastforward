# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


import contextlib
import pathlib
import sys
import types

from typing import Any, Iterable, Iterator, TypeAlias
from unittest.mock import patch

import fastforward
import libcst
import pytest
import torch

from fastforward._autoquant import pybuilder, pysource
from fastforward._autoquant.autoquant import (
    _find_known_quantized_modules,
    autoquant,
    autoquant_with_defaults,
    codeformat_with_defaults,
    default_source_context,
)
from fastforward._autoquant.cst import passes
from fastforward._autoquant.pysource import SourceContext
from fastforward._quantops import OperatorTable, optable
from fastforward.autoquant import autoquantize
from fastforward.testing.string import assert_strings_match_verbose, dedent_strip
from torch import Tensor as TensorAlias  # required for tests, do not remove
from typing_extensions import override

Tensor: TypeAlias = torch.Tensor  # Required for tests, do not remove


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
    source_context = default_source_context(use_type_inference=False)

    # WHEN a CST is obtained through default source context
    cst = source_context.get_cst("torch.nn.modules.conv", NodeType=libcst.Module)

    # THEN, the must be no Assign, AnAssign, or AugAssign in the
    # CST anymore.
    _ = cst.visit(_AssertNoAssignments())

    # GIVEN a source context that does not remove assignment nodes
    source_context = SourceContext()

    # WHEN a CST is obtained through this source context
    cst = source_context.get_cst("torch.nn.modules.conv", NodeType=libcst.Module)

    # THEN it is expected that the CST contains no assignment nodes
    with pytest.raises(AssertionError):
        _ = cst.visit(_AssertNoAssignments())


# Example module with an __init__ function and attributes
class ExampleModule1(torch.nn.Module):
    def __init__(self, z: torch.Tensor):
        super().__init__()
        self.z = z

    def forward(self, x: torch.Tensor) -> tuple[TensorAlias, torch.Tensor]:
        y = torch.sigmoid(x)
        return self.z, torch.relu(y)


FLOAT_MODULE_1 = ExampleModule1(z=torch.tensor([0]))

AUTOQUANTIZED_MODULE_OUT_1 = """
import torch

from torch import Tensor as TensorAlias

import fastforward

from tests.autoquant.test_autoquant import ExampleModule1


class QuantizedExampleModule1(fastforward.nn.QuantizedModule, ExampleModule1):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.quantizer_sigmoid: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_relu: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_x: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()

    def forward(self, x: torch.Tensor) -> tuple[TensorAlias, torch.Tensor]:
        x = self.quantizer_x(x)
        y = fastforward.nn.functional.sigmoid(x, output_quantizer=self.quantizer_sigmoid)
        return self.z, fastforward.nn.functional.relu(y, output_quantizer=self.quantizer_relu)
"""


# Example module without __init__ function
class ExampleModule2(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.nn.functional.conv2d(x, x)
        return torch.nn.functional.linear(y, y)


FLOAT_MODULE_2 = ExampleModule2()

AUTOQUANTIZED_MODULE_OUT_2 = """
import torch

import fastforward

from tests.autoquant.test_autoquant import ExampleModule2


class QuantizedExampleModule2(fastforward.nn.QuantizedModule, ExampleModule2):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.quantizer_conv2d: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_linear: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_x: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quantizer_x(x)
        y = fastforward.nn.functional.conv2d(x, x, output_quantizer=self.quantizer_conv2d)
        return fastforward.nn.functional.linear(y, y, output_quantizer=self.quantizer_linear)
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
import torch

import fastforward

from tests.autoquant.test_autoquant import ExampleModule3


class QuantizedExampleModule3(fastforward.nn.QuantizedModule, ExampleModule3):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.quantizer_add: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_bitwise_or: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_bitwise_xor: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_div: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_floor_divide: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_bitwise_left_shift: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_matmul: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_remainder: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_mul: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_pow: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_bitwise_right_shift: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_sub: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_x: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_y: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.quantizer_x(x)
        y = self.quantizer_y(y)
        s = fastforward.nn.functional.add(x, y, output_quantizer=self.quantizer_add)
        s = fastforward.nn.functional.bitwise_or(x, y, output_quantizer=self.quantizer_bitwise_or)
        s = fastforward.nn.functional.bitwise_xor(x, y, output_quantizer=self.quantizer_bitwise_xor)
        s = fastforward.nn.functional.div(x, y, output_quantizer=self.quantizer_div)
        s = fastforward.nn.functional.floor_divide(x, y, output_quantizer=self.quantizer_floor_divide)
        s = fastforward.nn.functional.bitwise_left_shift(x, y, output_quantizer=self.quantizer_bitwise_left_shift)
        s = fastforward.nn.functional.matmul(x, y, output_quantizer=self.quantizer_matmul)
        s = fastforward.nn.functional.remainder(x, y, output_quantizer=self.quantizer_remainder)
        s = fastforward.nn.functional.mul(x, y, output_quantizer=self.quantizer_mul)
        s = fastforward.nn.functional.pow(x, y, output_quantizer=self.quantizer_pow)
        s = fastforward.nn.functional.bitwise_right_shift(x, y, output_quantizer=self.quantizer_bitwise_right_shift)
        s = fastforward.nn.functional.sub(x, y, output_quantizer=self.quantizer_sub)
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
import torch

import fastforward

from tests.autoquant.test_autoquant import ExampleModule4


class QuantizedExampleModule4(fastforward.nn.QuantizedModule, ExampleModule4):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.quantizer_positive: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_negative: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_bitwise_not: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_x: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quantizer_x(x)
        s = fastforward.nn.functional.positive(x, output_quantizer=self.quantizer_positive)
        s = fastforward.nn.functional.negative(x, output_quantizer=self.quantizer_negative)
        s = fastforward.nn.functional.bitwise_not(x, output_quantizer=self.quantizer_bitwise_not)
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
        return x * x


FLOAT_MODULE_5 = ExampleModule5()

AUTOQUANTIZED_MODULE_OUT_5 = """
import torch

import fastforward

from tests.autoquant.test_autoquant import ExampleModule5


class QuantizedExampleModule5(fastforward.nn.QuantizedModule, ExampleModule5):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.quantizer_relu: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_sigmoid: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_x_1: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_x_2: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_mul: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_x_3: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quantizer_x_2(x)
        x = fastforward.nn.functional.relu(x, output_quantizer=self.quantizer_relu)
        x = self.do_something(x)
        x = self.do_something(x)
        x = self.quantizer_x_1(x)
        x = fastforward.nn.functional.sigmoid(x, output_quantizer=self.quantizer_sigmoid)
        return x

    def do_something(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quantizer_x_3(x)
        return fastforward.nn.functional.mul(x, x, output_quantizer=self.quantizer_mul)
"""


# Example with sub- and sub-sub-modules that do not have manually quantized counterparts
class ExampleSubModule6(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.module_1 = torch.nn.Identity()
        self.module_2 = torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.module_1(x)
        x = self.module_2(x)
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
import torch

from torch import Tensor
from torch.nn.modules.linear import Identity

import fastforward

from tests.autoquant.test_autoquant import ExampleModule6, ExampleSubModule6


class QuantizedExampleModule6(fastforward.nn.QuantizedModule, ExampleModule6):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.module(x)
        return x

class QuantizedExampleSubModule6(fastforward.nn.QuantizedModule, ExampleSubModule6):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.module_1(x)
        x = self.module_2(x)
        return x

class QuantizedIdentity(fastforward.nn.QuantizedModule, Identity):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()

    def forward(self, input: Tensor) -> Tensor:
        return input
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
import torch

import fastforward

from tests.autoquant.test_autoquant import ExampleModule7


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
    autoquant_code = autoquant(
        module=input_module, source_context=source_context, operator_table=operator_table
    )
    actual_code = codeformat_with_defaults(code=autoquant_code)

    # THEN the generated code is quantized as expected
    expected_output = dedent_strip(expected_codegen)[0]
    expected_output = codeformat_with_defaults(code=expected_output).strip()
    assert_strings_match_verbose(str2=expected_output, str1=actual_code.strip())


# Example with literal integer
class ExampleModule8(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        h = x.reshape((-1 + 2 // 3, self.num_features))
        h = h.reshape((999 - 12, self.num_features))
        h = h.reshape((-1, self.num_features))
        return h


FLOAT_MODULE_8 = ExampleModule8()

AUTOQUANTIZED_MODULE_OUT_8 = """
import fastforward

from tests.autoquant.test_autoquant import ExampleModule8, Tensor


class QuantizedExampleModule8(fastforward.nn.QuantizedModule, ExampleModule8):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()

    def forward(self, x: Tensor) -> Tensor:
        h = x.reshape((-1 + 2 // 3, self.num_features))
        h = h.reshape((999 - 12, self.num_features))
        h = h.reshape((-1, self.num_features))
        return h
"""


# Example with loops
class ExampleModule9(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        for _ in range(3):
            x = torch.sigmoid(x)
        x = torch.relu(x)
        test = True
        while test:
            for _ in range(3):
                x = _my_func(x)
            test = False
        return torch.relu(x)


FLOAT_MODULE_9 = ExampleModule9()

AUTOQUANTIZED_MODULE_OUT_9 = """
import fastforward

from tests.autoquant.test_autoquant import ExampleModule9, Tensor, _my_func


class QuantizedExampleModule9(fastforward.nn.QuantizedModule, ExampleModule9):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.quantizer_sigmoid: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_relu_1: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_relu_2: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_x_1: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_x_2: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()

    def forward(self, x: Tensor) -> Tensor:
        x = self.quantizer_x_2(x)
        for _ in range(3):
            x = fastforward.nn.functional.sigmoid(x, output_quantizer=self.quantizer_sigmoid)
        x = fastforward.nn.functional.relu(x, output_quantizer=self.quantizer_relu_1)
        test = True
        while test:
            for _ in range(3):
                x = _my_func(x)
                x = self.quantizer_x_1(x)
            test = False
        return fastforward.nn.functional.relu(x, output_quantizer=self.quantizer_relu_2)
"""


# Example with quantized loop variables
class ExampleModule10(torch.nn.Module):
    def forward(self, x: Tensor) -> Any:
        for a in x:
            _ = torch.relu(a)
        for a in x:
            if 0 > 0:
                break
            _ = torch.relu(a)
        for a in x:
            if 0 > 0:
                continue
            return
        return x


FLOAT_MODULE_10 = ExampleModule10()

AUTOQUANTIZED_MODULE_OUT_10 = """
from typing import Any

import fastforward

from tests.autoquant.test_autoquant import ExampleModule10, Tensor


class QuantizedExampleModule10(fastforward.nn.QuantizedModule, ExampleModule10):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.quantizer_relu_1: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_relu_2: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_a_1: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_a_2: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()

    def forward(self, x: Tensor) -> Any:
        for a in x:
            a = self.quantizer_a_1(a)
            _ = fastforward.nn.functional.relu(a, output_quantizer=self.quantizer_relu_1)
        for a in x:
            a = self.quantizer_a_2(a)
            if 0 > 0:
                break
            _ = fastforward.nn.functional.relu(a, output_quantizer=self.quantizer_relu_2)
        for a in x:
            if 0 > 0:
                continue
            return
        return x
"""


# Example with `with` statement
class ExampleModule11(torch.nn.Module):
    def forward(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        with _my_context(x) as xx, _my_context(y) as yy, _my_context(x):
            if True:
                pass
            out1 = torch.relu(xx)
            out2 = torch.sigmoid(yy)

        return out1, out2


FLOAT_MODULE_11 = ExampleModule11()

AUTOQUANTIZED_MODULE_OUT_11 = """
import fastforward

from tests.autoquant.test_autoquant import ExampleModule11, Tensor, _my_context


class QuantizedExampleModule11(fastforward.nn.QuantizedModule, ExampleModule11):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.quantizer_relu: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_sigmoid: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_xx: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_yy: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()

    def forward(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        with _my_context(x) as xx, _my_context(y) as yy, _my_context(x):
            xx = self.quantizer_xx(xx)
            yy = self.quantizer_yy(yy)
            if True:
                pass
            out1 = fastforward.nn.functional.relu(xx, output_quantizer=self.quantizer_relu)
            out2 = fastforward.nn.functional.sigmoid(yy, output_quantizer=self.quantizer_sigmoid)

        return out1, out2
"""


# Example with docstring and empty assignment
class FloatModule12(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """I am an important docstring.

        How multiline of me!
        """
        y: Tensor
        y = torch.zeros([0])
        return x + y


FLOAT_MODULE_12 = FloatModule12()
AUTOQUANTIZED_MODULE_OUT_12 = '''
import fastforward
import torch

from tests.autoquant.test_autoquant import FloatModule12


class QuantizedFloatModule12(fastforward.nn.QuantizedModule, FloatModule12):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.quantizer_add: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_y: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_x: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """I am an important docstring.
        
        How multiline of me!
        """
        x = self.quantizer_x(x)
        y: Tensor
        y = torch.zeros([0])
        y = self.quantizer_y(y)
        return fastforward.nn.functional.add(x, y, output_quantizer=self.quantizer_add)
'''


# Example with comprehensions
class ExampleModule13(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[Any]]:
        return [torch.relu(z) for y in x for z in y], [x + x for x in x if x > 0]


FLOAT_MODULE_13 = ExampleModule13()

AUTOQUANTIZED_MODULE_OUT_13 = """
from typing import Any

import fastforward
import torch

from tests.autoquant.test_autoquant import ExampleModule13


class QuantizedExampleModule13(fastforward.nn.QuantizedModule, ExampleModule13):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.quantizer_relu: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_add: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_z: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_x_1: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_x_2: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[Any]]:
        x = self.quantizer_x_2(x)
        return [
            fastforward.nn.functional.relu(
                self.quantizer_z(z), output_quantizer=self.quantizer_relu
            )
            for y in x
            for z in y
        ], [
            fastforward.nn.functional.add(
                (__ff0 := self.quantizer_x_1(x)), __ff0, output_quantizer=self.quantizer_add
            )
            for x in x
            if x > 0
        ]
"""


# Example with with statement
class ExampleModule14(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> Iterable[Tensor]:
        if 0 > 0:
            return {torch.relu(y) for y in x}
        elif 0 == 0:
            return (torch.relu(y) for y in x)
        else:
            return {torch.relu(y): torch.sigmoid(z) for y, z in x}


FLOAT_MODULE_14 = ExampleModule14()

AUTOQUANTIZED_MODULE_OUT_14 = """
from typing import Iterable

import fastforward
import torch

from tests.autoquant.test_autoquant import ExampleModule14, Tensor


class QuantizedExampleModule14(fastforward.nn.QuantizedModule, ExampleModule14):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.quantizer_relu_1: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_relu_2: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_relu_3: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_sigmoid: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_y_1: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_y_2: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_y_3: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_z: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()

    def forward(self, x: torch.Tensor) -> Iterable[Tensor]:
        if 0 > 0:
            return {
                fastforward.nn.functional.relu(
                    self.quantizer_y_1(y), output_quantizer=self.quantizer_relu_1
                )
                for y in x
            }
        elif 0 == 0:
            return (
                fastforward.nn.functional.relu(
                    self.quantizer_y_2(y), output_quantizer=self.quantizer_relu_2
                )
                for y in x
            )
        else:
            return {
                fastforward.nn.functional.relu(
                    self.quantizer_y_3(y), output_quantizer=self.quantizer_relu_3
                ): fastforward.nn.functional.sigmoid(
                    self.quantizer_z(z), output_quantizer=self.quantizer_sigmoid
                )
                for y, z in x
            }
"""


@pytest.mark.slow
@pytest.mark.parametrize(
    "input_module, expected_codegen",
    [
        (FLOAT_MODULE_8, AUTOQUANTIZED_MODULE_OUT_8),
        (FLOAT_MODULE_9, AUTOQUANTIZED_MODULE_OUT_9),
        (FLOAT_MODULE_10, AUTOQUANTIZED_MODULE_OUT_10),
        (FLOAT_MODULE_11, AUTOQUANTIZED_MODULE_OUT_11),
        (FLOAT_MODULE_12, AUTOQUANTIZED_MODULE_OUT_12),
        (FLOAT_MODULE_13, AUTOQUANTIZED_MODULE_OUT_13),
        (FLOAT_MODULE_14, AUTOQUANTIZED_MODULE_OUT_14),
    ],
    ids=[f"case-{i}" for i in range(8, 15)],
)
def test_autoquant_end_to_end(input_module: torch.nn.Module, expected_codegen: str) -> None:
    """Verifies autoquantization introduces the magic method and quantizers."""
    # GIVEN a torch module with a forward pass and quantizable function calls

    # GIVEN the default operator table
    operator_table = optable.OperatorTable.from_yaml(
        alias_extensions=optable.STR_ALIASES_EXTENSIONS
    )

    # GIVEN a default SourceContext
    source_context = default_source_context(use_type_inference=False)

    # WHEN we autoquantize the example module
    autoquantized = autoquant(
        module=input_module, source_context=source_context, operator_table=operator_table
    )
    actual_output = codeformat_with_defaults(code=autoquantized).strip()

    # THEN the generated code is quantized as expected
    expected_output = dedent_strip(expected_codegen)[0]
    assert_strings_match_verbose(str2=expected_output, str1=actual_output.strip())


class ExampleExpression(torch.nn.Module):
    def forward(self) -> None:
        print("This is not a quantized function.")


EXPECTED_OUTPUT = """
import fastforward

from tests.autoquant.test_autoquant import ExampleExpression


class QuantizedExampleExpression(fastforward.nn.QuantizedModule, ExampleExpression):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
    def forward(self) -> None:
        print("This is not a quantized function.")
"""


@pytest.mark.slow
def test_expressions_not_quantized() -> None:
    """Tests that expressions are not quantized (fixes #80)."""
    actual = autoquant_with_defaults(ExampleExpression(), use_type_inference=False)
    actual = codeformat_with_defaults(code=actual).strip()

    (expected_output,) = dedent_strip(EXPECTED_OUTPUT)

    expected_output = codeformat_with_defaults(code=expected_output).strip()
    assert_strings_match_verbose(expected_output, actual.strip())


UNFORMATTED_CODE = """
1 + 2 ==3
"""

FORMATTED_CODE = """
1 + 2 == 3
"""


def test_codeformat() -> None:
    """Tests that code is formatted correctly."""
    input = UNFORMATTED_CODE
    expected = FORMATTED_CODE.strip()
    actual = codeformat_with_defaults(code=input).strip()
    assert_strings_match_verbose(expected, actual)


class ExampleModule1B(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.nn.functional.conv2d(x, x)
        return torch.nn.functional.linear(y, y)


FLOAT_MODULE_1B = ExampleModule1B()

AUTOQUANTIZED_MODULE_OUT_1B = """
import fastforward
import torch

from tests.autoquant.test_autoquant import ExampleModule1B


class QuantizedExampleModule1B(fastforward.nn.QuantizedModule, ExampleModule1B):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.quantizer_conv2d: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_linear: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_x: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quantizer_x(x)
        y = fastforward.nn.functional.conv2d(x, x, output_quantizer=self.quantizer_conv2d)
        return fastforward.nn.functional.linear(y, y, output_quantizer=self.quantizer_linear)

"""


@pytest.mark.slow
def test_autoquant_emits_code_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    # GIVEN a torch module to autoquantize
    input = FLOAT_MODULE_1B
    # GIVEN expected atuqoantzied code
    (expected_output,) = dedent_strip(AUTOQUANTIZED_MODULE_OUT_1B)

    # WHEN we autoquantize the example code and write it with the default code writer
    autoquantized = autoquantize(
        module=input,
        code_writer=pybuilder.StdoutWriter(module_name="dummy_name"),
        auto_import=False,
        use_type_inference=False,
    )
    actual_output = autoquantized.code

    # WHEN we inspect the code written to stdout
    captured = capsys.readouterr()

    # THEN the the only code written to stdout is the expected code
    assert_strings_match_verbose(captured.out.strip(), expected_output.strip())

    # THEN the output matches our expectation
    assert_strings_match_verbose(actual_output.strip(), expected_output)


@pytest.mark.slow
def test_autoquant_writes_to_file(tmp_path: pathlib.Path) -> None:
    # GIVEN a torch module to autoquantize
    input = FLOAT_MODULE_1B
    # GIVEN expected autoquantized code
    (expected_output,) = dedent_strip(AUTOQUANTIZED_MODULE_OUT_1B)
    # GIVEN a target output file to write the code to
    out_dir = tmp_path / "some_nested_dir"

    # WHEN we autoquantize the example code and write it to a folder
    autoquantize(
        module=input,
        output_path=out_dir,
        auto_import=False,
        use_type_inference=False,
    )

    # THEN the output directory was created
    assert out_dir.is_dir()
    # THEN `__init__.py` was created in the output directory
    out_file = out_dir / "__init__.py"
    assert out_file.is_file()

    actual_output = out_file.read_text(encoding="utf-8")

    # THEN the file's contents matches our expectation
    assert_strings_match_verbose(actual_output.strip(), expected_output)

    # WHEN we try to overwrite the existing `__init__.py` file
    with pytest.raises(FileExistsError, match="already exists. Use `force_overwrite=True`"):
        # THEN we get a sensible error
        autoquantize(
            module=input,
            output_path=out_dir,
            auto_import=False,
            use_type_inference=False,
        )

    # WHEN we try to overwrite the `__init__.py` file with force-overwrite
    # THEN no error is thrown
    autoquantize(
        module=input,
        output_path=out_dir,
        force_overwrite=True,
        auto_import=False,
        use_type_inference=False,
    )

    # WHEN we autoquantize the example code and specify to write to a `.py`-file instead
    module_name = "some_file.py"
    output_path = tmp_path / module_name
    autoquantize(
        module=input,
        output_path=output_path,
        auto_import=False,
        use_type_inference=False,
    )
    # THEN we wrote to the `some_file.py`, and not to `__init__.py`
    assert output_path.is_file()


@pytest.mark.slow
@patch.dict("sys.modules")
def test_auto_import(tmp_path: pathlib.Path) -> None:
    # GIVEN a torch module
    input = FLOAT_MODULE_1B
    module_name = "ff_quant"
    output_path = tmp_path / module_name

    # WHEN we take note of the original system modules
    sys_modules = dict(sys.modules)

    # WHEN we autoquantize the example code
    # WHEN we add some additional imports (until this feature is added)
    autoquantized_code = autoquantize(
        module=input,
        output_path=output_path,
        auto_import=True,
        use_type_inference=False,
    )

    # THEN the system modules are extended by exactly one module
    new_modules = list(set(sys.modules.values()) - set(sys_modules.values()))
    assert len(new_modules) == 1
    # THEN the new module is ff_quant
    assert new_modules[0] == autoquantized_code.pymodule

    # THEN the module name is `__init__.py` default module name
    assert autoquantized_code.pymodule_name == module_name

    # THEN the generated module is a Python module
    pymodule = autoquantized_code.pymodule
    assert isinstance(pymodule, types.ModuleType)

    # THEN autoquant extends the system modules and we can import ff_quant
    import ff_quant as ff_quant  # type:ignore[import-not-found] # noqa: I001

    # THEN  `import` from the generated module works, too
    from ff_quant import (  # noqa: I001
        QuantizedExampleModule1B as QuantizedModule,
    )

    # THEN the quantized module is available in the generated Python module
    assert hasattr(pymodule, f"Quantized{ExampleModule1B.__name__}")
    assert hasattr(ff_quant, f"Quantized{ExampleModule1B.__name__}")

    # THEN the quantized module can be accessed via dot notation
    modules = [
        pymodule.QuantizedExampleModule1B,
        ff_quant.QuantizedExampleModule1B,
        QuantizedModule,
    ]

    for module in modules:
        # THEN the quantized module has expected static properties
        assert module is not None
        assert module.__name__ == f"Quantized{ExampleModule1B.__name__}"
        assert fastforward.nn.QuantizedModule in module.__bases__

        # THEN the quantized module can be instantiated
        instance = module()

        # THEN the instance has expected properties
        assert list(instance.modules())[-1].__class__ == fastforward.nn.QuantizerStub


def test_find_known_quantized_modules_subclasses_of_subclasses() -> None:
    # _find_known_quantized_modules should find all quantized modules, including
    # modules that don't directly subclass QuantizedModule. An example of this is
    # QuantizedRelu for which ReLU should be discovered.
    assert torch.nn.ReLU in _find_known_quantized_modules()


def _my_func(x: Any, *args: Any, **kwargs: Any) -> Any:
    """Used in autoquant examples."""
    del args, kwargs
    return x


@contextlib.contextmanager
def _my_context(x: Any) -> Iterator[Any]:
    yield x


# Test autoquant with custom Operator table
#
class SimpleLinearModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4, 4))
        self.bias = torch.nn.Parameter(torch.randn(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)


class CustomOpLinearModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight)


class MultiOpModule(torch.nn.Module):
    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x = torch.relu(x)
        x = torch.nn.functional.linear(x, weight)
        return torch.sigmoid(x)


def _custom_linear_dispatch(
    input: torch.Tensor,
    weight: torch.Tensor,
    *,
    quantizer_a: fastforward.nn.Quantizer,
    quantizer_b: fastforward.nn.Quantizer,
    output_quantizer: fastforward.nn.Quantizer,
) -> torch.Tensor:
    # The implementation is irrelevant for this test
    del weight, quantizer_a, quantizer_b, output_quantizer
    return input


Autoquantized_SimpleLinearModule = """
import fastforward
import torch

from tests.autoquant.test_autoquant import SimpleLinearModule


class QuantizedSimpleLinearModule(fastforward.nn.QuantizedModule, SimpleLinearModule):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.quantizer_linear: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_self_bias: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_self_weight: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_x: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quantizer_x(x)
        self_bias = self.quantizer_self_bias(self.bias)
        self_weight = self.quantizer_self_weight(self.weight)
        return fastforward.nn.functional.linear(
            x, self_weight, self_bias, output_quantizer=self.quantizer_linear
        )
"""

Autoquantized_CustomOpLinearModule = """
import fastforward
import torch

from tests.autoquant.test_autoquant import CustomOpLinearModule


class QuantizedCustomOpLinearModule(fastforward.nn.QuantizedModule, CustomOpLinearModule):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.quantizer_a: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_b: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_linear: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_self_weight: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_x: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quantizer_x(x)
        self_weight = self.quantizer_self_weight(self.weight)
        return tests.autoquant.test_autoquant._custom_linear_dispatch(
            x,
            self_weight,
            quantizer_a=self.quantizer_a,
            quantizer_b=self.quantizer_b,
            output_quantizer=self.quantizer_linear,
        )
"""

Autoquantized_MultiOpModule = """
import fastforward
import torch

from tests.autoquant.test_autoquant import MultiOpModule


class QuantizedMultiOpModule(fastforward.nn.QuantizedModule, MultiOpModule):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.quantizer_relu: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_a: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_b: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_linear: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_sigmoid: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_x: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()
        self.quantizer_weight: fastforward.nn.Quantizer = fastforward.nn.QuantizerStub()

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x = self.quantizer_x(x)
        weight = self.quantizer_weight(weight)
        x = fastforward.nn.functional.relu(x, output_quantizer=self.quantizer_relu)
        x = tests.autoquant.test_autoquant._custom_linear_dispatch(
            x,
            weight,
            quantizer_a=self.quantizer_a,
            quantizer_b=self.quantizer_b,
            output_quantizer=self.quantizer_linear,
        )
        return fastforward.nn.functional.sigmoid(x, output_quantizer=self.quantizer_sigmoid)
"""


@pytest.mark.slow
@pytest.mark.parametrize(
    "module,expected_code",
    [
        (SimpleLinearModule(), Autoquantized_SimpleLinearModule),
        (CustomOpLinearModule(), Autoquantized_CustomOpLinearModule),
        (MultiOpModule(), Autoquantized_MultiOpModule),
    ],
    ids=["SimpleLinearModule", "CustomOpLinearModule", "MultiOpModule"],
)
def test_autoquant_with_overloaded_operator_table(
    module: torch.nn.Module, expected_code: str
) -> None:
    # GIVEN a simple module that uses torch.nn.functional.linear

    # GIVEN a custom operator table loaded from YAML and an overloaded linear operator
    custom_optable = OperatorTable.from_yaml(alias_extensions=optable.STR_ALIASES_EXTENSIONS)
    custom_schema = "linear(input: Quantized, weight: Quantized) -> Quantized"
    custom_optable.add(
        custom_schema,
        torch.nn.functional.linear,
        dispatch_op=_custom_linear_dispatch,
        intermediate_quantizers=("quantizer_a", "quantizer_b"),
    )

    # GIVEN a default source context
    source_context = default_source_context(use_type_inference=False)

    # WHEN we autoquantize the module with the custom operator table
    autoquant_code = autoquant(
        module=module,
        source_context=source_context,
        operator_table=custom_optable,
    )

    # THEN the generated code must match expectations
    actual_code = codeformat_with_defaults(code=autoquant_code)
    fastforward.testing.string.assert_strings_match_verbose(
        actual_code.strip(), expected_code.strip()
    )
