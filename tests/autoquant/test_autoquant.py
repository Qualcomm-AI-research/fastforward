# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


import contextlib
import pathlib
import sys
import types

from typing import Any, Callable, Iterable, Iterator, TypeAlias
from unittest.mock import patch

import fastforward
import libcst
import pytest
import syrupy
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
from fastforward.testing.metrics import sqnr as metric_sqnr
from fastforward.testing.string import assert_strings_match_verbose
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


# ------------------------------------------------------------------------------


# Example module with an __init__ function and attributes
class ExampleModule1(torch.nn.Module):
    def __init__(self, z: torch.Tensor):
        super().__init__()
        self.z = z

    def forward(self, x: torch.Tensor) -> tuple[TensorAlias, torch.Tensor]:
        y = torch.sigmoid(x)
        return self.z, torch.relu(y)


# ------------------------------------------------------------------------------


# Example module without __init__ function
class ExampleModule2(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.nn.functional.conv2d(x, x)
        return torch.nn.functional.linear(y, y)


# ------------------------------------------------------------------------------


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


# ------------------------------------------------------------------------------


# Example module with unary operators
class ExampleModule4(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = +x
        s = -x
        s = ~x
        return s


# ------------------------------------------------------------------------------


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


# ------------------------------------------------------------------------------


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


# ------------------------------------------------------------------------------


# Example with manually quantized counterparts
class ExampleModule7(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.module = torch.nn.Linear(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.module(x)
        return x


# ------------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "input_module",
    [
        ExampleModule1(z=torch.tensor([0])),
        ExampleModule2(),
        ExampleModule3(),
        ExampleModule4(),
        ExampleModule5(),
        ExampleModule6(),
        ExampleModule7(),
    ],
    ids=[f"case-{i}" for i in range(1, 8)],
)
def test_autoquant_introduces_quantization_method(
    input_module: torch.nn.Module,
    snapshot: syrupy.assertion.SnapshotAssertion,
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

    # THEN the generated code matches an earlier snapshot
    assert snapshot == actual_code.strip()


# --------------------------------------------------------------------------------


# Example with literal integer
class ExampleModule8(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        h = x.reshape((-1 + 2 // 3, self.num_features))
        h = h.reshape((999 - 12, self.num_features))
        h = h.reshape((-1, self.num_features))
        return h


# --------------------------------------------------------------------------------


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


# --------------------------------------------------------------------------------


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


# --------------------------------------------------------------------------------


# Example with `with` statement
class ExampleModule11(torch.nn.Module):
    def forward(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        with _my_context(x) as xx, _my_context(y) as yy, _my_context(x):
            if True:
                pass
            out1 = torch.relu(xx)
            out2 = torch.sigmoid(yy)

        return out1, out2


# --------------------------------------------------------------------------------


# Example with docstring and empty assignment
class FloatModule12(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """I am an important docstring.

        How multiline of me!
        """
        y: Tensor
        y = torch.zeros([0])
        return x + y


# --------------------------------------------------------------------------------


# Example with comprehensions
class ExampleModule13(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[Any]]:
        return [torch.relu(z) for y in x for z in y], [x + x for x in x if x > 0]


# --------------------------------------------------------------------------------


# Example with branches
class ExampleModule14(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> Iterable[Tensor]:
        if 0 > 0:
            return {torch.relu(y) for y in x}
        elif 0 == 0:
            return (torch.relu(y) for y in x)
        else:
            return {torch.relu(y): torch.sigmoid(z) for y, z in x}


# --------------------------------------------------------------------------------


# When a decorator returns a wrapping function without using `functools.wraps` the name
# of the wrapping function is not updated. This case tests that the correct function
# (`custom_helper` and not `cusotm_decorator`) is quantized and the quantized function
# also has the decorator.


def custom_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return wrapper


@custom_decorator
def custom_helper(x: torch.Tensor) -> torch.Tensor:
    return x * 2


class ExampleModule15(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> Any:
        return custom_helper(x)


# --------------------------------------------------------------------------------

# When a helper function calls another helper function more than once, a unique set of
# quantizers is required for each call of the inner function.


def inner_helper(x: torch.Tensor) -> torch.Tensor:
    return x * 2


def outer_helper(x: torch.Tensor) -> torch.Tensor:
    return inner_helper(x) + inner_helper(x)


class ExampleModule16(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> Tensor:
        return outer_helper(x)


# --------------------------------------------------------------------------------

# When two functions are used that have the same `__name__` (here `sqnr`), multiple
# quantized functions must be created where each (except the first) has a count suffix.


def sqnr(a: torch.Tensor, b: torch.Tensor, flag: bool = True, eps: float = 1e-15) -> torch.Tensor:
    del flag, eps
    return a - b


# Example with branches
class ExampleModule17(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> Tensor:
        return sqnr(x, x) + metric_sqnr(x, x)


# --------------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "input_module",
    [
        ExampleModule8(),
        ExampleModule9(),
        ExampleModule10(),
        ExampleModule11(),
        FloatModule12(),
        ExampleModule13(),
        ExampleModule14(),
        ExampleModule15(),
        ExampleModule16(),
        ExampleModule17(),
    ],
    ids=[f"case-{i}" for i in range(8, 18)],
)
def test_autoquant_end_to_end(
    input_module: torch.nn.Module,
    snapshot: syrupy.assertion.SnapshotAssertion,
) -> None:
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

    # THEN the generated code matches an earlier snapshot
    assert snapshot == actual_output


class ExampleExpression(torch.nn.Module):
    def forward(self) -> None:
        print("This is not a quantized function.")


@pytest.mark.slow
def test_expressions_not_quantized(snapshot: syrupy.assertion.SnapshotAssertion) -> None:
    """Tests that expressions are not quantized (fixes #80)."""
    actual = autoquant_with_defaults(ExampleExpression(), use_type_inference=False)
    actual = codeformat_with_defaults(code=actual).strip()
    assert snapshot == actual


def test_codeformat() -> None:
    """Tests that code is formatted correctly."""
    input = "1 + 2 ==3"
    expected = "1 + 2 == 3"
    actual = codeformat_with_defaults(code=input).strip()
    assert_strings_match_verbose(expected, actual)


class ExampleModule1B(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.nn.functional.conv2d(x, x)
        return torch.nn.functional.linear(y, y)


@pytest.mark.slow
def test_autoquant_emits_code_to_stdout(
    capsys: pytest.CaptureFixture[str], snapshot: syrupy.assertion.SnapshotAssertion
) -> None:
    # GIVEN a torch module to autoquantize
    input = ExampleModule1B()

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

    # THEN the only code written to stdout is the snapshot
    assert snapshot(name="captured") == captured

    # THEN the output matches the snapshot
    assert snapshot(name="actual_output") == actual_output


@pytest.mark.slow
def test_autoquant_writes_to_file(
    tmp_path: pathlib.Path, snapshot: syrupy.assertion.SnapshotAssertion
) -> None:
    # GIVEN a torch module to autoquantize
    input = ExampleModule1B()
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

    # THEN the file's contents matches the snapshot
    assert snapshot == actual_output

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
    input = ExampleModule1B()
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


@pytest.mark.slow
@pytest.mark.parametrize(
    "module",
    [
        SimpleLinearModule(),
        CustomOpLinearModule(),
        MultiOpModule(),
    ],
    ids=["SimpleLinearModule", "CustomOpLinearModule", "MultiOpModule"],
)
def test_autoquant_with_overloaded_operator_table(
    module: torch.nn.Module, snapshot: syrupy.assertion.SnapshotAssertion
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

    # THEN the generated code must match the snapshot
    actual_code = codeformat_with_defaults(code=autoquant_code)
    assert snapshot == actual_code
