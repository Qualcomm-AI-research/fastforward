import functools
import inspect
import textwrap

from typing import Callable, Optional, Sequence, cast

import libcst
import libcst.display
import libcst.matchers
import torch

import fastforward as ff

from fastforward._quantops import optable

from . import pysource
from .cst import passes

# TODO:
# 1. bases classes


def _validate_correct_module(
    torch_module: type[torch.nn.Module], py_module_name: str, module_cst: libcst.Module
) -> None:
    module_src = textwrap.dedent(inspect.getsource(torch_module))
    expected_cst = cast(libcst.ClassDef, libcst.parse_module(module_src).body[0])

    observed_csts = libcst.matchers.findall(
        module_cst,
        libcst.matchers.ClassDef(libcst.matchers.Name(torch_module.__name__)),
    )

    if len(observed_csts) == 0:
        raise ff.exceptions.AutoquantError(
            f"{py_module_name} does not define a class '{torch_module.__name__}'"
        )
    if len(observed_csts) > 1:
        raise ff.exceptions.AutoquantError(
            f"{py_module_name} defines multiple '{torch_module.__name__}', this is not supported"
        )
    observed_cst = observed_csts[0]

    if not isinstance(observed_cst, libcst.ClassDef):
        raise TypeError("pass")

    try:
        assert expected_cst.name.deep_equals(observed_cst.name)
        assert expected_cst.body.deep_equals(observed_cst.body)
        for left, right in zip(expected_cst.bases, observed_cst.bases):
            assert left.deep_equals(right)
        for left, right in zip(expected_cst.keywords, observed_cst.keywords):
            assert left.deep_equals(right)
    except AssertionError as e:
        raise ff.exceptions.AutoquantError(
            f"Found class '{torch_module.__name__}' in module '{py_module_name}', but its source "
            f"does not match with that of the provided {torch_module}"
        ) from e


def _parse_pymodule_for_torch_module(
    module: torch.nn.Module,
    validators: Sequence[Callable[[libcst.Module], None]] = (),
    preprocessors: Sequence[libcst.CSTTransformer] = (),
) -> pysource.PySourceModule:
    if not (py_module := inspect.getmodule(type(module))):
        raise RuntimeError(f"Cannot infer module of {module.__class__}")

    validators = [
        functools.partial(_validate_correct_module, type(module), py_module.__name__)
    ] + list(validators)

    preprocessors = [
        passes.SimpleStatementSuiteToIndentedBlock(),
        passes.MarkReplacementCandidates(),
        passes.IsolateReplacementCandidates(),
    ] + list(preprocessors)

    return pysource.PySourceModule(py_module, validators=validators, preprocessors=preprocessors)


def autoquant(
    module: torch.nn.Module, operator_table: Optional[optable.OperatorTable] = None
) -> None:
    operator_table = operator_table or optable.OperatorTable.from_yaml(
        alias_extensions=optable.STR_ALIASES_EXTENSIONS
    )
    _autoquant(module, operator_table)


def _autoquant(module: torch.nn.Module, operator_table: optable.OperatorTable) -> None:
    src_module = _parse_pymodule_for_torch_module(module)

    src_class = src_module.get_member(type(module).__name__)
    # TODO: setup imports correctly
    dst_class = pysource.ClassBuilder(
        f"Quantized{src_class.name}",
        bases=("fastforward.nn.QuantizedModule", src_class.name),
    )

    # TODO: make forward lookup more robust
    forward_src = src_module.get_member(src_class.name).forward
    dst_class.add_method(pysource.convert_method(forward_src, dst_class, operator_table))
    print(libcst.Module([dst_class.build()]).code)
