import inspect

from typing import Any

import fastforward as ff
import libcst
import pytest
import torch

from fastforward.autoquant import pysource


@pytest.mark.parametrize(
    "name,object_",
    [
        ("torch.nn.Linear", torch.nn.Linear),
        (("torch", "nn", "modules", "linear", "Linear"), torch.nn.Linear),
        (("torch.nn.modules.Linear"), torch.nn.Linear),
        ("fastforward.QuantizedTensor", ff.QuantizedTensor),
        (("fastforward", "QuantizedTensor"), ff.QuantizedTensor),
        (("fastforward.quantized_tensor", "QuantizedTensor"), ff.QuantizedTensor),
        ("fastforward.random.random_quantized", ff.random.random_quantized),
    ],
)
def test_source_context_get(
    name: str | tuple[str, ...],
    object_: Any,
) -> None:
    # NOTE: This test may fail because some of the symbols that are used to test
    # have been altered or (re)moved.

    def _get_cst(
        name: str | tuple[str, ...],
    ) -> libcst.ClassDef | libcst.FunctionDef | libcst.Module:
        if isinstance(name, str):
            name = (name,)

        source = source_ctx.get(name[0])
        for member in name[1:]:
            source = source.member(member)
        cst: libcst.CSTNode = source.cst()
        assert isinstance(cst, (libcst.ClassDef, libcst.FunctionDef, libcst.Module))
        return cst

    # GIVEN a SourceContext
    source_ctx = pysource.SourceContext()

    # WHEN we obtain a CST for a source symbol using the source_context
    cst = _get_cst(name)

    # WHEN we obtain a CST from the source code directly
    object_source = inspect.getsource(object_)
    object_cst = libcst.parse_statement(object_source)
    assert isinstance(object_cst, (libcst.ClassDef, libcst.FunctionDef, libcst.Module))

    # THEN the CSTs must match exactly except for the leading lines.
    altered_cst = cst.with_changes(leading_lines=object_cst.leading_lines)
    assert altered_cst.deep_equals(object_cst)

    # THEN the generated code from the CST from the source context must match
    # the ground truth source exactly.
    if isinstance(altered_cst, libcst.Module):
        module_cst = altered_cst
    else:
        module_cst = libcst.Module(body=(altered_cst,))
    assert module_cst.code == object_source


@pytest.mark.parametrize(
    "name,expected_type",
    [
        ("fastforward.QuantizedTensor", "class"),
        ("fastforward.random.random_quantized", "function"),
        ("fastforward.random", "module"),
    ],
)
def test_pysource_is_type_methods(name: str, expected_type: str) -> None:
    # NOTE: This test may fail because some of the symbols that are used to test
    # have been altered or (re)moved.

    # GIVEN a SourceContext, a qualified name for an object and its expected
    # pysource type
    source_ctx = pysource.SourceContext()

    # WHEN a pysource object is obtained
    source_object = source_ctx.get(name)

    # THEN the the `is_class`, `is_function` and `is_module` method
    # must return accordingly
    match expected_type:
        case "class":
            assert source_object.is_class()
            assert not source_object.is_function()
            assert not source_object.is_module()
        case "function":
            assert not source_object.is_class()
            assert source_object.is_function()
            assert not source_object.is_module()
        case "module":
            assert not source_object.is_class()
            assert not source_object.is_function()
            assert source_object.is_module()
        case _:
            raise ValueError(f"'{expected_type}' is not a valid expected_type")


def test_pysource_module() -> None:
    # NOTE: This test may fail because some of the symbols that are used to test
    # have been altered or (re)moved.

    source_ctx = pysource.SourceContext()

    # GIVEN a function_source and module_source
    function_source = source_ctx.get("fastforward.random.random_quantized")
    module_source = source_ctx.get("fastforward.random")

    # WHEN the module CST is obtained through the function source
    module_cst_from_function: libcst.CSTNode = function_source.module().cst()

    # THEN the corresponding CSTs refer to the same object
    assert module_cst_from_function is module_source.cst()

    # WHEN the module of module source object is obtained
    module_of_module = module_source.module()

    # THEN the result is the same object
    assert module_of_module is module_source
