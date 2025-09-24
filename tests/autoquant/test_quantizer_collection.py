# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Any, Sequence

import libcst
import pytest
import torch

from fastforward._autoquant.cst import nodes
from fastforward._autoquant.function_context import FunctionContext
from fastforward._autoquant.pybuilder.quantizer_collection import QuantizerReferenceCollection


# Example PyTorch modules for testing
class ExampleTorchModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def instance_method(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def static_method(x: torch.Tensor) -> torch.Tensor:
        return x


class AnotherTorchModule(torch.nn.Module):
    def instance_method(self, x: torch.Tensor) -> torch.Tensor:
        return x


def example_function(x: Any) -> Any:
    return x


def test_create_quantizer_expression_no_context() -> None:
    """Test quantizer expression creation without context returns simple reference."""
    # GIVEN a collection with no context
    collection = QuantizerReferenceCollection()

    # WHEN creating a quantizer expression
    expr = collection.create_quantizer_expression("ant")

    # THEN expression is a simple reference
    assert isinstance(expr, nodes.QuantizerReference)
    assert expr.value == "ant"


def test_create_quantizer_expression_instance_method() -> None:
    """Test quantizer expression creation for instance method returns attribute access."""
    # GIVEN a collection with instance method context
    collection = QuantizerReferenceCollection()
    instance_context = FunctionContext.from_method(ExampleTorchModule, "instance_method")

    # WHEN creating a quantizer expression in instance context
    with collection.push_context(instance_context):
        expr = collection.create_quantizer_expression("ant")

    # THEN expression is an attribute access on self
    assert isinstance(expr, libcst.Attribute)
    assert isinstance(expr.value, libcst.Name)
    assert expr.value.value == "self"
    assert isinstance(expr.attr, nodes.QuantizerReference)


def test_create_quantizer_expression_static_method() -> None:
    """Test quantizer expression creation for static method returns simple reference."""
    # GIVEN a collection with static method context
    collection = QuantizerReferenceCollection()
    static_context = FunctionContext.from_method(ExampleTorchModule, "static_method")

    # WHEN creating a quantizer expression in static context
    with collection.push_context(static_context):
        expr = collection.create_quantizer_expression("ant")

    # THEN expression is a simple reference
    assert isinstance(expr, nodes.QuantizerReference)
    assert expr.value == "ant"


@pytest.mark.parametrize(
    "names,expected_disambiguated",
    [
        (["ant"], ["quantizer_ant"]),
        (["ant", "ant"], ["quantizer_ant_1", "quantizer_ant_2"]),
        (["ant", "ant", "bat"], ["quantizer_ant_1", "quantizer_ant_2", "quantizer_bat"]),
    ],
)
def test_name_disambiguation_same_context(
    names: Sequence[str], expected_disambiguated: Sequence[str]
) -> None:
    """Test name disambiguation within the same context."""
    # GIVEN a collection with multiple quantizers of same/different names
    collection = QuantizerReferenceCollection()
    refs = [collection.create_reference(name) for name in names]

    # WHEN disambiguating all names
    collection.disambiguate_all_names()

    # THEN names are disambiguated correctly
    actual_names = [collection.disambiguate_reference(ref) for ref in refs]
    assert actual_names == expected_disambiguated


def test_name_disambiguation_different_instance_contexts() -> None:
    """Test that quantizers in different instance method contexts don't conflict."""
    # GIVEN a collection with quantizers in different module instance contexts
    collection = QuantizerReferenceCollection()
    context1 = FunctionContext.from_method(ExampleTorchModule, "instance_method")
    context2 = FunctionContext.from_method(AnotherTorchModule, "instance_method")

    with collection.push_context(context1):
        ref1 = collection.create_reference("ant")

    with collection.push_context(context2):
        ref2 = collection.create_reference("ant")

    # WHEN disambiguating names
    collection.disambiguate_all_names()

    # THEN same name in different module contexts should not conflict
    assert collection.disambiguate_reference(ref1) == "quantizer_ant"
    assert collection.disambiguate_reference(ref2) == "quantizer_ant"


def test_name_disambiguation_different_function_contexts() -> None:
    """Test that quantizers in different function contexts don't conflict."""
    # GIVEN a collection with quantizers in different function contexts
    collection = QuantizerReferenceCollection()
    static_context = FunctionContext.from_method(ExampleTorchModule, "static_method")
    func_context = FunctionContext.from_function_reference(example_function, None)

    with collection.push_context(static_context):
        ref1 = collection.create_reference("ant")

    with collection.push_context(func_context):
        ref2 = collection.create_reference("ant")

    # WHEN disambiguating names
    collection.disambiguate_all_names()

    # THEN same name in different function contexts should not conflict
    assert collection.disambiguate_reference(ref1) == "quantizer_ant"
    assert collection.disambiguate_reference(ref2) == "quantizer_ant"


def test_local_quantizers_for_func() -> None:
    """Test retrieval of local quantizers for functions."""
    # GIVEN a collection with quantizers in different contexts
    collection = QuantizerReferenceCollection()

    # Create quantizers in static method context
    static_context = FunctionContext.from_method(ExampleTorchModule, "static_method")
    with collection.push_context(static_context):
        ref1 = collection.create_reference("ant")
        ref2 = collection.create_reference("bat")

    # Create quantizer in instance method context (should not be local)
    instance_context = FunctionContext.from_method(ExampleTorchModule, "instance_method")
    with collection.push_context(instance_context):
        collection.create_reference("cat")

    # WHEN getting local quantizers for the static method
    local_quantizers = list(collection.local_quantizers_for_func(ExampleTorchModule.static_method))

    # THEN only static method quantizers are returned
    refids = {q.refid for q in local_quantizers}
    assert refids == {ref1.refid, ref2.refid}


def test_instance_quantizers_for_module() -> None:
    """Test retrieval of instance quantizers for modules."""
    # GIVEN a collection with quantizers in different contexts
    collection = QuantizerReferenceCollection()

    # Create quantizers in instance method context
    instance_context = FunctionContext.from_method(ExampleTorchModule, "instance_method")
    with collection.push_context(instance_context):
        ref1 = collection.create_reference("ant")
        ref2 = collection.create_reference("bat")

    # Create quantizer in static method context (should not be instance)
    static_context = FunctionContext.from_method(ExampleTorchModule, "static_method")
    with collection.push_context(static_context):
        collection.create_reference("cat")

    # Create quantizer in different module (should not be included)
    other_context = FunctionContext.from_method(AnotherTorchModule, "instance_method")
    with collection.push_context(other_context):
        collection.create_reference("dog")

    # WHEN getting instance quantizers for the module
    instance_quantizers = list(collection.instance_quantizers_for_module(ExampleTorchModule))

    # THEN only instance method quantizers from the correct module are returned
    refids = {q.refid for q in instance_quantizers}
    assert refids == {ref1.refid, ref2.refid}


def test_context_manager_behavior() -> None:
    """Test that context manager properly manages context state."""
    # GIVEN a collection and contexts
    collection = QuantizerReferenceCollection()
    context = FunctionContext.from_method(ExampleTorchModule, "instance_method")

    # WHEN using context manager
    assert collection._quantization_context is None

    with collection.push_context(context):
        assert collection._quantization_context == context

        # Test nested context
        other_context = FunctionContext.from_method(ExampleTorchModule, "static_method")
        with collection.push_context(other_context):
            assert collection._quantization_context == other_context

        assert collection._quantization_context == context

    # THEN context is properly restored
    assert collection._quantization_context is None


def test_custom_prefix() -> None:
    """Test collection with custom quantizer name prefix."""
    # GIVEN a collection with custom prefix
    collection = QuantizerReferenceCollection("custom_")
    ref = collection.create_reference("ant")

    # WHEN disambiguating names
    collection.disambiguate_all_names()

    # THEN custom prefix is used
    assert collection.disambiguate_reference(ref) == "custom_ant"


def test_disambiguate_reference_by_id() -> None:
    """Test reference disambiguation by reference ID."""
    # GIVEN a collection with a quantizer
    collection = QuantizerReferenceCollection()
    ref = collection.create_reference("ant")

    # WHEN disambiguating by reference ID
    name = collection.disambiguate_reference(ref.refid)

    # THEN correct name is returned
    assert name == "quantizer_ant"


def test_disambiguate_reference_invalid_id() -> None:
    """Test reference disambiguation with invalid ID raises KeyError."""
    # GIVEN a collection
    collection = QuantizerReferenceCollection()

    # WHEN disambiguating with invalid ID
    # THEN KeyError is raised
    with pytest.raises(KeyError):
        collection.disambiguate_reference(999)


def test_prefix_no_duplicate_when_already_present() -> None:
    """Test that prefix is not duplicated if name already includes it."""
    # GIVEN a collection
    collection = QuantizerReferenceCollection(quantizer_name_prefix="a_quantizer_")
    # WHEN creating a reference with a name that already includes the prefix
    # and disambiguating all names
    ref_ant = collection.create_reference("a_quantizer_ant")
    ref_bat = collection.create_reference("bat_quantizer")
    collection.disambiguate_all_names()
    # THEN the disambiguated name should remain unchanged
    assert collection.disambiguate_reference(ref_ant) == "a_quantizer_ant"
    assert collection.disambiguate_reference(ref_bat) == "a_quantizer_bat_quantizer"
