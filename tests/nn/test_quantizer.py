# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import copy

from typing import Any, Callable

import pytest
import torch

from fastforward.nn.quantizer import Quantizer, QuantizerMetadata, Tag


def test_register_override() -> None:
    quantizer = Quantizer()

    def override1(
        quantizer_ctx: Quantizer,
        fn: Callable[..., torch.Tensor],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> torch.Tensor:
        assert quantizer_ctx is quantizer, "Quantizer should be passed into override as context"
        del fn
        del args
        del kwargs
        return torch.tensor(100)

    def override2(
        quantizer_ctx: Quantizer,
        fn: Callable[..., torch.Tensor],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> torch.Tensor:
        assert quantizer_ctx is quantizer, "Quantizer should be passed into override as context"
        return fn(*args, **kwargs) * 2

    with quantizer.register_override(override1):
        assert quantizer(torch.rand(10)) == torch.tensor(100)
        with quantizer.register_override(override2):
            assert quantizer(torch.rand(10)) == torch.tensor(200)
        assert quantizer(torch.rand(10)) == torch.tensor(100)

    # Override should be removed after closing with block. The default
    # implementation of Quantizer.quantizer raises NotImplementedError.
    with pytest.raises(NotImplementedError):
        quantizer(torch.rand(10))


def test_tag_copy() -> None:
    # GIVEN a tag
    tag = Tag("tag")

    # WHEN the tag is copied or deep copied
    # THEN the resulting tag is the same as the original
    assert copy.copy(tag) is tag
    assert copy.deepcopy(tag) is tag


def test_quantizermetadata_copy() -> None:
    # GIVEN a metadata object
    metadata = QuantizerMetadata()

    # WHEN the metadata is copied or deep copied
    # THEN the resulting metadata is a new object
    #      and copying does not fail
    assert copy.copy(metadata) is not metadata
    assert copy.deepcopy(metadata) is not metadata


def test_quantizer_getstate_with_custom_attributes() -> None:
    """Test __getstate__ preserves custom attributes."""
    # GIVEN a quantizer with custom attributes
    quantizer = Quantizer()
    setattr(quantizer, "custom_attr", "test_value")
    setattr(quantizer, "custom_number", 42)
    setattr(quantizer, "custom_tensor", torch.tensor([1, 2, 3]))

    # WHEN getting the state
    state = quantizer.__getstate__()

    # THEN custom attributes should be preserved
    assert state["custom_attr"] == "test_value"
    assert state["custom_number"] == 42
    assert "custom_tensor" not in state


def test_quantizer_getstate_filters_stdlib_attributes() -> None:
    """Test that __getstate__ filters out standard library and torch attributes."""
    # GIVEN a quantizer
    quantizer = Quantizer()

    # WHEN getting the state
    state = quantizer.__getstate__()

    # THEN standard library attributes should be filtered out
    # Check that common object attributes are not included
    stdlib_attrs = [
        "__class__",
        "__dict__",
        "__doc__",
        "__module__",
        "__weakref__",
        "__orig_bases__",
        "__parameters__",
    ]
    for attr in stdlib_attrs:
        assert attr not in state, f"Standard library attribute {attr} should be filtered out"

    # AND torch module attributes should be filtered out
    torch_attrs = ["training", "dump_patches"]
    for attr in torch_attrs:
        if hasattr(quantizer, attr):
            assert attr not in state, f"Torch attribute {attr} should be filtered out"


def test_quantizer_setstate_basic() -> None:
    """Test basic functionality of __setstate__ method."""
    # GIVEN a quantizer and a state dictionary
    quantizer = Quantizer()
    state: dict[str, Any] = {
        "_quantizer_overrides": {},
        "quant_metadata": None,
        "custom_attr": "restored_value",
    }

    # WHEN setting the state
    quantizer.__setstate__(state)

    # THEN the attributes should be restored
    assert quantizer.custom_attr == "restored_value"
    assert quantizer.quant_metadata is None


def test_quantizer_setstate_with_custom_attributes() -> None:
    """Test __setstate__ restores custom attributes correctly."""
    # GIVEN a quantizer and state with various attribute types
    quantizer = object.__new__(Quantizer)
    test_tensor = torch.tensor([4, 5, 6])
    state = {
        "_quantizer_overrides": {},
        "quant_metadata": None,
        "string_attr": "test_string",
        "int_attr": 123,
        "float_attr": 3.14,
        "tensor_attr": test_tensor,
        "list_attr": [1, 2, 3],
        "dict_attr": {"key": "value"},
    }

    # WHEN setting the state
    quantizer.__setstate__(state)

    # THEN all attributes should be restored correctly
    assert quantizer.string_attr == "test_string"
    assert quantizer.int_attr == 123
    assert quantizer.float_attr == 3.14
    assert torch.equal(quantizer.tensor_attr, test_tensor)
    assert quantizer.list_attr == [1, 2, 3]
    assert quantizer.dict_attr == {"key": "value"}


def test_quantizer_getstate_excludes_properties() -> None:
    """Test that __getstate__ excludes property attributes."""

    # Create a custom quantizer class with a property for testing
    class TestQuantizer(Quantizer):
        regular_attr: str

        @property
        def test_property(self) -> str:
            return "property_value"

    # GIVEN a quantizer with properties
    quantizer = TestQuantizer()
    quantizer.regular_attr = "regular_value"

    # WHEN getting the state
    state = quantizer.__getstate__()

    # THEN properties should be excluded, regular attributes should be included
    assert "test_property" not in state
    assert "regular_attr" in state
    assert state["regular_attr"] == "regular_value"


def test_quantizer_state_with_overrides() -> None:
    """Test state preservation with registered overrides."""
    # GIVEN a quantizer with registered overrides
    quantizer = Quantizer()

    def test_override(
        _quantizer_ctx: Quantizer,
        _fn: Callable[..., torch.Tensor],
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> torch.Tensor:
        return torch.tensor(42)

    # Register an override
    quantizer.register_override(test_override)

    # WHEN getting raise an exception
    with pytest.raises(RuntimeError):
        quantizer.__getstate__()
