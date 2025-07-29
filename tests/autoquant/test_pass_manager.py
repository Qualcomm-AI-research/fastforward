# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#
import libcst
import libcst.metadata
import pytest

from fastforward._autoquant.pass_manager import (
    MetadataTransformer,
    PassManager,
    PassManagerError,
    _visit_cst_same,
)


class SimpleTransformer(libcst.CSTTransformer):
    def leave_Name(self, original_node: libcst.Name, updated_node: libcst.Name) -> libcst.Name:
        if original_node.value == "ant":
            return updated_node.with_changes(value="bat")
        return updated_node


class ErrorTransformer(libcst.CSTTransformer):
    def leave_Name(self, original_node: libcst.Name, updated_node: libcst.Name) -> libcst.Name:
        del original_node, updated_node
        raise ValueError("Test error")


class TypeChangingTransformer(libcst.CSTTransformer):
    def leave_Name(self, original_node: libcst.Name, updated_node: libcst.Name) -> libcst.Name:
        del original_node, updated_node
        return libcst.Integer("42")  # type: ignore[return-value]


class MetadataRequiringTransformer(libcst.CSTTransformer):
    METADATA_DEPENDENCIES = (libcst.metadata.ScopeProvider,)

    def leave_Name(self, original_node: libcst.Name, updated_node: libcst.Name) -> libcst.Name:
        # Just access metadata to ensure it's available
        self.get_metadata(libcst.metadata.ScopeProvider, original_node)
        if original_node.value == "ant":
            return updated_node.with_changes(value="cat")
        return updated_node


def test_pass_manager_simple_transformation() -> None:
    """Test that PassManager applies a simple transformation correctly."""
    # GIVEN a CST node and a transformer that changes 'ant' to 'bat'
    node = libcst.parse_expression("ant + bat")
    manager = PassManager([SimpleTransformer()])

    # WHEN the pass manager is applied to the node
    result = manager(node)

    # THEN the transformation should be applied correctly
    assert libcst.Module([]).code_for_node(result) == "bat + bat"


def test_pass_manager_no_change() -> None:
    """Test that PassManager returns the original node when no changes are made."""
    # GIVEN a CST node that won't be modified by the transformer
    node = libcst.parse_expression("cat + dog")
    manager = PassManager([libcst.CSTVisitor()])

    # WHEN the pass manager is applied to the node
    result = manager(node)

    # THEN the original node should be returned unchanged
    assert result is node  # Should return the same object if no changes


def test_pass_manager_multiple_passes() -> None:
    """Test that PassManager applies multiple transformations in sequence."""

    # GIVEN a CST node and multiple transformers
    class SecondTransformer(libcst.CSTTransformer):
        def leave_Name(self, original_node: libcst.Name, updated_node: libcst.Name) -> libcst.Name:
            if original_node.value == "bat":
                return updated_node.with_changes(value="dog")
            return updated_node

    node = libcst.parse_expression("ant + bat")
    manager = PassManager([SimpleTransformer(), SecondTransformer()])

    # WHEN the pass manager is applied with multiple passes
    result = manager(node)

    # THEN all transformations should be applied in sequence
    assert libcst.Module([]).code_for_node(result) == "dog + dog"


def test_pass_manager_error_handling() -> None:
    """Test that PassManager properly handles errors in transformers."""
    # GIVEN a CST node and a transformer that raises an error
    node = libcst.parse_expression("ant + bat")
    manager = PassManager([ErrorTransformer()])

    # WHEN/THEN the pass manager is applied, it should raise a PassManagerError
    with pytest.raises(PassManagerError):
        manager(node)


def test_metadata_transformer() -> None:
    """Test that PassManager correctly handles MetadataTransformer."""
    # GIVEN a CST node and a MetadataTransformer that requires scope information
    node = libcst.parse_expression("ant + bat")
    metadata_transformer = MetadataTransformer(
        transformer=MetadataRequiringTransformer(), wrap_in_module=True
    )
    manager = PassManager([metadata_transformer])

    # WHEN the pass manager is applied with the metadata transformer
    result = manager(node)

    # THEN the transformation should be applied with access to required metadata
    assert libcst.Module([]).code_for_node(result) == "cat + bat"


def test_type_changing_transformer_error() -> None:
    """Test that PassManager raises an error when a transformer changes node type."""
    # GIVEN a CST node and a transformer that changes node types
    node = libcst.parse_expression("ant")
    manager = PassManager([TypeChangingTransformer()])

    # WHEN/THEN the pass manager is applied, it should raise a ValueError
    with pytest.raises(PassManagerError):
        manager(node)


def test_visit_cst_same() -> None:
    """Test that _visit_cst_same correctly applies a visitor to a CST node."""
    # GIVEN a CST node and a transformer
    node = libcst.parse_expression("ant")

    # WHEN _visit_cst_same is called
    result = _visit_cst_same(node, SimpleTransformer())

    # THEN the transformation should be applied correctly
    assert libcst.Module([]).code_for_node(result) == "bat"


def test_visit_cst_same_type_error() -> None:
    """Test that _visit_cst_same raises an error when node type changes."""
    # GIVEN a CST node and a transformer that changes node types
    node = libcst.parse_expression("ant")

    # WHEN/THEN _visit_cst_same is called, it should raise a ValueError
    with pytest.raises(ValueError):
        _visit_cst_same(node, TypeChangingTransformer())


def test_metadata_transformer_unwrapping() -> None:
    """Test that MetadataTransformer correctly unwraps module after transformation."""
    # GIVEN a CST node and a MetadataTransformer with wrap_in_module=True
    node = libcst.parse_expression("ant")
    metadata_transformer = MetadataTransformer(
        transformer=MetadataRequiringTransformer(), wrap_in_module=True
    )
    manager = PassManager([metadata_transformer])

    # WHEN the pass manager is applied
    result = manager(node)

    # THEN the result should be properly unwrapped and transformed
    assert isinstance(result, type(node))
    assert libcst.Module([]).code_for_node(result) == "cat"
