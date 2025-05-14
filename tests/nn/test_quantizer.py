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
