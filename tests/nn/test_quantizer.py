# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import Any, Callable

import pytest
import torch

from fastforward.nn.quantizer import Quantizer


def test_register_override() -> None:
    quantizer = Quantizer()

    def override1(
        quantizer_ctx: Quantizer,
        fn: Callable[..., torch.Tensor],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> torch.Tensor:
        assert quantizer_ctx is quantizer, "Quantizer should be passed into override as context"
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
