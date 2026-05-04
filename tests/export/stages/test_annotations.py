# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import pytest
import torch

from fastforward.export.stages.annotations import _ff_quantizer_spec


def _build_quantize_node(num_bits: int | float) -> tuple[torch.fx.Node, dict[str, torch.Tensor]]:
    graph = torch.fx.Graph()

    data = graph.placeholder("x")
    scale = graph.get_attr("scale")
    offset = graph.get_attr("offset")
    quantize = graph.call_function(
        torch.ops.fastforward.quantize_by_tile.default,
        args=(data, scale, (1,), num_bits, torch.int8, offset),
    )
    graph.output(quantize)

    quant_params = {
        "scale": torch.tensor([0.5]),
        "offset": torch.tensor([10.0]),
    }
    return quantize, quant_params


def test_ff_quantizer_spec_accepts_integer_num_bits() -> None:
    # GIVEN: A quantize node with integer num_bits.
    node, quant_params = _build_quantize_node(num_bits=8)

    # WHEN: Building an FF quantizer spec from the node.
    spec = _ff_quantizer_spec(node, quant_params)

    # THEN: num_bits should be preserved as an integer in the resulting spec.
    assert spec.num_bits == 8


def test_ff_quantizer_spec_rejects_non_integer_float_num_bits() -> None:
    # GIVEN: A quantize node with a non-integer float num_bits.
    node, quant_params = _build_quantize_node(num_bits=7.5)

    # WHEN: Building an FF quantizer spec from the node.
    # THEN: A ValueError should be raised.
    with pytest.raises(ValueError, match="Cannot export non-integer bitwidths"):
        _ff_quantizer_spec(node, quant_params)
