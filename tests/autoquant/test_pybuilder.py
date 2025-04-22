# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import libcst
import libcst.helpers
import pytest

from fastforward._autoquant.cst.nodes import QuantizerReference
from fastforward._autoquant.pybuilder.builder import _disambiguate_quantizers


@pytest.mark.parametrize(
    ["quant_references", "reference_template", "expected_output_str"],
    [
        (
            {"node": QuantizerReference("my_node")},
            "placeholder = {node}",
            "placeholder = quantizer_my_node",
        ),
        (
            {"node": QuantizerReference("my_node")},
            "placeholder = ({node}, {node})",
            "placeholder = (quantizer_my_node, quantizer_my_node)",
        ),
        (
            {"node": QuantizerReference("my_node"), "other_node": QuantizerReference("other_node")},
            "placeholder = ({node}, {other_node})",
            "placeholder = (quantizer_my_node, quantizer_other_node)",
        ),
        (
            {"node": QuantizerReference("my_node"), "other_node": QuantizerReference("my_node")},
            "placeholder = ({node}, {other_node})",
            "placeholder = (quantizer_my_node_1, quantizer_my_node_2)",
        ),
    ],
)
def test_disambiguate_quantizers(
    quant_references: dict[str, QuantizerReference],
    reference_template: str,
    expected_output_str: str,
) -> None:
    # GIVEN a CST with QuantizerReference nodes
    statement = libcst.helpers.parse_template_statement(reference_template, **quant_references)  # type: ignore[arg-type]

    # WHEN the quantizer reference nodes are disambiguated
    actual_output = _disambiguate_quantizers(
        statement, [ref.quantizer_info for ref in quant_references.values()]
    )

    # THEN the QuantizerReference nodes must be replaces by libcst.Name nodes
    # and the resulting CST must match the expected CST.
    expected_output = libcst.parse_statement(expected_output_str)
    assert expected_output.deep_equals(actual_output)
