# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


from typing import Any, Callable

import libcst
import pytest
import torch

from fastforward._quantops import OperatorTable
from fastforward.autoquant.cfg import blocks
from fastforward.autoquant.cfg.variable_tracking import QuantizationStatus, infer_block_dataflow
from fastforward.autoquant.cst.passes import QuantizedCounterpartReplacer
from fastforward.autoquant.pybuilder import QuantizedModuleBuilder

from tests.autoquant.cfg.cfg_test import CFGTest


class TestQuantizationStatus(CFGTest):
    def test_quantization_status(
        self, cfg: blocks.FunctionBlock, expected_status: dict[str, QuantizationStatus]
    ) -> None:
        # GIVEN a CFG

        # WHEN the variables are tracked
        block_trackers = infer_block_dataflow(cfg)

        # THEN the quantization status must match the expected quantization status
        var_status: dict[str, QuantizationStatus] = {}
        for tracker in block_trackers.values():
            for var in tracker.vars_gen:
                var_status[f"{var.name}:{var.version}"] = var.quantized

        for name, status in expected_status.items():
            assert var_status[name] is status

    @CFGTest.case
    def simple_case(
        self, input: torch.Tensor, other: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out1 = torch.relu(input)
        out2 = other
        return out1, out2

    @CFGTest.case
    def if_else_case(
        self, input: torch.Tensor, other: torch.Tensor, value: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if value < 0:
            out1 = torch.relu(input)
            out2 = other
        elif value > 0:
            out1 = other
            out2 = torch.sigmoid(input)
        else:
            out1 = torch.pow(input, 2)
            out2 = other
        return out1, out2

    @pytest.fixture(scope="class")
    @classmethod
    def expected_status(cls, case: Callable[..., Any]) -> dict[str, QuantizationStatus]:
        """Returns the expected status for each case."""
        if case is cls.simple_case:
            return {
                "input:0": QuantizationStatus.NotQuantized,
                "other:0": QuantizationStatus.NotQuantized,
                "out1:0": QuantizationStatus.Quantized,
                "out2:0": QuantizationStatus.NotQuantized,
            }
        elif case is cls.if_else_case:
            return {
                "input:0": QuantizationStatus.NotQuantized,
                "other:0": QuantizationStatus.NotQuantized,
                "value:0": QuantizationStatus.NotQuantized,
                "out1:0": QuantizationStatus.Quantized,
                "out2:0": QuantizationStatus.NotQuantized,
                "out1:1": QuantizationStatus.NotQuantized,
                "out2:1": QuantizationStatus.Quantized,
                "out1:2": QuantizationStatus.Quantized,
                "out2:2": QuantizationStatus.NotQuantized,
            }
        else:
            raise ValueError("Unknown case")

    @pytest.fixture(scope="class")
    def cst(self, raw_cst: libcst.FunctionDef, optable: OperatorTable) -> libcst.FunctionDef:  # type: ignore[override]
        """Replace function calls with their quantized counterpart for each CST."""
        quantized_counterpart_replacer = QuantizedCounterpartReplacer(
            optable=optable, quantizer_list=QuantizedModuleBuilder("Dummy", ())
        )
        cst = raw_cst.visit(quantized_counterpart_replacer)
        assert isinstance(cst, libcst.FunctionDef)
        return cst
