# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import fastforward as ff
import torch


class QuantizedNet(ff.nn.QuantizedModule):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = ff.nn.QuantizedLinear(3, 3)
        self.w2 = ff.nn.QuantizedLinear(3, 3)

    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.quantizer_pow = ff.nn.QuantizerStub()
        self.quantizer_add = ff.nn.QuantizerStub()
        self.quantizer_sigmoid = ff.nn.QuantizerStub()
        self.quantizer_relu = ff.nn.QuantizerStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.w1(x)
        h = ff.nn.functional.pow(h, 2, output_quantizer=self.quantizer_pow)
        h = ff.nn.functional.add(h, self.w2(x), output_quantizer=self.quantizer_add)
        h = ff.nn.functional.sigmoid(h, output_quantizer=self.quantizer_sigmoid)
        h = ff.nn.functional.relu(h, output_quantizer=self.quantizer_relu)
        return h


def test_annotate_operator_metadata() -> None:
    # GIVEN: an quantized module.
    model = QuantizedNet()
    sample_input = torch.randn(1, 3)

    # WHEN: the model is annotated.
    ff.annotate_operator_metadata(model, sample_input)

    # THEN: searching with mpath should return the correct operators.

    # We have two linear layers, each with three quantizers before (inputs, weight, biases).
    before_linear = [s.module for s in ff.mpath.search("**/[quantizer:before/linear]", model)]
    assert len(before_linear) == 6
    assert model.w1.input_quantizer in before_linear
    assert model.w1.weight_quantizer in before_linear
    assert model.w1.bias_quantizer in before_linear
    assert model.w2.input_quantizer in before_linear
    assert model.w2.weight_quantizer in before_linear
    assert model.w2.bias_quantizer in before_linear

    # Each linear also has a single output quantizer.
    after_linear = [s.module for s in ff.mpath.search("**/[quantizer:after/linear]", model)]
    assert model.w1.output_quantizer in after_linear
    assert model.w2.output_quantizer in after_linear

    # Following this is a power of 2 operation which happens after the linear layer.
    before_pow = [s.module for s in ff.mpath.search("**/[quantizer:before/pow]", model)]
    assert len(before_pow) == 1
    assert model.w1.output_quantizer in before_pow

    # And the power of 2 operation is followed by its output quantizer.
    after_pow = [s.module for s in ff.mpath.search("**/[quantizer:after/pow]", model)]
    assert len(after_pow) == 1
    assert model.quantizer_pow in after_pow

    # Following this is an add operation with the output of the second linear layer and the power operation.
    before_add = [s.module for s in ff.mpath.search("**/[quantizer:before/add]", model)]
    assert len(before_add) == 2
    assert model.w2.output_quantizer in before_add
    assert model.quantizer_pow in before_add

    # And the add operation is followed by its output quantizer.
    after_add = [s.module for s in ff.mpath.search("**/[quantizer:after/add]", model)]
    assert len(after_add) == 1
    assert model.quantizer_add in after_add

    # Following this is a sigmoid operation with the output of the add operation.
    before_sigmoid = [s.module for s in ff.mpath.search("**/[quantizer:before/sigmoid]", model)]
    assert len(before_sigmoid) == 1
    assert model.quantizer_add in before_sigmoid

    # And the sigmoid operation is followed by its output quantizer.
    after_sigmoid = [s.module for s in ff.mpath.search("**/[quantizer:after/sigmoid]", model)]
    assert len(after_sigmoid) == 1
    assert model.quantizer_sigmoid in after_sigmoid

    # Following this is a relu operation with the output of the sigmoid operation.
    before_relu = [s.module for s in ff.mpath.search("**/[quantizer:before/relu]", model)]
    assert len(before_relu) == 1
    assert model.quantizer_sigmoid in before_relu

    # And the relu operation is followed by its output quantizer.
    after_relu = [s.module for s in ff.mpath.search("**/[quantizer:after/relu]", model)]
    assert len(after_relu) == 1
    assert model.quantizer_relu in after_relu
