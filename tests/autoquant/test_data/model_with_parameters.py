# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import torch


class ModelWithParameters(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(10, 10))
        self.bias = torch.nn.Parameter(torch.randn(10))
        self.alt_bias = torch.nn.Parameter(torch.randn(10))
        
    def forward(self, input: torch.Tensor, aux: int) -> torch.Tensor:
        """Example that uses parameters.

        This example uses parameters that are defined on the instance. The
        parameters are expected to be quantized only if they are used. In this
        example, self.weight is used in every branch and can be quantized once
        in the top-level branch. For bias and alt_bias, however, this is not
        the case as they are only used in the true branch of the first and
        second if statement.
        """
        if aux < 10:
            # normal weight and bias
            return torch.nn.functional.linear(input, self.weight, self.bias)
        if aux < 20:
            # normal weight alternative bias
            return torch.nn.functional.linear(input, self.weight, self.alt_bias)
        else:
            # no bias
            return torch.nn.functional.linear(input, self.weight)


def get_model() -> torch.nn.Module:
    return ModelWithParameters()