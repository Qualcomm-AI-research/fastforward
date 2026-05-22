# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


from dataclasses import dataclass
from typing import TypeAlias

import syrupy
import torch

from fastforward._autoquant.autoquant import autoquant_with_defaults, codeformat_with_defaults

#  required for tests, do not remove:
from torch import Tensor as TensorAlias  # noqa # pylint: disable=unused-import

Tensor: TypeAlias = torch.Tensor  # Required for tests, do not remove


def _repeat(x: torch.Tensor) -> torch.Tensor:
    return x.repeat_interleave(2, dim=1)


# ------------------------------------------------------------------------------
class ReassignmentModel(torch.nn.Module):
    """Test model: autoquant ability to insert hoisted quantize statements in the correct order.

    The Attribute access `k.T` on an argument of a quantized op gets hoisted
    out of the original statement into a new `k_T = quantizer(k.T)` line.
    The hoisted statement must be placed AFTER the reassignment of `k` (here
    `k = _repeat(k)`), otherwise `k.T` is captured against the original
    (non-repeated) value and produces a tensor of the wrong shape.

    (issue #575)
    """

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(8, 8)
        self.k_proj = torch.nn.Linear(8, 4)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        k = _repeat(k)
        return torch.matmul(q, k.T)


def test_quantize_statement_respects_reassignment(
    snapshot: syrupy.assertion.SnapshotAssertion,
) -> None:
    """Test autoquant ability to insert in the correct order hoisted quantize statements."""
    # GIVEN a torch model having a reassignment of a Tensor variable in the
    #       forward pass and soon after an inline method-call/property-access
    #       over that variable within an expression.

    # WHEN autoquant is called on the given model
    code = autoquant_with_defaults(ReassignmentModel(), use_type_inference=False)

    # THEN the generated quantized model will have a new statement that call
    #      the method/property over the tensor and store the output into a
    #      temporary variable that is used in the following operation,
    #      implying an identical operation execution order of the original model
    #      with the addition of the quantization operations.
    formatted = codeformat_with_defaults(code).strip()
    assert snapshot == formatted


# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
class ReassignmentModelPartiallyClashingName(torch.nn.Module):
    """Test model: autoquant ability to insert hoisted quantize statements in the correct order.

    This model is similar to ReassignmentModel, the only difference is that we
    have a variable with a (T) that is identical to the property name `k.T`.

    Autoquant should insert a new statement `k_T = quantizer(k.T)`.
    Specifically, autoquant should not blidly look at the property name `T`
    confusing it for the local variable with the same name, i.e. should not
    insert a wrong statement like `k_T = quantizer(T)`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(8, 8)
        self.k_proj = torch.nn.Linear(8, 4)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        k = _repeat(k)
        T = torch.tensor([10.0])
        return torch.matmul(q, k.T * T)


def test_quantize_statement_respects_reassignment_when_partially_clashing_name(
    snapshot: syrupy.assertion.SnapshotAssertion,
) -> None:
    """Test autoquant ability to insert in the correct order hoisted quantize statements."""
    # GIVEN a torch model having a reassignment of a Tensor variable in the
    #       forward pass and soon after an inline method-call/property-access
    #       over that variable within an expression, with the method/property
    #       name clashing with another local variable name

    # WHEN autoquant is called on the given model
    code = autoquant_with_defaults(
        ReassignmentModelPartiallyClashingName(), use_type_inference=False
    )

    # THEN the generated quantized model will have a new statement that call
    #      the method/property over the tensor and store the output into a
    #      temporary variable that is used in the following operation, without
    #      confusing the clashing local variable with the method/property name,
    #      implying an identical operation execution order of the original model
    #      with the addition of the quantization operations.
    formatted = codeformat_with_defaults(code).strip()
    assert snapshot == formatted


# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
@dataclass
class TimeData:
    T: torch.Tensor


class ReassignmentModelWithObjectProperty(torch.nn.Module):
    """Test model: autoquant ability to insert hoisted quantize statements in the correct order.

    This model is similar to ReassignmentModel, the only difference is that we
    are accessing the tensor `t` that is stored within an object `ts`.

    Autoquant should be able to understand that the object that should be
    quantized is the transposed version of `ts.t`, i.e. `ts.t.T`, and should
    create a local temporary variable in which to store the quantized version
    of this tensor: `ts_t_T = quantizer(ts.t.T)`
    """

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(8, 8)
        self.k_proj = torch.nn.Linear(8, 4)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        ts = TimeData(T=k)
        ts.T = _repeat(ts.T)
        return torch.matmul(q, ts.T.T)


def test_quantize_statement_respects_reassignment_accessing_object_property(
    snapshot: syrupy.assertion.SnapshotAssertion,
) -> None:
    """Test autoquant ability to insert in the correct order hoisted quantize statements.

    See issue #575 and ReassignmentModelWithObjectProperty docstring.
    """
    # GIVEN a torch model having in the forward pass a reassignment of a Tensor
    #       property stored in a dataclass object, and soon after an inline
    #       method-call/property-access over that tensor within an expression

    # WHEN autoquant is called on the given model
    code = autoquant_with_defaults(ReassignmentModelWithObjectProperty(), use_type_inference=False)

    # THEN the generated quantized model will have a new statement that call
    #      the method/property over the tensor and store the output into a
    #      temporary variable that is used in the following operation,
    #      implying an identical operation execution order of the original model
    #      with the addition of the quantization operations.
    formatted = codeformat_with_defaults(code).strip()
    assert snapshot == formatted


# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
class ComplexReassignmentModel(torch.nn.Module):
    """Test model: autoquant ability to insert hoisted quantize statements in the correct order.

    The Attribute access `k.T` on an argument of a quantized op gets hoisted
    out of the original statement into a new `k_T = quantizer(k.T)` line.
    The hoisted statement must be placed AFTER the reassignment of `k` (here
    `k = _repeat(k)`), otherwise `k.T` is captured against the original
    (non-repeated) value and produces a tensor of the wrong shape.

    (issue #575)
    """

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(8, 8)
        self.k_proj = torch.nn.Linear(8, 4)
        self.v_proj = torch.nn.Linear(8, 4)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        v = v + k
        k = k + v
        k = _repeat(k)
        score = torch.matmul(q, k.T)
        prob = torch.matmul(v * k, score.T)
        return prob


def test_quantize_statement_respects_reassignment_complex_model(
    snapshot: syrupy.assertion.SnapshotAssertion,
) -> None:
    """Test autoquant ability to insert in the correct order hoisted quantize statements."""
    # GIVEN a torch model having a reassignment of a Tensor variable in the
    #       forward pass and soon after an inline method-call/property-access
    #       over that variable within an expression.

    # WHEN autoquant is called on the given model
    code = autoquant_with_defaults(ComplexReassignmentModel(), use_type_inference=False)

    # THEN the generated quantized model will have a new statement that call
    #      the method/property over the tensor and store the output into a
    #      temporary variable that is used in the following operation,
    #      implying an identical operation execution order of the original model
    #      with the addition of the quantization operations.
    formatted = codeformat_with_defaults(code).strip()
    assert snapshot == formatted


# ------------------------------------------------------------------------------
