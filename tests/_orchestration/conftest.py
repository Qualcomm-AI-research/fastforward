# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Shared fixtures and delegates for the orchestration test suite.

Model classes live in `_models.py`; this module wraps them in function-scoped
fixtures (fresh instances per test, since several tests mutate weights) and
provides the canonical `sgd_step` / `noop` optimization delegates.
"""

from typing import Any

import pytest
import torch

from fastforward._orchestration.instruction_engine import ActivationBundle

from ._models import (
    AddConstant,
    CatModel,
    ConstReturn,
    ConstReturnKwargs,
    DecoderLayer,
    DeviceCastModel,
    DualOutLayer,
    DualOutModel,
    KwargForward,
    MixedOps,
    Model,
    MultiAxisIndex,
    MultiOutputModel,
    NestedMLP,
    NestedWithTensorOps,
    ProbeModule,
    ReturnTuple,
    RNGTensor,
    SmallAttn,
    SmallMLP,
    TinyMLP,
    TinyModel,
    ToyLlama,
    TupleOut,
    TwoLayerModel,
    TwoLinear,
    WithBuffer,
)


def sgd_step(module: torch.nn.Module, bundle: ActivationBundle, lr: float = 0.1) -> None:
    """Run one SGD step per calibration batch through `module`.

    The canonical optimization delegate used throughout the orchestration tests:
    iterate the bundle, compute `(module(*args, **kwargs) ** 2).mean()`, and step.
    """
    optim = torch.optim.SGD(params=module.parameters(), lr=lr)
    for args, kwargs in bundle:
        optim.zero_grad()
        loss = (module(*args, **kwargs) ** 2).mean()
        loss.backward()
        optim.step()


def noop(*_args: Any, **_kwargs: Any) -> None:
    """Delegate that does nothing; used for partitioning-only specs."""
    return None


@pytest.fixture(name="model")
def model_fixture() -> Model:
    """Fresh two-residual-block `Model` (tests mutate its weights)."""
    return Model()


@pytest.fixture(name="multi_output_model")
def multi_output_model_fixture() -> MultiOutputModel:
    """Fresh `MultiOutputModel` (tests mutate its weights)."""
    return MultiOutputModel()


@pytest.fixture(name="two_layer_model")
def two_layer_model_fixture() -> TwoLayerModel:
    """Fresh transformer-shaped `TwoLayerModel`."""
    return TwoLayerModel()


@pytest.fixture(name="dual_out_model")
def dual_out_model_fixture() -> DualOutModel:
    """Fresh `DualOutModel` with a multi-output inner fold."""
    return DualOutModel()


@pytest.fixture(name="dual_out_layer")
def dual_out_layer_fixture() -> DualOutLayer:
    """Fresh `DualOutLayer` with two independent output leaves."""
    return DualOutLayer()


@pytest.fixture(name="cat_model")
def cat_model_fixture() -> CatModel:
    """Fresh `CatModel` that passes a list of tensors to torch.cat."""
    return CatModel()


@pytest.fixture(name="two_linear")
def two_linear_fixture() -> TwoLinear:
    """Fresh `TwoLinear` model with two independently-targetable Linear leaves."""
    return TwoLinear()


@pytest.fixture(name="tiny_model")
def tiny_model_fixture() -> TinyModel:
    """Fresh `TinyModel` (two Linear leaves + Conv2d) for registry resolution."""
    return TinyModel()


@pytest.fixture(name="small_attn")
def small_attn_fixture() -> SmallAttn:
    """Fresh `SmallAttn` self-attention block."""
    return SmallAttn()


@pytest.fixture(name="small_mlp")
def small_mlp_fixture() -> SmallMLP:
    """Fresh `SmallMLP` block."""
    return SmallMLP()


@pytest.fixture(name="decoder_layer")
def decoder_layer_fixture() -> DecoderLayer:
    """Fresh `DecoderLayer` (attn + mlp fold)."""
    return DecoderLayer()


@pytest.fixture(name="const_return")
def const_return_fixture() -> ConstReturn:
    """Fresh `ConstReturn` module."""
    return ConstReturn()


@pytest.fixture(name="const_return_kwargs")
def const_return_kwargs_fixture() -> ConstReturnKwargs:
    """Fresh `ConstReturnKwargs` module."""
    return ConstReturnKwargs()


@pytest.fixture(name="add_constant")
def add_constant_fixture() -> AddConstant:
    """Fresh `AddConstant` module."""
    return AddConstant()


@pytest.fixture(name="return_tuple")
def return_tuple_fixture() -> ReturnTuple:
    """Fresh `ReturnTuple` module."""
    return ReturnTuple()


@pytest.fixture(name="rng_tensor")
def rng_tensor_fixture() -> RNGTensor:
    """Fresh `RNGTensor` no-input generator module."""
    return RNGTensor()


@pytest.fixture(name="probe_module")
def probe_module_fixture() -> ProbeModule:
    """Fresh `ProbeModule` that records inference-mode state."""
    return ProbeModule()


@pytest.fixture(name="tiny_mlp")
def tiny_mlp_fixture() -> TinyMLP:
    """Fresh `TinyMLP` (canonical tracer happy-path model)."""
    return TinyMLP()


@pytest.fixture(name="tuple_out")
def tuple_out_fixture() -> TupleOut:
    """Fresh `TupleOut` module (tuple-returning forward)."""
    return TupleOut()


@pytest.fixture(name="multi_axis_index")
def multi_axis_index_fixture() -> MultiAxisIndex:
    """Fresh `MultiAxisIndex` module."""
    return MultiAxisIndex()


@pytest.fixture(name="with_buffer")
def with_buffer_fixture() -> WithBuffer:
    """Fresh `WithBuffer` module (registered buffer read in forward)."""
    return WithBuffer()


@pytest.fixture(name="kwarg_forward")
def kwarg_forward_fixture() -> KwargForward:
    """Fresh `KwargForward` module (keyword-only argument)."""
    return KwargForward()


@pytest.fixture(name="nested_mlp")
def nested_mlp_fixture() -> NestedMLP:
    """Fresh `NestedMLP` module (two-level nesting)."""
    return NestedMLP()


@pytest.fixture(name="mixed_ops")
def mixed_ops_fixture() -> MixedOps:
    """Fresh `MixedOps` module (module calls + tensor ops + buffer)."""
    return MixedOps()


@pytest.fixture(name="nested_with_tensor_ops")
def nested_with_tensor_ops_fixture() -> NestedWithTensorOps:
    """Fresh `NestedWithTensorOps` module (nests MixedOps in a subgraph)."""
    return NestedWithTensorOps()


@pytest.fixture(name="device_cast_model")
def device_cast_model_fixture() -> DeviceCastModel:
    """Fresh `DeviceCastModel` module (device-following buffer cast)."""
    return DeviceCastModel()


@pytest.fixture(name="toy_llama")
def toy_llama_fixture() -> ToyLlama:
    """Fresh `ToyLlama` (Llama-shaped 2-layer stack)."""
    return ToyLlama()
