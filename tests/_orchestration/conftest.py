# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Shared models, fixtures, and delegates for the orchestration test suite.

The graph-module and instruction-engine tests historically each defined their
own structurally identical copies of `Add`/`Residual`/`Model` and re-implemented
the same SGD optimization delegate inline in nearly every `local_optimize` test.
This module consolidates those into reusable model classes, function-scoped
fixtures (fresh instances per test, since several tests mutate weights), and a
single canonical `sgd_step` delegate.

The transformer-shaped models in `test_trace.py` are intentionally NOT shared:
they mirror HuggingFace Llama naming so mpath patterns match, and are built via
`trace()` rather than `to_graph_module()`.
"""

from typing import Any

import pytest
import torch

from fastforward._orchestration.graph_module import GraphModule
from fastforward._orchestration.instruction_engine import ActivationBundle


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


class Add(torch.nn.Module):
    """Placeholder due to lack of torch.nn.Add()."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Add two tensors together and return the result."""
        return x + y

    def to_graph_module(self) -> GraphModule:
        """Transform 'Add' to a GraphModule."""
        graph = GraphModule()
        x = graph.add_input("x")
        y = graph.add_input("y")
        output = graph.add_node("add", self, [x, y])
        graph.add_output(output)
        return graph


class Residual(torch.nn.Module):
    """Simple residual forward pass."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(5, 5)

    def forward(self, x: torch.Tensor) -> Any:
        """Forward pass of Residual."""
        h = self.linear(x)
        h = torch.nn.ReLU()(h)
        return x + h

    def to_graph_module(self) -> GraphModule:
        """Transform 'Residual' to a GraphModule."""
        graph = GraphModule()
        input = graph.add_input("input")
        linear = graph.add_node("linear", self.linear, [input])
        relu = graph.add_node("relu", torch.nn.ReLU(), [linear])
        (add,) = graph.add_subgraph("add", Add().to_graph_module(), [input, relu])
        graph.add_output(add)
        return graph


class Model(torch.nn.Module):
    """Simple Module class with two residual blocks for demonstrative purposes."""

    def __init__(self) -> None:
        super().__init__()
        self.residual_1 = Residual()
        self.residual_2 = Residual()

    def forward(self, x: torch.Tensor) -> Any:
        """Forward pass of Model."""
        h = self.residual_1(x)
        h = self.residual_2(h)
        return torch.nn.Sigmoid()(h)

    def to_graph_module(self) -> GraphModule:
        """Transform 'Model' to a GraphModule."""
        graph = GraphModule()
        input = graph.add_input("input")
        (residual_1,) = graph.add_subgraph("residual_1", self.residual_1.to_graph_module(), [input])
        (residual_2,) = graph.add_subgraph(
            "residual_2", self.residual_2.to_graph_module(), [residual_1]
        )
        sigmoid = graph.add_node("sigmoid", torch.nn.Sigmoid(), [residual_2])
        graph.add_output(sigmoid)
        return graph


class MultiOutputModel(torch.nn.Module):
    """Model with multiple outputs."""

    def __init__(self) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(5, 3)
        self.linear2 = torch.nn.Linear(5, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning two outputs."""
        out1 = torch.nn.ReLU()(self.linear1(x))
        out2 = torch.nn.Sigmoid()(self.linear2(x))
        return out1, out2

    def to_graph_module(self) -> GraphModule:
        """Convert to GraphModule with multiple outputs."""
        graph = GraphModule()
        input = graph.add_input("input")
        linear1_out = graph.add_node("linear1", self.linear1, args=[input])
        relu_out = graph.add_node("relu", torch.nn.ReLU(), args=[linear1_out])
        linear2_out = graph.add_node("linear2", self.linear2, args=[input])
        sigmoid_out = graph.add_node("sigmoid", torch.nn.Sigmoid(), args=[linear2_out])
        graph.add_output(relu_out, sigmoid_out)
        return graph


class ConstReturn(torch.nn.Module):
    """A model that returns only a constant value."""

    def forward(self, _: Any, c: Any) -> Any:
        """Return the constant `c`, ignoring the first argument."""
        return c


class ConstReturnKwargs(torch.nn.Module):
    """A model that returns only a keyword argument."""

    def forward(self, _: Any, const_kwarg: Any = None) -> Any:
        """Return the `const_kwarg` keyword argument, ignoring the first argument."""
        return const_kwarg


class SmallAttn(torch.nn.Module):
    """Tiny self-attention block (q/k/v/out projections + SDPA)."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(dim, dim, bias=False)
        self.k_proj = torch.nn.Linear(dim, dim, bias=False)
        self.v_proj = torch.nn.Linear(dim, dim, bias=False)
        self.out_proj = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> Any:
        """Compute scaled dot-product attention over q/k/v projections."""
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return self.out_proj(attn)

    def to_graph_module(self) -> GraphModule:
        """Transform 'SmallAttn' to a GraphModule."""
        graph = GraphModule()
        inp = graph.add_input("x")
        q = graph.add_node("q_proj", self.q_proj, [inp])
        k = graph.add_node("k_proj", self.k_proj, [inp])
        v = graph.add_node("v_proj", self.v_proj, [inp])
        attn = graph.add_node("sdpa", torch.nn.functional.scaled_dot_product_attention, [q, k, v])
        out = graph.add_node("out_proj", self.out_proj, [attn])
        graph.add_output(out)
        return graph


class SmallMLP(torch.nn.Module):
    """Tiny two-layer MLP with a ReLU activation."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.up = torch.nn.Linear(dim, dim, bias=False)
        self.down = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> Any:
        """Forward pass: down(relu(up(x)))."""
        return self.down(torch.nn.functional.relu(self.up(x)))

    def to_graph_module(self) -> GraphModule:
        """Transform 'SmallMLP' to a GraphModule."""
        graph = GraphModule()
        inp = graph.add_input("x")
        up = graph.add_node("up", self.up, [inp])
        act = graph.add_node("act", torch.nn.functional.relu, [up])
        down = graph.add_node("down", self.down, [act])
        graph.add_output(down)
        return graph


class DecoderLayer(torch.nn.Module):
    """Transformer-shaped decoder layer wrapping an attention and an MLP fold."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.attn = SmallAttn(dim)
        self.mlp = SmallMLP(dim)

    def forward(self, x: torch.Tensor) -> Any:
        """Forward pass: mlp(attn(x))."""
        return self.mlp(self.attn(x))

    def to_graph_module(self) -> GraphModule:
        """Transform 'DecoderLayer' to a GraphModule."""
        graph = GraphModule()
        inp = graph.add_input("x")
        (attn_out,) = graph.add_subgraph(
            "attn", self.attn.to_graph_module(), [inp], original_module=self.attn
        )
        (mlp_out,) = graph.add_subgraph(
            "mlp", self.mlp.to_graph_module(), [attn_out], original_module=self.mlp
        )
        graph.add_output(mlp_out)
        return graph


class TwoLayerModel(torch.nn.Module):
    """Two stacked decoder layers; exercises nested-fold resolution reduction."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.layer_0 = DecoderLayer(dim)
        self.layer_1 = DecoderLayer(dim)

    def forward(self, x: torch.Tensor) -> Any:
        """Forward pass: layer_1(layer_0(x))."""
        return self.layer_1(self.layer_0(x))

    def to_graph_module(self) -> GraphModule:
        """Transform 'TwoLayerModel' to a GraphModule."""
        graph = GraphModule()
        inp = graph.add_input("x")
        (layer_0_out,) = graph.add_subgraph(
            "layer_0", self.layer_0.to_graph_module(), [inp], original_module=self.layer_0
        )
        (layer_1_out,) = graph.add_subgraph(
            "layer_1",
            self.layer_1.to_graph_module(),
            [layer_0_out],
            original_module=self.layer_1,
        )
        graph.add_output(layer_1_out)
        return graph


class DualOutLayer(torch.nn.Module):
    """Layer with two output leaves consumed independently downstream."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.left = torch.nn.Linear(dim, dim, bias=False)
        self.right = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (left(x), right(x))."""
        return self.left(x), self.right(x)

    def to_graph_module(self) -> GraphModule:
        """Transform 'DualOutLayer' to a GraphModule."""
        graph = GraphModule()
        inp = graph.add_input("x")
        left = graph.add_node("left", self.left, [inp])
        right = graph.add_node("right", self.right, [inp])
        graph.add_output(left, right)
        return graph


class DualOutModel(torch.nn.Module):
    """Wraps a multi-output fold so we can test coarse rewiring of tuple outputs."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.layer = DualOutLayer(dim)
        self.combine = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> Any:
        """Forward pass: combine(a + b) where (a, b) = layer(x)."""
        a, b = self.layer(x)
        return self.combine(a + b)

    def to_graph_module(self) -> GraphModule:
        """Transform 'DualOutModel' to a GraphModule."""
        graph = GraphModule()
        inp = graph.add_input("x")
        a, b = graph.add_subgraph(
            "layer", self.layer.to_graph_module(), [inp], original_module=self.layer
        )
        merged = graph.add_node("merge", torch.add, [a, b])
        out = graph.add_node("combine", self.combine, [merged])
        graph.add_output(out)
        return graph


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
