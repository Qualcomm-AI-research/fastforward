# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Shared model definitions for the orchestration test suite.

Every `torch.nn.Module` used by the orchestration tests lives here rather than in
the individual test files, so that models can be reused and fused. Fixtures that
wrap these models live in `conftest.py`.

Groups below:
  * Graph-module / instruction-engine models (built via `to_graph_module()`).
  * Small helper modules (constants, tuple outputs, no-input generators, probes).
  * Tracer-shape probes (exercise specific `torch.export` / FX behaviours).
  * Llama-shaped "toy" family (mirrors HuggingFace naming so mpath patterns match).
"""

from typing import Any

import torch

from fastforward._orchestration.graph_module import GraphModule

# ---------------------------------------------------------------------------
# Graph-module / instruction-engine models (built via `to_graph_module()`).
# ---------------------------------------------------------------------------


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


class CatModel(torch.nn.Module):
    """Passes a list of tensors to torch.cat, forcing a synthesized container node."""

    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate the projected input with the raw input along the last dim."""
        return torch.cat([self.fc(x), x], dim=-1)


class TwoLinear(torch.nn.Module):
    """Minimal traceable model with two independently-targetable Linear leaves."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(8, 8)
        self.act = torch.nn.SiLU()
        self.fc2 = torch.nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: fc2(act(fc1(x)))."""
        return self.fc2(self.act(self.fc1(x)))  # type: ignore[no-any-return]


class TinyModel(torch.nn.Module):
    """Two Linear leaves plus a Conv2d; used for registry target resolution."""

    def __init__(self) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(4, 4)
        self.linear2 = torch.nn.Linear(4, 4)
        self.conv = torch.nn.Conv2d(3, 3, 1)


# ---------------------------------------------------------------------------
# Small helper modules.
# ---------------------------------------------------------------------------


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


class AddConstant(torch.nn.Module):
    """Adds a (constant) second argument to its input."""

    def forward(self, x: torch.Tensor, const: int) -> torch.Tensor:
        """Return x + const."""
        return x + const


class ReturnTuple(torch.nn.Module):
    """Returns a 2-tuple `(x, x * 2)` to exercise tuple outputs / AttributeRef."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (x, x * 2)."""
        return x, x * 2


class RNGTensor(torch.nn.Module):
    """Generate a tensor from no inputs (exercises zero-input nodes)."""

    def forward(self) -> torch.Tensor:
        """Return a fresh random tensor."""
        return torch.randn(5)


class ProbeModule(torch.nn.Module):
    """Records whether torch inference mode was enabled during forward."""

    def __init__(self) -> None:
        super().__init__()
        self.is_on_inference_mode = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Record the inference-mode flag and return the input unchanged."""
        self.is_on_inference_mode = torch.is_inference_mode_enabled()
        return x


# ---------------------------------------------------------------------------
# Small attention / MLP / decoder family (built via `to_graph_module()`).
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tracer-shape probes (exercise specific torch.export / FX behaviours).
# ---------------------------------------------------------------------------


class TinyMLP(torch.nn.Module):
    """Three-leaf MLP (fc1 / act / fc2); the canonical tracer happy-path model."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(8, 16)
        self.act = torch.nn.SiLU()
        self.fc2 = torch.nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: fc2(act(fc1(x)))."""
        y: torch.Tensor = self.fc2(self.act(self.fc1(x)))
        return y


class TupleOut(torch.nn.Module):
    """Returns a tuple to exercise output_unflatten."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Return a 1-tuple wrapping fc(x)."""
        return (self.fc(x),)


class MultiAxisIndex(torch.nn.Module):
    """Exercises unflatten-arg for tuple/None/slice indexing."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Index with a multi-axis None/slice key."""
        return x[None, :, None]


class WithBuffer(torch.nn.Module):
    """Module with a registered buffer accessed during forward."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor([2.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Multiply the input by the registered buffer."""
        assert isinstance(self.scale, torch.Tensor)
        return x * self.scale


class KwargForward(torch.nn.Module):
    """Exercises l_kw_*_ unmangling."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor, *, scale: torch.Tensor) -> torch.Tensor:
        """Forward pass with a keyword-only `scale`: fc(x) * scale."""
        y: torch.Tensor = self.fc(x) * scale
        return y


class NestedMLP(torch.nn.Module):
    """Two-level nesting to exercise add_subgraph scoping."""

    def __init__(self) -> None:
        super().__init__()
        self.block = TinyMLP()
        self.head = torch.nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: head(block(x))."""
        return self.head(self.block(x))  # type: ignore[no-any-return]


class MixedOps(torch.nn.Module):
    """Mix of nn.Module calls, tensor ops (transpose, mul), and a buffer access.

    Used to verify that the tracer assigns the correct Op kind per node:
    nn.Module calls become Op.torch_module (callable is the Module), tensor
    ops become Op.call_function (callable is an aten op, not a Module), buffer
    reads become Op.get_attr (callable is a closure).
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(8, 8)
        self.act = torch.nn.SiLU()
        self.register_buffer("scale", torch.tensor(2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass mixing module calls, transposes, and a buffer read."""
        z = self.fc(x)
        z = self.act(z)
        # transpose twice keeps shapes valid while emitting two call_function nodes
        y = torch.transpose(z, -1, -2)
        out = torch.transpose(y, -1, -2)
        assert isinstance(self.scale, torch.Tensor)
        return out * self.scale


class NestedWithTensorOps(torch.nn.Module):
    """Nests `MixedOps` so tensor ops live inside an inlined subgraph.

    `add_subgraph` used to drop the `op` field of inlined nodes, so an aten op
    nested inside `block` would surface in the parent graph as Op.torch_module
    with an OpOverload as its target. This model pins that contract.
    """

    def __init__(self) -> None:
        super().__init__()
        self.block = MixedOps()  # contains fc + act + transpose + mul + buffer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass delegating entirely to the nested block."""
        return self.block(x)  # type: ignore[no-any-return]


class DeviceCastModel(torch.nn.Module):
    """Follows the input's runtime device via `.float().to(x.device)`.

    torch.export bakes `x.device` as a cpu literal; ff.trace strips it. Used to
    verify the device-agnostic cast handling.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("inv_freq", torch.ones(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Multiply by a buffer cast to the input's runtime device."""
        assert isinstance(self.inv_freq, torch.Tensor)
        return x * self.inv_freq.float().to(x.device)


# ---------------------------------------------------------------------------
# Llama-shaped "toy" family. Mirrors HuggingFace LlamaForCausalLM naming so the
# GPTQ notebook's mpath patterns (e.g. "*/q_proj", "*/gate_proj") match the same
# way: model.layers.{i}.self_attn.{q,k,v,o}_proj / .mlp.{gate,up,down}_proj.
# Only the structure matters — the math is a stand-in (no real attention/SiLU).
# ---------------------------------------------------------------------------


class ToyAttention(torch.nn.Module):
    """Llama-named attention block (q/k/v/o_proj) with a stand-in forward."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(dim, dim)
        self.k_proj = torch.nn.Linear(dim, dim)
        self.v_proj = torch.nn.Linear(dim, dim)
        self.o_proj = torch.nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: o_proj(q_proj(x) + k_proj(x) + v_proj(x))."""
        return self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))  # type: ignore[no-any-return]


class ToyMLP(torch.nn.Module):
    """Llama-named MLP block (gate/up/down_proj) with a stand-in forward."""

    def __init__(self, dim: int, ff_mult: int = 2) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(dim, dim * ff_mult)
        self.up_proj = torch.nn.Linear(dim, dim * ff_mult)
        self.down_proj = torch.nn.Linear(dim * ff_mult, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: down_proj(gate_proj(x) * up_proj(x))."""
        return self.down_proj(self.gate_proj(x) * self.up_proj(x))  # type: ignore[no-any-return]


class ToyDecoderLayer(torch.nn.Module):
    """Llama-named decoder layer (self_attn + mlp) with residual adds."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.self_attn = ToyAttention(dim)
        self.mlp = ToyMLP(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections around attn and mlp."""
        x = x + self.self_attn(x)
        x = x + self.mlp(x)
        return x


class ToyLlama(torch.nn.Module):
    """Llama-shaped stack of decoder layers behind an nn.ModuleList."""

    def __init__(self, dim: int = 16, n_layers: int = 2) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([ToyDecoderLayer(dim) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: run the input through each decoder layer in order."""
        for layer in self.layers:
            x = layer(x)
        return x
