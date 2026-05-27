# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

# pylint: disable=missing-function-docstring
import functools
import importlib.util

from collections.abc import Iterable

import pytest
import torch

from fastforward._orchestration.graph_module import (
    GraphModule,
    Op,
    SubgraphSpec,
    inference_mode,
    local_optimize,
)
from fastforward._orchestration.trace import _MIN_TORCH_VERSION, trace
from packaging.version import Version
from torch import nn

pytestmark = pytest.mark.skipif(
    Version(torch.__version__.split("+", 1)[0]) < _MIN_TORCH_VERSION,
    reason=f"requires PyTorch >= {_MIN_TORCH_VERSION}",
)

# Transformers and torchvision are not in the project's test deps (only docs).
skip_without_torchvision = pytest.mark.skipif(
    importlib.util.find_spec("torchvision") is None,
    reason="torchvision is not installed",
)


def _has_transformers_min_version(min_version: str) -> bool:
    if importlib.util.find_spec("transformers") is None:
        return False
    import transformers  # type: ignore[import-untyped]

    from packaging.version import Version

    return Version(transformers.__version__) >= Version(min_version)


skip_without_transformers = pytest.mark.skipif(
    not _has_transformers_min_version("4.51"),
    reason="transformers >= 4.51 is not installed",
)


class _TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y: torch.Tensor = self.fc2(self.act(self.fc1(x)))
        return y


class _TupleOut(nn.Module):
    """Returns a tuple to exercise output_unflatten."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (self.fc(x),)


class _MultiAxisIndex(nn.Module):
    """Exercises unflatten-arg for tuple/None/slice indexing."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[None, :, None]


class _WithBuffer(nn.Module):
    """Module with a registered buffer accessed during forward."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor([2.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(self.scale, torch.Tensor)
        return x * self.scale


class _KwargForward(nn.Module):
    """Exercises l_kw_*_ unmangling."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor, *, scale: torch.Tensor) -> torch.Tensor:
        y: torch.Tensor = self.fc(x) * scale
        return y


class _NestedMLP(nn.Module):
    """Two-level nesting to exercise add_subgraph scoping."""

    def __init__(self) -> None:
        super().__init__()
        self.block = _TinyMLP()
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.block(x))  # type: ignore[no-any-return]


class _MixedOps(nn.Module):
    """Mix of nn.Module calls, tensor ops (transpose, mul), and a buffer access.

    Used to verify that the tracer assigns the correct Op kind per node:
    nn.Module calls become Op.torch_module (callable is the Module), tensor
    ops become Op.call_function (callable is an aten op, not a Module), buffer
    reads become Op.get_attr (callable is a closure).
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 8)
        self.act = nn.SiLU()
        self.register_buffer("scale", torch.tensor(2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.fc(x)
        z = self.act(z)
        # transpose twice keeps shapes valid while emitting two call_function nodes
        y = torch.transpose(z, -1, -2)
        out = torch.transpose(y, -1, -2)
        assert isinstance(self.scale, torch.Tensor)
        return out * self.scale


# ---------------------------------------------------------------------------
# Toy Llama-shaped model used for the GPTQ-pipeline smoke test. The structure
# mirrors HuggingFace's LlamaForCausalLM well enough that mpath patterns from
# the GPTQ notebook (e.g. "*/q_proj", "*/gate_proj") match the same way:
#   model.layers.{i}.self_attn.{q,k,v,o}_proj
#   model.layers.{i}.mlp.{gate,up,down}_proj
# Only the structure matters — the math is a stand-in (no attention, no SiLU).
# ---------------------------------------------------------------------------


class _ToyAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))  # type: ignore[no-any-return]


class _ToyMLP(nn.Module):
    def __init__(self, dim: int, ff_mult: int = 2) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, dim * ff_mult)
        self.up_proj = nn.Linear(dim, dim * ff_mult)
        self.down_proj = nn.Linear(dim * ff_mult, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.gate_proj(x) * self.up_proj(x))  # type: ignore[no-any-return]


class _ToyDecoderLayer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.self_attn = _ToyAttention(dim)
        self.mlp = _ToyMLP(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(x)
        x = x + self.mlp(x)
        return x


class _ToyLlama(nn.Module):
    def __init__(self, dim: int = 16, n_layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_ToyDecoderLayer(dim) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def test_trace_tiny_mlp_returns_graph_module() -> None:
    # GIVEN a tiny MLP module and an example input
    model = _TinyMLP().eval()
    x = torch.randn(2, 8)

    # WHEN the module is traced
    graph = trace(model, x)

    # THEN the returned object is a GraphModule with at least one input
    assert isinstance(graph, GraphModule)
    assert graph.input_names


def test_trace_tiny_mlp_preserves_module_nodes() -> None:
    # GIVEN a tiny MLP module with three submodules (fc1, act, fc2)
    model = _TinyMLP().eval()
    x = torch.randn(2, 8)

    # WHEN the module is traced
    graph = trace(model, x)

    # THEN each submodule is represented as a torch_module node
    module_nodes = {node.name: node for node in graph._nodes.values() if node.op == Op.torch_module}
    assert "fc1" in module_nodes
    assert "act" in module_nodes
    assert "fc2" in module_nodes


def test_trace_recovers_leaf_modules_by_identity() -> None:
    # GIVEN a tiny MLP with parametric and parameter-free leaf submodules
    model = _TinyMLP().eval()
    x = torch.randn(2, 8)

    # WHEN the module is traced
    graph = trace(model, x)

    # THEN each leaf submodule is the original object (not a copy), reachable both
    # via dotted-path submodule lookup and via reverse module-to-NodeRef lookup
    assert graph.get_submodule("fc1") is model.fc1
    assert graph.get_submodule("fc2") is model.fc2
    assert graph.get_submodule("act") is model.act

    # node_ref(module) is the canonical reverse lookup used by SubgraphSpec etc.
    assert graph.node_ref(model.fc1).name == "fc1"
    assert graph.node_ref(model.fc2).name == "fc2"
    assert graph.node_ref(model.act).name == "act"


def test_trace_recovers_nested_modules_by_dotted_path() -> None:
    # GIVEN a model with a nested submodule (block.fc1, block.fc2, ...)
    model = _NestedMLP().eval()
    x = torch.randn(2, 8)

    # WHEN the module is traced
    graph = trace(model, x)

    # THEN leaf modules at any depth are recoverable via dotted lookup
    # (add_subgraph scopes the inner module names with the parent prefix)
    assert graph.get_submodule("block.fc1") is model.block.fc1
    assert graph.get_submodule("block.fc2") is model.block.fc2
    assert graph.get_submodule("block.act") is model.block.act
    assert graph.get_submodule("head") is model.head

    # And the reverse lookup (module -> NodeRef) finds nested modules too —
    # add_subgraph inlines subgraph nodes into the parent's flat _nodes dict
    assert graph.node_ref(model.block.fc1).name == "block.fc1"
    assert graph.node_ref(model.block.fc2).name == "block.fc2"
    assert graph.node_ref(model.head).name == "head"


def test_trace_recovers_module_list_children_by_numeric_dotted_path() -> None:
    # GIVEN a model with nn.ModuleList (layers.0, layers.1, ...)
    model = _ToyLlama(dim=16, n_layers=2).eval()
    x = torch.randn(1, 16)

    # WHEN the module is traced
    graph = trace(model, x)
    layer0 = model.layers[0]
    layer1 = model.layers[1]
    assert isinstance(layer0, _ToyDecoderLayer)
    assert isinstance(layer1, _ToyDecoderLayer)

    # THEN leaf modules behind numeric indices are recoverable via dotted lookup
    assert graph.get_submodule("layers.0.self_attn.q_proj") is layer0.self_attn.q_proj
    assert graph.get_submodule("layers.1.mlp.gate_proj") is layer1.mlp.gate_proj

    # And the reverse lookup resolves numeric path segments correctly
    assert graph.node_ref(layer0.self_attn.q_proj).name == "layers.0.self_attn.q_proj"
    assert graph.node_ref(layer1.mlp.gate_proj).name == "layers.1.mlp.gate_proj"


def test_trace_nested_model_forward_matches_eager() -> None:
    # GIVEN a nested model with a sub-block of submodules
    model = _NestedMLP().eval()
    x = torch.randn(2, 8)
    with torch.no_grad():
        expected = model(x)

    # WHEN the nested model is traced and the graph is run
    graph = trace(model, x)
    with torch.no_grad():
        got = graph(x)

    # THEN the graph output matches the eager output (subgraph wiring is correct)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_trace_tiny_mlp_forward_matches_eager() -> None:
    # GIVEN a tiny MLP module and the eager-mode forward output for an example input
    model = _TinyMLP().eval()
    x = torch.randn(2, 8)
    with torch.no_grad():
        expected = model(x)

    # WHEN the module is traced and the resulting graph is run with the same input
    graph = trace(model, x)
    with torch.no_grad():
        got = graph(x)

    # THEN the graph's forward output matches the eager forward output element-wise
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_trace_preserves_output_structure_tuple() -> None:
    # GIVEN a module whose forward returns a 1-tuple of tensors
    model = _TupleOut().eval()
    x = torch.randn(2, 4)
    with torch.no_grad():
        expected = model(x)

    # WHEN the module is traced and the graph is run
    graph = trace(model, x)
    with torch.no_grad():
        got = graph(x)

    # THEN the graph returns a tuple (not a flat tensor) with the same content
    assert type(got) is type(expected)
    assert isinstance(got, tuple)
    assert len(got) == 1
    torch.testing.assert_close(got[0], expected[0], atol=1e-5, rtol=1e-5)


def test_trace_kwarg_input_names_match_module_signature() -> None:
    # GIVEN a module whose forward signature includes a keyword-only argument `scale`
    model = _KwargForward().eval()
    x = torch.randn(2, 4)
    scale = torch.randn(4)
    with torch.no_grad():
        expected = model(x, scale=scale)

    # WHEN the module is traced with the kwarg and the graph is invoked the same way
    graph = trace(model, x, scale=scale)
    with torch.no_grad():
        got = graph(x, scale=scale)

    # THEN the graph exposes the input under its original name (no l_kw_*_ leakage)
    #      and the forward output matches the eager output
    assert "scale" in graph.input_names
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_trace_constant_indexing_via_unflatten_arg() -> None:
    # GIVEN a module that indexes into a tensor with a multi-axis None/slice key
    model = _MultiAxisIndex().eval()
    x = torch.randn(8)
    with torch.no_grad():
        expected = model(x)

    # WHEN the module is traced (which forces _unflatten_arg to wrap the indexing tuple)
    graph = trace(model, x)
    with torch.no_grad():
        got = graph(x)

    # THEN the graph's forward output matches the eager output
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_trace_with_buffer_forward_matches_eager() -> None:
    # GIVEN a module that reads from a registered buffer during forward
    model = _WithBuffer().eval()
    x = torch.randn(4)
    with torch.no_grad():
        expected = model(x)

    # WHEN the module is traced and the graph is run
    graph = trace(model, x)
    with torch.no_grad():
        got = graph(x)

    # THEN the buffer was resolved correctly and the output matches eager
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)

    # THEN calling .to() on the resulting graph does not raise
    moved = graph.to("cpu")
    assert isinstance(moved, GraphModule)


def test_trace_shares_weights_by_identity() -> None:
    # GIVEN a tiny MLP that has been traced into a graph
    model = _TinyMLP().eval()
    x = torch.randn(2, 8)
    graph = trace(model, x)

    # WHEN a parameter on the original module is mutated in-place after tracing
    with torch.no_grad():
        model.fc1.weight.zero_()

    # THEN re-running the graph reflects the mutation (weights are shared by identity)
    with torch.no_grad():
        got = graph(x)
        expected = model(x)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_trace_backward_matches_eager() -> None:
    # GIVEN a tiny MLP in train mode and the eager gradients of `loss = model(x).sum()`
    model = _TinyMLP()
    x = torch.randn(2, 8)
    eager_out = model(x).sum()
    eager_grads = torch.autograd.grad(eager_out, list(model.parameters()))

    # WHEN the module is traced and the same loss is computed through the graph,
    #      gradients are collected via torch.autograd.grad (no in-place accumulation)
    graph = trace(model, x)
    graph_out = graph(x).sum()
    graph_grads = torch.autograd.grad(graph_out, list(model.parameters()))

    # THEN each parameter's gradient through the graph matches its eager gradient
    for eg, gg in zip(eager_grads, graph_grads, strict=True):
        torch.testing.assert_close(eg, gg, atol=1e-5, rtol=1e-5)


def test_trace_module_calls_emit_torch_module_op_with_module_callable() -> None:
    # GIVEN a model mixing nn.Module calls, tensor ops, and a buffer read
    model = _MixedOps().eval()
    x = torch.randn(2, 8)

    # WHEN the model is traced
    graph = trace(model, x)
    torch_module_nodes = [n for n in graph._nodes.values() if n.op is Op.torch_module]

    # THEN every torch_module node carries an actual nn.Module instance, and the
    # set of those instances is exactly the user's leaf modules — not aten ops
    assert len(torch_module_nodes) == 2  # fc + act
    for node in torch_module_nodes:
        assert isinstance(node.module, nn.Module), (
            f"Op.torch_module node {node.name!r} has non-Module callable "
            f"{type(node.module).__name__}"
        )
    assert {id(n.module) for n in torch_module_nodes} == {id(model.fc), id(model.act)}


def test_trace_tensor_ops_emit_call_function_op_with_non_module_callable() -> None:
    # GIVEN a traced model with torch.transpose and aten.mul calls
    model = _MixedOps().eval()
    x = torch.randn(2, 8)
    graph = trace(model, x)

    # WHEN we filter for call_function nodes (excluding the synthetic _prepare_output
    # node the tracer adds at the root output)
    call_function_nodes = [
        n for n in graph._nodes.values() if n.op is Op.call_function and n.name != "_prepare_output"
    ]

    # THEN there is at least one call_function node and none of them are nn.Modules
    assert len(call_function_nodes) >= 3  # 2x transpose + mul (transpose may decompose further)
    for node in call_function_nodes:
        assert callable(node.module), f"call_function node {node.name!r} not callable"
        assert not isinstance(node.module, nn.Module), (
            f"call_function node {node.name!r} should not wrap an nn.Module "
            f"(got {type(node.module).__name__})"
        )


def test_trace_buffer_access_emits_get_attr_op_with_closure_callable() -> None:
    # GIVEN a traced model with a registered buffer read in forward
    model = _MixedOps().eval()
    x = torch.randn(2, 8)
    graph = trace(model, x)

    # WHEN we collect get_attr nodes
    get_attr_nodes = [n for n in graph._nodes.values() if n.op is Op.get_attr]

    # THEN there is one get_attr node for the buffer, its callable is a no-arg
    # closure, and calling it returns the original buffer tensor
    assert len(get_attr_nodes) == 1
    [node] = get_attr_nodes
    module = node.module
    assert callable(module)
    assert not isinstance(module, nn.Module)
    resolved = module()
    assert isinstance(resolved, torch.Tensor)
    assert isinstance(model.scale, torch.Tensor)
    assert resolved.data_ptr() == model.scale.data_ptr()


def test_trace_node_op_invariants_hold_for_every_node() -> None:
    # GIVEN a traced model with a representative mix of operations
    model = _MixedOps().eval()
    x = torch.randn(2, 8)
    graph = trace(model, x)

    # WHEN we walk every node in the graph
    seen_ops: set[Op] = set()
    for node in graph._nodes.values():
        seen_ops.add(node.op)

        # THEN each node satisfies the contract for its op kind
        match node.op:
            case Op.torch_module:
                assert isinstance(node.module, nn.Module), (
                    f"{node.name!r} has Op.torch_module but module is {type(node.module).__name__}"
                )
            case Op.call_function | Op.call_method | Op.get_attr:
                assert callable(node.module), f"{node.name!r} module is not callable"
                assert not isinstance(node.module, nn.Module), (
                    f"{node.name!r} has {node.op} but module is an nn.Module "
                    f"({type(node.module).__name__}); only Op.torch_module should "
                    f"carry an nn.Module"
                )

    # THEN the tracer exercised at least these three Op kinds for this model
    assert {Op.torch_module, Op.call_function, Op.get_attr}.issubset(seen_ops)


def test_trace_node_op_invariants_hold_through_add_subgraph_inlining() -> None:
    # GIVEN a NESTED model where tensor ops live inside a submodule. add_subgraph
    # used to drop the `op` field of inlined nodes — so an aten op (call_function)
    # nested inside `block` would surface in the parent graph as Op.torch_module
    # with an OpOverload as its `module`. That breaks OffloadEverything (it tries
    # to .to(device) the OpOverload) and any code that assumes torch_module nodes
    # wrap real nn.Modules. This test walks the inlined parent graph and pins
    # the op contract for every node.
    class _NestedWithTensorOps(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = _MixedOps()  # contains fc + act + transpose + mul + buffer

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(x)  # type: ignore[no-any-return]

    model = _NestedWithTensorOps().eval()
    x = torch.randn(2, 8)
    graph = trace(model, x)

    # WHEN we walk every node post-inlining (block.* are all in graph._nodes)
    op_kinds: set[Op] = set()
    for node in graph._nodes.values():
        op_kinds.add(node.op)
        match node.op:
            case Op.torch_module:
                assert isinstance(node.module, nn.Module), (
                    f"{node.name!r} has Op.torch_module after inlining but module "
                    f"is {type(node.module).__name__} (likely an aten op leaked from "
                    f"the subgraph; add_subgraph must preserve node.op)"
                )
            case Op.call_function | Op.call_method | Op.get_attr:
                assert not isinstance(node.module, nn.Module), (
                    f"{node.name!r} has {node.op} but module is an nn.Module "
                    f"({type(node.module).__name__})"
                )

    # THEN the inlined graph still exposes call_function nodes (e.g. transpose,
    # mul) at the parent level — they didn't get re-tagged as torch_module
    assert {Op.torch_module, Op.call_function, Op.get_attr}.issubset(op_kinds)


def _sgd_step(module: nn.Module, dataset: Iterable[torch.Tensor], lr: float) -> None:
    """Minimal optimization fn: one SGD step per calibration batch through `module`."""
    optim = torch.optim.SGD(params=module.parameters(), lr=lr)
    for batch in dataset:
        optim.zero_grad()
        loss = (module(batch) ** 2).mean()
        loss.backward()
        optim.step()


def test_trace_then_local_optimize_only_targets_specified_module() -> None:
    # GIVEN a traced TinyMLP and a SubgraphSpec targeting only fc1
    model = _TinyMLP().eval()
    x = torch.randn(2, 8)
    graph = trace(model, x)
    initial_w1 = model.fc1.weight.data.clone()
    initial_w2 = model.fc2.weight.data.clone()

    specs = [
        SubgraphSpec(model.fc1, model.fc1, fn=functools.partial(_sgd_step, lr=0.1)),
    ]
    calibration = [torch.randn(1, 8) for _ in range(4)]

    # WHEN the graph is run under local_optimize with the spec
    with local_optimize(graph, specs):
        graph(calibration)

    # THEN only fc1's weights changed; fc2 is untouched (single-spec partitioning)
    assert not torch.allclose(initial_w1, model.fc1.weight.data)
    assert torch.allclose(initial_w2, model.fc2.weight.data)


def test_trace_then_local_optimize_fn_receives_original_module_and_dataset() -> None:
    # GIVEN a traced TinyMLP — verify the delegate fn contract: it gets a callable
    # whose parameters include the targeted module's parameters, and the calibration
    # data arrives as a non-empty iterable of batches in the captured context
    model = _TinyMLP().eval()
    x = torch.randn(2, 8)
    graph = trace(model, x)
    calibration = [torch.randn(1, 8) for _ in range(3)]
    received: dict[str, object] = {}

    def spy(module: nn.Module, dataset: Iterable[torch.Tensor]) -> None:
        # Identity by id() because Tensors don't compare with `in` (uses __eq__)
        received["param_ids"] = {id(p) for p in module.parameters()}
        received["batches"] = list(dataset)

    # WHEN we run local_optimize with the spy fn
    specs = [SubgraphSpec(model.fc1, model.fc1, fn=spy)]
    with local_optimize(graph, specs):
        graph(calibration)

    # THEN the fn received fc1's parameters (only) by identity and all calibration batches
    assert id(model.fc1.weight) in received["param_ids"]  # type: ignore[operator]
    assert id(model.fc1.bias) in received["param_ids"]  # type: ignore[operator]
    assert id(model.fc2.weight) not in received["param_ids"]  # type: ignore[operator]
    assert isinstance(received["batches"], list)
    assert len(received["batches"]) == len(calibration)


def test_trace_then_inference_mode_forward_matches_eager() -> None:
    # GIVEN a traced TinyMLP and the eager forward output
    model = _TinyMLP().eval()
    x = torch.randn(2, 8)
    with torch.no_grad():
        expected = model(x)

    # WHEN the graph is run under inference_mode (the GPTQ notebook's eval path)
    graph = trace(model, x)
    with inference_mode(graph):
        got = graph(x)

    # THEN the inference-mode output matches eager element-wise
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_trace_then_local_optimize_multiple_specs_each_targets_its_module() -> None:
    # GIVEN a traced TinyMLP and two SubgraphSpecs (one per Linear) — the typical
    # GPTQ pattern where every projection gets its own single-module spec
    model = _TinyMLP().eval()
    x = torch.randn(2, 8)
    graph = trace(model, x)
    call_log: list[str] = []

    def record(name: str, _module: nn.Module, dataset: Iterable[torch.Tensor]) -> None:
        del dataset
        call_log.append(name)

    specs = [
        SubgraphSpec(model.fc1, model.fc1, fn=functools.partial(record, "fc1")),
        SubgraphSpec(model.fc2, model.fc2, fn=functools.partial(record, "fc2")),
    ]
    calibration = [torch.randn(1, 8) for _ in range(2)]

    # WHEN local_optimize runs with both specs
    with local_optimize(graph, specs):
        graph(calibration)

    # THEN each spec's fn was invoked exactly once, in topological order
    assert call_log == ["fc1", "fc2"]


@pytest.mark.slow
@skip_without_torchvision
def test_trace_resnet18_forward_matches_eager() -> None:
    import torchvision  # type: ignore[import-not-found]

    # GIVEN a fresh, untrained ResNet18 in eval mode and an example input
    model = torchvision.models.resnet18(weights=None).eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        expected = model(x)

    # WHEN the model is traced and the resulting graph is run with the same input
    graph = trace(model, x)
    with torch.no_grad():
        got = graph(x)

    # THEN the graph's forward output matches the eager forward output
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.slow
@skip_without_transformers
def test_trace_llama_7b_forward_matches_eager() -> None:
    from transformers import AutoConfig
    from transformers.models.llama.modeling_llama import (  # type: ignore[import-untyped]
        LlamaForCausalLM,
    )

    # GIVEN a config-only Llama-7B causal LM
    config = AutoConfig.from_pretrained("huggyllama/llama-7b")
    model = LlamaForCausalLM(config).eval()

    inputs = {**model.dummy_inputs, "use_cache": False}

    with torch.no_grad():
        expected = model(**inputs)

    # WHEN the full model is traced and the graph is invoked with the same inputs
    graph = trace(model, **inputs)
    with torch.no_grad():
        got = graph(**inputs)

    # THEN the graph returns the same dataclass type and logits match tightly
    assert type(got) is type(expected)
    torch.testing.assert_close(got.logits, expected.logits)


@pytest.mark.slow
@skip_without_transformers
def test_trace_qwen3_8b_forward_matches_eager() -> None:
    from transformers import AutoConfig, AutoModelForCausalLM

    # GIVEN a config-only Qwen3-8B causal LM
    config = AutoConfig.from_pretrained("Qwen/Qwen3-8B")
    model = AutoModelForCausalLM.from_config(config).eval()
    inputs = {**model.dummy_inputs, "use_cache": False}
    with torch.no_grad():
        expected = model(**inputs)

    # WHEN the full model is traced and the graph is invoked with the same inputs
    graph = trace(model, **inputs)
    with torch.no_grad():
        got = graph(**inputs)

    # THEN the graph returns the same output type and its logits match
    assert type(got) is type(expected)
    torch.testing.assert_close(got.logits, expected.logits, atol=0.05, rtol=0.01)


def test_trace_keeps_dtype_layout_cast_device_agnostic() -> None:
    """torch.export bakes x.device as a cpu literal; ff.trace strips it."""

    # GIVEN a module whose forward follows x's runtime device via .float().to(x.device)
    class _M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("inv_freq", torch.ones(4))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            assert isinstance(self.inv_freq, torch.Tensor)
            return x * self.inv_freq.float().to(x.device)

    model = _M().eval()
    example = torch.randn(1, 4)

    # WHEN torch.export captures the cast, x.device is resolved to a literal cpu kwarg
    exported = torch.export.export(model, args=(example,), strict=False)
    unflattened = torch.export.unflatten(exported)
    pre_casts = [n for n in unflattened.graph.nodes if "dtype" in n.kwargs and "device" in n.kwargs]
    assert len(pre_casts) >= 1, "torch.export should bake at least one hardcoded device kwarg"
    for node in pre_casts:
        assert node.kwargs["dtype"] is torch.float32
        assert node.kwargs["device"] == torch.device("cpu")

    # THEN ff.trace strips the literal so the cast falls back to the input's runtime device
    graph = trace(model, example)
    matches = [n for n in graph._nodes.values() if "dtype" in n.kwargs]
    assert len(matches) >= 1
    for post_cast in matches:
        assert "device" not in post_cast.kwargs

    # AND the forward pass through the GraphModule matches eager _M().forward()
    with torch.no_grad():
        expected = model(example)
        got = graph(example)

    torch.testing.assert_close(got, expected)
