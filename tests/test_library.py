# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Any

import fastforward as ff
import pytest
import torch

from fastforward import library
from fastforward.flags import compiled_quant_funcs
from fastforward.library import custom_quant_op, deregister_quant_op, register_quant_fake

# Torch 2.10 - 2.12 warn "Profiler clears events" on every profiler (even single-cycle ones):
# the check runs after the inner profiler is created, so the warning is spurious. It is emitted
# only once per process, i.e. by whichever test opens a profiler first. `acc_events=True` is not
# an option: it does not exist in torch < 2.5.
pytestmark = pytest.mark.filterwarnings("ignore:Warning. Profiler clears events:UserWarning")


@pytest.fixture
def compile_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace `torch.compile` by a recorder that returns the function unchanged.

    Returns the list of keyword arguments of every `torch.compile` call.
    """
    calls: list[dict[str, Any]] = []

    def fake_compile(fn: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    return calls


def _scale(x: torch.Tensor) -> torch.Tensor:
    return x * 2


@pytest.mark.parametrize(
    ("conditional_compile", "expect_compile"),
    [(True, True), (False, False)],
    ids=["compiled", "uncompiled"],
)
def test_custom_quant_op_compiles_only_when_enabled(
    compile_calls: list[dict[str, Any]],
    conditional_compile: bool,
    expect_compile: bool,
) -> None:
    # GIVEN an op with the given `conditional_compile` setting and compiled funcs enabled
    scale_op = custom_quant_op(
        f"test_compile_{conditional_compile}", conditional_compile=conditional_compile
    )(_scale)
    x = torch.ones(3)

    # WHEN calling the op eagerly
    with compiled_quant_funcs(True):
        out = scale_op(x)

    # THEN the result is unchanged and `torch.compile` is used only when expected
    assert torch.equal(out, x * 2)
    assert len(compile_calls) == int(expect_compile)


def test_custom_quant_op_is_compiled_when_called_from_torch_ops(
    compile_calls: list[dict[str, Any]],
) -> None:
    # GIVEN an op with conditional compilation enabled
    custom_quant_op("test_registered_op")(_scale)
    x = torch.ones(3)

    # WHEN calling the registered op with compiled funcs enabled from torch op library
    with compiled_quant_funcs(True):
        out = torch.ops.fastforward.test_registered_op(x)

    # THEN the registered op is compiled
    assert torch.equal(out, x * 2)
    assert len(compile_calls) == 1


def test_custom_quant_op_is_compiled_when_called_from_ff_ops(
    compile_calls: list[dict[str, Any]],
) -> None:
    # GIVEN an op with conditional compilation enabled
    custom_quant_op("test_registered_op_2")(_scale)
    x = torch.ones(3)

    # WHEN calling the registered op with compiled funcs enabled from fastforward op library
    with compiled_quant_funcs(True):
        out = ff.ops.test_registered_op_2(x)

    # THEN the registered op is compiled
    assert torch.equal(out, x * 2)
    assert len(compile_calls) == 1


def test_custom_quant_op_fullgraph(compile_calls: list[dict[str, Any]]) -> None:
    # GIVEN an op with `fullgraph=True`
    scale_op = custom_quant_op("test_fullgraph", fullgraph=True)(_scale)
    x = torch.ones(3)

    # WHEN calling it eagerly with compiled funcs enabled
    with compiled_quant_funcs(True):
        scale_op(x)

    # THEN `torch.compile` is called with `fullgraph=True`
    assert compile_calls == [{"fullgraph": True}]


def test_custom_quant_op_fullgraph_requires_conditional_compile() -> None:
    # GIVEN `fullgraph=True` without conditional compilation
    # WHEN decorating a function with it
    # THEN a ValueError is raised
    with pytest.raises(ValueError, match="fullgraph"):
        custom_quant_op("test_invalid_fullgraph", conditional_compile=False, fullgraph=True)


def _triple(x: torch.Tensor) -> torch.Tensor:
    return x * 3


class _ScaleModule(torch.nn.Module):
    def __init__(self, op: Any) -> None:
        super().__init__()
        self.op = op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.op(x)
        return out + 1


def _graph_targets(graph_module: torch.fx.GraphModule) -> list[Any]:
    return [node.target for node in graph_module.graph.nodes if node.op == "call_function"]


def _capture_backend(graphs: list[torch.fx.GraphModule]) -> Any:
    def backend(gm: torch.fx.GraphModule, example_inputs: Any) -> Any:
        del example_inputs
        graphs.append(gm)
        return gm.forward

    return backend


@pytest.fixture(scope="module")
def scale_op() -> Any:
    """A `custom_quant_op` (with fake impl) that is only registered once."""
    op = custom_quant_op("test_dispatch")(_scale)
    register_quant_fake("test_dispatch")(_scale)
    return op


@pytest.mark.parametrize(
    "use_export_mode", [False, True], ids=["torch_export_only", "export_mode_flag"]
)
def test_custom_quant_op_is_opaque_when_exporting(use_export_mode: bool, scale_op: Any) -> None:
    # GIVEN a module calling a custom quant op
    module = _ScaleModule(scale_op)

    # WHEN exporting it (with or without the `export_mode` flag)
    with ff.export_mode(use_export_mode):
        exported = torch.export.export(module, (torch.ones(3),))

    # THEN the registered op is an (opaque) node of the exported graph
    assert torch.ops.fastforward.test_dispatch.default in _graph_targets(exported.graph_module)


def test_custom_quant_op_is_transparent_under_torch_compile(scale_op: Any) -> None:
    # GIVEN a module calling a custom quant op
    module = _ScaleModule(scale_op)
    graphs: list[torch.fx.GraphModule] = []
    torch._dynamo.reset()

    # WHEN compiling it with `torch.compile`
    out = torch.compile(module, backend=_capture_backend(graphs))(torch.ones(3))

    # THEN the registered op is not part of the graph and the result is unchanged
    assert graphs
    assert all("fastforward" not in str(t) for gm in graphs for t in _graph_targets(gm))
    assert torch.equal(out, torch.ones(3) * 2 + 1)


def test_custom_quant_op_is_registered_in_fastforward_ops() -> None:
    # GIVEN a custom quant op
    op = custom_quant_op("test_ff_ops_registration")(_scale)

    # WHEN accessing it through `fastforward.ops`
    # THEN the dispatching callable is returned and listed
    assert ff.ops.test_ff_ops_registration is op
    assert "test_ff_ops_registration" in dir(ff.ops)
    assert torch.equal(ff.ops.test_ff_ops_registration(torch.ones(3)), torch.ones(3) * 2)


def test_fastforward_ops_unknown_op_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="no_such_op"):
        ff.ops.no_such_op  # noqa: B018


def test_fastforward_ops_exposes_affine_ops() -> None:
    # GIVEN the affine ops defined with `custom_quant_op`
    # WHEN accessing them through `fastforward.ops`
    # THEN they are the same callables as the ones defined in the affine `_ops` module
    assert ff.ops.affine_static_qdq is ff.quantization.affine._ops.affine_static_qdq_op
    assert ff.ops.affine_static_quantize is ff.quantization.affine._ops.affine_static_quantize_op
    assert ff.ops.affine_dequantize is ff.quantization.affine._ops.affine_dequantize_op


def test_custom_quant_op_output_does_not_require_grad_under_torch_compile(scale_op: Any) -> None:
    # GIVEN a compiled module calling the op and an input that requires grad
    module = _ScaleModule(scale_op)
    torch._dynamo.reset()
    compiled = torch.compile(module, backend=_capture_backend([]))
    x = torch.ones(3, requires_grad=True)

    # WHEN calling the compiled module
    out = compiled(x)

    # THEN the op does not add to the autograd graph
    assert not out.requires_grad


@pytest.mark.parametrize("use_export_mode", [False, True], ids=["eager", "export_mode"])
def test_custom_quant_op_gradient_behavior_is_consistent_across_branches(
    scale_op: Any, use_export_mode: bool
) -> None:
    # GIVEN an input that requires grad
    x = torch.ones(3, requires_grad=True)

    # WHEN calling the op directly in the eager branch or in the (non-tracing) export branch
    with ff.export_mode(use_export_mode):
        out = scale_op(x)

    # THEN the output is not part of the autograd graph in either branch
    assert not out.requires_grad
    assert out.grad_fn is None


def test_custom_quant_op_profiler_range_recorded_under_profiler(scale_op: Any) -> None:
    # GIVEN a custom quant op
    # WHEN calling the op eagerly under the profiler
    with torch.profiler.profile() as prof:
        scale_op(torch.ones(3))

    # THEN the profiler has a range named like the registered op
    keys = {event.key for event in prof.key_averages()}
    assert "fastforward::test_dispatch" in keys


def test_is_profiler_active_follows_profiler_context() -> None:
    # GIVEN no active profiler
    # THEN no profiler is detected
    assert not library._is_profiler_active()

    # WHEN a profiler is active
    with torch.profiler.profile():
        # THEN it is detected
        assert library._is_profiler_active()

    # THEN it is not detected anymore after leaving the context
    assert not library._is_profiler_active()


def _make_ff_registration_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `fastforward.ops._register_op` raise, simulating a failed ff op registration."""

    def fake_register_op(name: str, op: Any) -> None:
        del name, op
        raise RuntimeError("ff registration failed")

    monkeypatch.setattr(ff.ops, "_register_op", fake_register_op)


def test_custom_quant_op_ff_registration_failure_does_not_register_torch_op(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # GIVEN a name that is not registered yet and a failing ff op registration
    name = "test_ff_failure_fresh"
    _make_ff_registration_fail(monkeypatch)

    # WHEN decorating a function and the resulting error is caught
    with pytest.raises(RuntimeError, match="ff registration failed"):
        custom_quant_op(name)(_scale)

    # THEN neither the torch op nor the ff op is registered
    assert not hasattr(torch.ops.fastforward, name)
    assert name not in dir(ff.ops)


def test_custom_quant_op_duplicate_name_keeps_previous_ops() -> None:
    # GIVEN an op (with a fake impl) that is already registered in both torch and fastforward
    name = "test_duplicate_name"
    previous = custom_quant_op(name)(_scale)
    register_quant_fake(name)(_scale)
    x = torch.ones(3)

    # WHEN registering another function under the same name and the error is caught
    with pytest.raises(ValueError, match="already registered"):
        custom_quant_op(name)(_triple)

    # THEN the previously registered fastforward and torch ops are unchanged, fake impl included
    assert getattr(ff.ops, name) is previous
    assert torch.equal(getattr(ff.ops, name)(x), x * 2)
    assert torch.equal(getattr(torch.ops.fastforward, name)(x), x * 2)
    exported = torch.export.export(_ScaleModule(previous), (x,))
    assert getattr(torch.ops.fastforward, name).default in _graph_targets(exported.graph_module)


def _unannotated(x):  # type: ignore[no-untyped-def]  # noqa: ANN001, ANN202
    return x


@pytest.mark.parametrize("name", ["bad name", "1_leading_digit", "dash-ed"])
def test_custom_quant_op_invalid_name_is_not_registered(name: str) -> None:
    # GIVEN a name torch cannot use as an op name
    # WHEN decorating a function with it
    # THEN a ValueError is raised and nothing is registered in `fastforward.ops`
    with pytest.raises(ValueError, match="Invalid quant op name"):
        custom_quant_op(name)(_scale)
    assert name not in dir(ff.ops)


def test_custom_quant_op_invalid_signature_is_not_registered() -> None:
    # GIVEN a function torch cannot infer a schema for
    name = "test_invalid_signature"

    # WHEN decorating it
    # THEN a ValueError is raised and neither the ff op nor the torch op is registered
    with pytest.raises(ValueError, match="type annotation"):
        custom_quant_op(name)(_unannotated)
    assert name not in dir(ff.ops)
    assert not hasattr(torch.ops.fastforward, name)


def test_deregister_quant_op_removes_ff_and_torch_ops() -> None:
    # GIVEN a registered op
    name = "test_deregister"
    custom_quant_op(name)(_scale)

    # WHEN deregistering it
    deregister_quant_op(name)

    # THEN it is neither available in `fastforward.ops` nor in the torch op library
    assert name not in dir(ff.ops)
    with pytest.raises(AttributeError, match=name):
        getattr(ff.ops, name)
    assert not hasattr(torch.ops.fastforward, name)


def test_deregister_quant_op_allows_registering_a_replacement() -> None:
    # GIVEN a deregistered op
    name = "test_deregister_replace"
    custom_quant_op(name)(_scale)
    deregister_quant_op(name)
    x = torch.ones(3)

    # WHEN registering a different function (and fake impl) under the same name
    replacement = custom_quant_op(name)(_triple)
    register_quant_fake(name)(_triple)

    # THEN the replacement is the registered op in `fastforward.ops` and in the torch op library
    assert getattr(ff.ops, name) is replacement
    assert torch.equal(getattr(ff.ops, name)(x), x * 3)
    assert torch.equal(getattr(torch.ops.fastforward, name)(x), x * 3)
    exported = torch.export.export(_ScaleModule(replacement), (x,))
    assert getattr(torch.ops.fastforward, name).default in _graph_targets(exported.graph_module)


def test_deregister_quant_op_removes_fake_impl() -> None:
    # GIVEN an op with a fake impl that is deregistered
    name = "test_deregister_fake"
    custom_quant_op(name)(_scale)
    register_quant_fake(name)(_scale)
    deregister_quant_op(name)

    # WHEN registering an op under the same name without a fake impl and exporting it
    replacement = custom_quant_op(name)(_scale)

    # THEN the previous fake impl is not used
    with pytest.raises(Exception, match="fake"):  # noqa: B017
        torch.export.export(_ScaleModule(replacement), (torch.ones(3),))


def test_deregister_quant_op_missing_op() -> None:
    # GIVEN a name that is not registered
    name = "test_deregister_missing"

    # WHEN deregistering it
    # THEN an error is raised unless `missing_ok` is set
    with pytest.raises(ValueError, match="No op named"):
        deregister_quant_op(name)
    deregister_quant_op(name, missing_ok=True)


def test_fastforward_ops_deregister_op() -> None:
    # GIVEN an op registered in `fastforward.ops` only
    name = "test_ff_ops_deregister"
    ff.ops._register_op(name, _scale)
    with pytest.raises(ValueError, match="already registered"):
        ff.ops._register_op(name, _triple)

    # WHEN deregistering it
    ff.ops._deregister_op(name)

    # THEN it is removed and a second deregistration only passes with `missing_ok`
    assert name not in dir(ff.ops)
    with pytest.raises(ValueError, match="No op named"):
        ff.ops._deregister_op(name)
    ff.ops._deregister_op(name, missing_ok=True)
