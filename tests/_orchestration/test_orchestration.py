# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

# pylint: disable=missing-function-docstring
import copy
import dataclasses
import functools
import operator

from collections.abc import Iterator
from pathlib import Path

import fastforward as ff
import pytest
import torch

from fastforward._orchestration import registry
from fastforward._orchestration.data_flow import (
    FlowOp,
    Gradient,
    InputActivations,
    ModelOutputs,
    OutputActivations,
)
from fastforward._orchestration.graph_module import Region, Span
from fastforward._orchestration.instruction_engine import ActivationBundle
from fastforward._orchestration.location import BudgetLocation, DiskLocation, LocationLike
from fastforward._orchestration.registry import AlgorithmSpec, normalize
from fastforward._orchestration.trace import _MIN_TORCH_VERSION, trace
from fastforward.orchestration import Offload, _batched
from packaging.version import Version
from torch import nn
from typing_extensions import override

from ._models import KwargForward, TwoLinear
from .conftest import make_flows, sgd_step

pytestmark = pytest.mark.skipif(
    Version(torch.__version__.split("+", 1)[0]) < _MIN_TORCH_VERSION,
    reason=f"requires PyTorch >= {_MIN_TORCH_VERSION}",
)


def test_batched_returns_complete_and_partial_chunks() -> None:
    # WHEN data is split into chunks
    batches = _batched(range(5), 2)

    # THEN the final partial chunk is retained
    assert list(batches) == [(0, 1), (2, 3), (4,)]


def test_batched_consumes_data_lazily() -> None:
    # GIVEN an iterator that records consumption
    consumed: list[int] = []

    def data() -> Iterator[int]:
        for item in range(3):
            consumed.append(item)
            yield item

    # WHEN only the first chunk is requested
    batches = _batched(data(), 2)
    assert next(batches) == (0, 1)

    # THEN the remaining data has not been consumed
    assert consumed == [0, 1]


@pytest.mark.slow
def test_layerwise_optimize_targets_only_selected_module(two_linear: TwoLinear) -> None:
    # GIVEN a traceable model and calibration data
    model = two_linear.eval()
    calibration = [torch.randn(2, 8) for _ in range(4)]
    initial_w1 = model.fc1.weight.data.clone()
    initial_w2 = model.fc2.weight.data.clone()

    # GIVEN a spec that targets only fc1
    spec = AlgorithmSpec(
        fn=functools.partial(sgd_step, lr=0.1),
        selector=normalize([model.fc1]),
        flows=make_flows(),
    )

    # WHEN we run the public layerwise_optimize
    ff.layerwise_optimize(model, calibration, spec, sample_args=(calibration[0],))

    # THEN fc1's weights changed and fc2's did not (target resolution + reduction)
    assert not torch.allclose(initial_w1, model.fc1.weight.data)
    assert torch.allclose(initial_w2, model.fc2.weight.data)


def test_layerwise_optimize_with_prebuilt_graph_skips_tracing(two_linear: TwoLinear) -> None:
    # GIVEN a model whose graph we trace ahead of time
    model = two_linear.eval()
    calibration = [torch.randn(2, 8) for _ in range(3)]
    graph = trace(model, calibration[0])
    initial_w1 = model.fc1.weight.data.clone()

    # GIVEN a spec targeting fc1
    spec = AlgorithmSpec(
        fn=functools.partial(sgd_step, lr=0.1),
        selector=normalize([model.fc1]),
        flows=make_flows(),
    )

    # WHEN we pass the prebuilt graph (so layerwise_optimize does not trace again)
    ff.layerwise_optimize(model, calibration, spec, graph=graph)

    # THEN the targeted layer was still optimized through the supplied graph
    assert not torch.allclose(initial_w1, model.fc1.weight.data)


def test_layerwise_optimize_without_sample_or_graph_raises(two_linear: TwoLinear) -> None:
    # GIVEN a model and calibration data, but no example input and no graph
    model = two_linear.eval()
    calibration = [torch.randn(2, 8) for _ in range(2)]
    spec = AlgorithmSpec(fn=sgd_step, selector=normalize([model.fc1]), flows=make_flows())

    # WHEN we omit both, there is nothing to trace with
    # THEN a TypeError is raised instead of a guess at the example input
    with pytest.raises(TypeError, match="needs an example input"):
        ff.layerwise_optimize(model, calibration, spec)


def test_layerwise_optimize_with_graph_and_sample_raises(two_linear: TwoLinear) -> None:
    # GIVEN a model with a pre-built graph
    model = two_linear.eval()
    calibration = [torch.randn(2, 8) for _ in range(2)]
    graph = trace(model, calibration[0])
    spec = AlgorithmSpec(fn=sgd_step, selector=normalize([model.fc1]), flows=make_flows())

    # WHEN we also pass an example input, which the given graph would never use
    # THEN a TypeError is raised
    with pytest.raises(TypeError, match="Cannot combine graph="):
        ff.layerwise_optimize(model, calibration, spec, graph=graph, sample_args=(calibration[0],))


@pytest.mark.slow
def test_layerwise_optimize_traces_with_both_sample_args_and_kwargs() -> None:
    # GIVEN a model whose forward takes one positional and one keyword-only input
    model = KwargForward().eval()
    calibration = [{"x": torch.randn(2, 4), "scale": torch.randn(2, 4)} for _ in range(3)]
    initial_w = model.fc.weight.data.clone()

    spec = AlgorithmSpec(
        fn=functools.partial(sgd_step, lr=0.1),
        selector=normalize([model.fc]),
        flows=make_flows(),
    )

    # WHEN the example input is split over sample_args and sample_kwargs
    ff.layerwise_optimize(
        model,
        calibration,
        spec,
        sample_args=(calibration[0]["x"],),
        sample_kwargs={"scale": calibration[0]["scale"]},
    )

    # THEN the model traced and the targeted layer was optimized
    assert not torch.allclose(initial_w, model.fc.weight.data)


def test_layerwise_optimize_with_offloading_runs_execution_context(two_linear: TwoLinear) -> None:
    # GIVEN a model, calibration data, and a CPU-to-CPU offloading strategy
    model = two_linear.eval()
    calibration = [torch.randn(2, 8) for _ in range(3)]
    cpu = torch.device("cpu")
    initial_w1 = model.fc1.weight.data.clone()
    initial_w2 = model.fc2.weight.data.clone()

    # GIVEN a spec targeting fc1
    spec = AlgorithmSpec(
        fn=functools.partial(sgd_step, lr=0.1),
        selector=normalize([model.fc1]),
        flows=make_flows(),
    )

    # WHEN we run with an offloading strategy (exercises _ExecutionContext's pass wiring)
    ff.layerwise_optimize(
        model,
        calibration,
        spec,
        sample_args=(calibration[0],),
        config=ff.OrchestrationConfig(offloading=Offload(compute=cpu, weights=cpu)),
    )

    # THEN only the targeted layer changed
    assert not torch.allclose(initial_w1, model.fc1.weight.data)
    assert torch.allclose(initial_w2, model.fc2.weight.data)


@dataclasses.dataclass(frozen=True, eq=False)
class _CountingDiskLocation(DiskLocation):
    """A disk location that counts the tensors it stores in files."""

    stored_shapes: list[torch.Size] = dataclasses.field(default_factory=list)

    @override
    def receive(self, tensor: torch.Tensor) -> torch.Tensor:
        self.stored_shapes.append(tensor.shape)
        return super().receive(tensor)


def test_file_backed_activations_match_memory_backed_results(
    two_linear: TwoLinear, tmp_path: Path
) -> None:
    # GIVEN one model that keeps activations in memory and a copy that stores them in files
    in_memory_model = two_linear.eval()
    on_disk_model = copy.deepcopy(in_memory_model)
    calibration_data = [torch.randn(2, 8) for _ in range(3)]
    cpu = torch.device("cpu")
    disk_location = _CountingDiskLocation(tmp_path)

    def run(model: TwoLinear, activation_location: LocationLike) -> None:
        spec = AlgorithmSpec(fn=sgd_step, selector=normalize([model.fc2]), flows=make_flows())
        ff.layerwise_optimize(
            model,
            calibration_data,
            spec,
            sample_args=(calibration_data[0],),
            config=ff.OrchestrationConfig(
                offloading=Offload(compute=cpu, weights=cpu, activations=activation_location)
            ),
        )

    # WHEN we optimize both
    run(in_memory_model, cpu)
    run(on_disk_model, disk_location)

    # THEN both layer outputs from every calibration batch were stored on disk
    assert len(disk_location.stored_shapes) == 2 * len(calibration_data)

    # THEN temporary activation files are cleaned up after execution
    assert list(tmp_path.iterdir()) == []

    # THEN file-based activation storage produces the same updated weights
    assert torch.equal(on_disk_model.fc2.weight.data, in_memory_model.fc2.weight.data)


def test_budgeted_activations_match_memory_backed_results(
    two_linear: TwoLinear, tmp_path: Path
) -> None:
    # GIVEN one model that keeps activations in memory and a copy with a budget for two of them
    in_memory_model = two_linear.eval()
    budget_model = copy.deepcopy(in_memory_model)
    calibration_data = [torch.randn(2, 8) for _ in range(3)]
    cpu = torch.device("cpu")
    disk_location = _CountingDiskLocation(tmp_path)
    budget_location = BudgetLocation(cpu, budget=2 * 2 * 8 * 4, spill=disk_location)

    def run(model: TwoLinear, activation_location: LocationLike) -> None:
        spec = AlgorithmSpec(fn=sgd_step, selector=normalize([model.fc2]), flows=make_flows())
        ff.layerwise_optimize(
            model,
            calibration_data,
            spec,
            sample_args=(calibration_data[0],),
            config=ff.OrchestrationConfig(
                offloading=Offload(compute=cpu, weights=cpu, activations=activation_location)
            ),
        )

    # WHEN we optimize both
    run(in_memory_model, cpu)
    run(budget_model, budget_location)

    # THEN some activations spilled, and some stayed in memory
    assert 0 < len(disk_location.stored_shapes) < 2 * len(calibration_data)

    # THEN all room and all files are given back after execution
    assert budget_location.held_bytes == 0
    assert list(tmp_path.iterdir()) == []

    # THEN a budget does not change the updated weights
    assert torch.equal(budget_model.fc2.weight.data, in_memory_model.fc2.weight.data)


def test_offloading_refuses_a_name_that_is_no_device() -> None:
    # GIVEN a compute field that names no device
    # WHEN we build the strategy
    # THEN it is refused at once, before any model runs
    with pytest.raises(ValueError, match="does not name a device"):
        Offload(compute="gpu0")


def test_offload_rejects_directory_as_compute_location(tmp_path: Path) -> None:
    # GIVEN a compute field that names a directory
    # WHEN we build the strategy
    # THEN it is refused, because computation happens on a device
    with pytest.raises(TypeError, match="'compute' must be"):
        Offload(compute=tmp_path)


def test_offloading_can_rest_weights_in_files(two_linear: TwoLinear, tmp_path: Path) -> None:
    # GIVEN one model that rests its weights in memory and a copy that rests them in files
    in_memory_model = two_linear.eval()
    on_disk_model = copy.deepcopy(in_memory_model)
    calibration_data = [torch.randn(2, 8) for _ in range(3)]
    cpu = torch.device("cpu")

    def run(model: TwoLinear, weight_location: LocationLike) -> None:
        spec = AlgorithmSpec(fn=sgd_step, selector=normalize([model.fc1]), flows=make_flows())
        ff.layerwise_optimize(
            model,
            calibration_data,
            spec,
            sample_args=(calibration_data[0],),
            config=ff.OrchestrationConfig(offloading=Offload(compute=cpu, weights=weight_location)),
        )

    # WHEN we optimize both
    run(in_memory_model, cpu)
    run(on_disk_model, tmp_path)

    # THEN resting weights in files leaves the model weights unchanged
    assert torch.equal(on_disk_model.fc1.weight.data, in_memory_model.fc1.weight.data)

    # THEN the model is restored to ordinary device storage and temporary files are removed
    assert not on_disk_model.fc1.weight.is_shared()
    assert list(tmp_path.iterdir()) == []


def test_layerwise_optimize_calls_algorithm_once_per_target(two_linear: TwoLinear) -> None:
    # GIVEN a model and an algorithm that records the modules it is invoked on
    model = two_linear.eval()
    calibration = [torch.randn(2, 8) for _ in range(2)]
    seen: list[nn.Module] = []

    def spy(module: nn.Module, bundle: ActivationBundle) -> None:
        del bundle
        seen.append(module)

    # GIVEN a spec targeting both Linear layers
    spec = AlgorithmSpec(fn=spy, selector=normalize([model.fc1, model.fc2]), flows=make_flows())

    # WHEN we optimize
    ff.layerwise_optimize(model, calibration, spec, sample_args=(calibration[0],))

    # THEN the algorithm ran exactly once per targeted module
    assert seen == [model.fc1, model.fc2]


def test_layerwise_optimize_override_restores_registry_state(two_linear: TwoLinear) -> None:
    # GIVEN an algorithm with a pre-existing registry entry (Conv2d)
    def algorithm(module: nn.Module, bundle: ActivationBundle) -> None:
        del module, bundle

    registry.register(algorithm, torch.nn.Conv2d, flows=make_flows())
    spec_before = registry._registry[algorithm]
    try:
        model = two_linear.eval()
        calibration = [torch.randn(2, 8) for _ in range(2)]

        # WHEN layerwise_optimize runs with a `targets` override
        ff.layerwise_optimize(
            model, calibration, algorithm, targets=[model.fc1], sample_args=(calibration[0],)
        )

        # THEN the original Conv2d registration is restored after the override exits
        assert registry._registry[algorithm] == spec_before
    finally:
        registry._registry._specs.pop(algorithm, None)


def test_layerwise_optimize_with_explicit_spec(two_linear: TwoLinear) -> None:
    # GIVEN a model and an explicit AlgorithmSpec targeting fc1
    model = two_linear.eval()
    calibration = [torch.randn(2, 8) for _ in range(3)]
    initial_w1 = model.fc1.weight.data.clone()
    initial_w2 = model.fc2.weight.data.clone()

    spec = AlgorithmSpec(
        fn=functools.partial(sgd_step, lr=0.1),
        selector=normalize([model.fc1]),
        flows=make_flows(),
    )

    # WHEN we pass the spec directly (no register() needed)
    ff.layerwise_optimize(model, calibration, spec, sample_args=(calibration[0],))

    # THEN fc1's weights changed and fc2's did not
    assert not torch.allclose(initial_w1, model.fc1.weight.data)
    assert torch.allclose(initial_w2, model.fc2.weight.data)


def test_layerwise_optimize_with_multiple_specs(two_linear: TwoLinear) -> None:
    # GIVEN two specs with different algorithms targeting different modules
    model = two_linear.eval()
    calibration = [torch.randn(2, 8) for _ in range(3)]
    initial_w1 = model.fc1.weight.data.clone()
    initial_w2 = model.fc2.weight.data.clone()

    spec_fc1 = AlgorithmSpec(
        fn=functools.partial(sgd_step, lr=0.1),
        selector=normalize([model.fc1]),
        flows=make_flows(),
    )
    spec_fc2 = AlgorithmSpec(
        fn=functools.partial(sgd_step, lr=0.05),
        selector=normalize([model.fc2]),
        flows=make_flows(),
    )

    # WHEN we pass both specs as a list
    ff.layerwise_optimize(model, calibration, [spec_fc1, spec_fc2], sample_args=(calibration[0],))

    # THEN both modules were optimized
    assert not torch.allclose(initial_w1, model.fc1.weight.data)
    assert not torch.allclose(initial_w2, model.fc2.weight.data)


def _double_weight(module: nn.Linear) -> None:
    """An algorithm that reads no data, and doubles the weight."""
    with torch.no_grad():
        module.weight.mul_(2.0)


@pytest.mark.parametrize("data_free_first", [True, False], ids=["data-free-first", "data-first"])
def test_layerwise_optimize_runs_the_specs_on_one_layer_in_list_order(
    two_linear: TwoLinear, data_free_first: bool
) -> None:
    # GIVEN a spec on fc1 that reads no data, and one that reads the original inputs of fc1.
    # The two steps give a different result in the other order.
    model = two_linear.eval()
    reference = copy.deepcopy(model)
    calibration = [torch.randn(2, 8) for _ in range(3)]
    double = AlgorithmSpec(fn=_double_weight, selector=normalize([model.fc1]), flows=[])
    step = AlgorithmSpec(
        fn=sgd_step, selector=normalize([model.fc1]), flows=[InputActivations("original")]
    )
    specs = [double, step] if data_free_first else [step, double]

    # WHEN we optimize
    ff.layerwise_optimize(model, calibration, specs, sample_args=(calibration[0],))

    # THEN fc1 is the result of the two steps run eagerly, in list order, and fc2 is unchanged
    for spec in specs:
        if spec is double:
            _double_weight(reference.fc1)
        else:
            sgd_step(reference.fc1, [((batch,), {}) for batch in calibration])  # type: ignore[arg-type]
    torch.testing.assert_close(model.fc1.weight, reference.fc1.weight)
    torch.testing.assert_close(model.fc2.weight, reference.fc2.weight)


def test_layerwise_optimize_runs_specs_that_read_no_data_on_one_layer(
    two_linear: TwoLinear,
) -> None:
    # GIVEN two specs on fc1 that read no data
    model = two_linear.eval()
    initial = model.fc1.weight.detach().clone()
    calibration = [torch.randn(2, 8) for _ in range(2)]
    spec = AlgorithmSpec(fn=_double_weight, selector=normalize([model.fc1]), flows=[])

    # WHEN we optimize
    ff.layerwise_optimize(model, calibration, [spec, spec], sample_args=(calibration[0],))

    # THEN both ran
    torch.testing.assert_close(model.fc1.weight, 4 * initial)


@pytest.mark.parametrize("chunk_size", [None, 2], ids=["whole", "chunked"])
def test_layerwise_optimize_gives_each_spec_on_one_layer_the_result_of_its_flowop(
    two_linear: TwoLinear, chunk_size: int | None
) -> None:
    # GIVEN two specs on fc1 that read an op over the original outputs of fc1, and the first
    # spec changes the weight
    model = two_linear.eval()
    calibration = [torch.randn(2, 5, 8) for _ in range(3)]
    with torch.no_grad():
        expected = sum(model.fc1(batch).sum() for batch in calibration)
    received: list[torch.Tensor] = []

    def record_then_double(module: nn.Linear, total: torch.Tensor) -> None:
        received.append(total)
        _double_weight(module)

    def record(module: nn.Module, total: torch.Tensor) -> None:
        del module
        received.append(total)

    op = FlowOp(_total, combine=operator.add, activations=OutputActivations("original"))
    specs = [
        AlgorithmSpec(fn=fn, selector=normalize([model.fc1]), flows=[op])
        for fn in (record_then_double, record)
    ]

    # WHEN the model is optimized
    with torch.no_grad():
        ff.layerwise_optimize(
            model,
            calibration,
            specs,
            sample_args=(calibration[0],),
            config=ff.OrchestrationConfig(chunk_size=chunk_size),
        )

    # THEN both specs received the result of the op on the weights from before the change
    assert len(received) == 2
    for total in received:
        torch.testing.assert_close(total, expected)


def _squared_outputs(outputs: ActivationBundle) -> Iterator[torch.Tensor]:
    """Yield the sum of the squared model outputs for every batch."""
    for (output,), _ in outputs:
        yield output.pow(2).sum()


def test_layerwise_optimize_gives_each_spec_on_one_layer_the_original_gradient(
    two_linear: TwoLinear,
) -> None:
    # GIVEN two specs on fc1 that read the gradient of the original outputs of fc1, and change
    # the weight
    model = two_linear.eval()
    calibration = [torch.randn(2, 8) for _ in range(2)]
    expected = []
    for batch in calibration:
        hidden = model.fc1(batch).detach().requires_grad_()
        model.fc2(model.act(hidden)).pow(2).sum().backward()
        assert hidden.grad is not None
        expected.append(hidden.grad)
    model.zero_grad()
    received: list[list[torch.Tensor]] = []

    def record_then_double(module: nn.Linear, gradient: ActivationBundle) -> None:
        received.append([g for (g,), _ in gradient])
        _double_weight(module)

    gradient = Gradient(_squared_outputs, outputs=ModelOutputs("original"))
    spec = AlgorithmSpec(fn=record_then_double, selector=normalize([model.fc1]), flows=[gradient])

    # WHEN we optimize
    ff.layerwise_optimize(model, calibration, [spec, spec], sample_args=(calibration[0],))

    # THEN both specs received the gradient on the original weights
    assert len(received) == 2
    for gradients in received:
        torch.testing.assert_close(gradients, expected)


def test_layerwise_optimize_runs_specs_on_an_equal_span_on_one_subgraph(
    two_linear: TwoLinear,
) -> None:
    # GIVEN two specs on the same Span
    model = two_linear.eval()
    calibration = [torch.randn(2, 8) for _ in range(2)]
    seen: list[tuple[str, nn.Module]] = []

    class _SpanSelector(registry.Selector):
        def resolve(self, model: nn.Module) -> list[Region]:
            return [Span(start=model.fc1, end=model.act)]  # type: ignore[arg-type]

    def record(name: str, module: nn.Module, bundle: ActivationBundle) -> None:
        del bundle
        seen.append((name, module))

    specs = [
        AlgorithmSpec(
            fn=functools.partial(record, name), selector=_SpanSelector(), flows=make_flows()
        )
        for name in ("first", "second")
    ]

    # WHEN we optimize
    ff.layerwise_optimize(model, calibration, specs, sample_args=(calibration[0],))

    # THEN both specs ran in list order, on one subgraph
    assert [name for name, _ in seen] == ["first", "second"]
    assert seen[0][1] is seen[1][1]


def test_layerwise_optimize_rejects_specs_that_share_part_of_a_region(
    two_linear: TwoLinear,
) -> None:
    # GIVEN a spec on a Span over fc1 and act, and a spec on fc1 only
    model = two_linear.eval()
    calibration = [torch.randn(2, 8) for _ in range(2)]

    class _SpanSelector(registry.Selector):
        def resolve(self, model: nn.Module) -> list[Region]:
            return [Span(start=model.fc1, end=model.act)]  # type: ignore[arg-type]

    span = AlgorithmSpec(fn=sgd_step, selector=_SpanSelector(), flows=make_flows())
    fc1 = AlgorithmSpec(fn=sgd_step, selector=normalize([model.fc1]), flows=make_flows())

    # WHEN we optimize
    # THEN it is refused, because the order of the two specs on fc1 is not clear
    with pytest.raises(ValueError, match="equal or disjoint regions"):
        ff.layerwise_optimize(model, calibration, [span, fc1], sample_args=(calibration[0],))


def test_layerwise_optimize_with_span_region(two_linear: TwoLinear) -> None:
    # GIVEN a model and a Span covering fc1 through act
    model = two_linear.eval()
    calibration = [torch.randn(2, 8) for _ in range(4)]
    initial_w1 = model.fc1.weight.data.clone()
    initial_w2 = model.fc2.weight.data.clone()

    class _SpanSelector(registry.Selector):
        def resolve(self, model: nn.Module) -> list[Region]:
            return [Span(start=model.fc1, end=model.act)]  # type: ignore[arg-type]

    spec = AlgorithmSpec(
        fn=functools.partial(sgd_step, lr=0.1), selector=_SpanSelector(), flows=make_flows()
    )

    # WHEN we optimize with a Span region
    ff.layerwise_optimize(model, calibration, spec, sample_args=(calibration[0],))

    # THEN fc1 is optimized (it's inside the span) and fc2 is not
    assert not torch.allclose(initial_w1, model.fc1.weight.data)
    assert torch.allclose(initial_w2, model.fc2.weight.data)


def test_layerwise_optimize_entire_graph_span(two_linear: TwoLinear) -> None:
    # GIVEN a model and a Span covering the entire graph (fc1 through fc2)
    model = two_linear.eval()
    calibration = [torch.randn(2, 8) for _ in range(4)]
    initial_w1 = model.fc1.weight.data.clone()
    initial_w2 = model.fc2.weight.data.clone()

    class _FullSpanSelector(registry.Selector):
        def resolve(self, model: nn.Module) -> list[Region]:
            return [Span(start=model.fc1, end=model.fc2)]  # type: ignore[arg-type]

    spec = AlgorithmSpec(
        fn=functools.partial(sgd_step, lr=0.1), selector=_FullSpanSelector(), flows=make_flows()
    )

    # WHEN we optimize spanning the whole graph
    ff.layerwise_optimize(model, calibration, spec, sample_args=(calibration[0],))

    # THEN all weights changed
    assert not torch.allclose(initial_w1, model.fc1.weight.data)
    assert not torch.allclose(initial_w2, model.fc2.weight.data)


def test_layerwise_optimize_fn_receives_correct_params_and_batches(two_linear: TwoLinear) -> None:
    # GIVEN a model and a spy algorithm that records what it receives
    model = two_linear.eval()
    calibration = [torch.randn(2, 8) for _ in range(5)]
    received: dict[str, object] = {}

    def spy(module: nn.Module, bundle: ActivationBundle) -> None:
        received["param_ids"] = {id(p) for p in module.parameters()}
        received["batches"] = list(bundle)

    spec = AlgorithmSpec(fn=spy, selector=normalize([model.fc1]), flows=make_flows())

    # WHEN we optimize
    ff.layerwise_optimize(model, calibration, spec, sample_args=(calibration[0],))

    # THEN the fn received fc1's parameters by identity and the correct batch count
    assert id(model.fc1.weight) in received["param_ids"]  # type: ignore[operator]
    assert id(model.fc1.bias) in received["param_ids"]  # type: ignore[operator]
    assert id(model.fc2.weight) not in received["param_ids"]  # type: ignore[operator]
    assert len(received["batches"]) == len(calibration)  # type: ignore[arg-type]


def _total(activations: ActivationBundle) -> torch.Tensor:
    """The sum of all values: one tensor for the whole calibration set."""
    return torch.stack([args[0].sum() for args, _ in activations]).sum()


@pytest.mark.parametrize("chunk_size", [None, 2], ids=["whole", "chunked"])
@pytest.mark.parametrize("offload", [False, True], ids=["in-place", "offloaded"])
def test_layerwise_optimize_gives_the_algorithm_the_result_of_a_flowop(
    two_linear: TwoLinear, offload: bool, chunk_size: int | None
) -> None:
    # GIVEN an algorithm that records what it receives, and a FlowOp over the original inputs
    model = two_linear.eval()
    calibration = [torch.randn(2, 5, 8) for _ in range(3)]
    received: dict[nn.Module, torch.Tensor] = {}

    def record(module: nn.Module, total: torch.Tensor) -> None:
        received[module] = total

    spec = AlgorithmSpec(
        fn=record,
        selector=normalize([model.fc1, model.fc2]),
        flows=[FlowOp(_total, combine=operator.add, activations=InputActivations("original"))],
    )
    cpu = torch.device("cpu")
    offloading = Offload(compute=cpu, weights=cpu, activations=cpu) if offload else None

    # WHEN the model is optimized
    with torch.no_grad():
        ff.layerwise_optimize(
            model,
            calibration,
            spec,
            sample_args=(calibration[0],),
            config=ff.OrchestrationConfig(offloading=offloading, chunk_size=chunk_size),
        )

    # THEN each layer received the result of the op over all calibration batches, as a tensor
    hidden = [model.act(model.fc1(batch)) for batch in calibration]
    assert received.keys() == {model.fc1, model.fc2}
    for module, batches in ((model.fc1, calibration), (model.fc2, hidden)):
        torch.testing.assert_close(received[module], sum(batch.sum() for batch in batches))


@pytest.mark.parametrize("chunk_size", [None, 2], ids=["whole", "chunked"])
def test_layerwise_optimize_binds_dict_items_to_the_inputs(chunk_size: int | None) -> None:
    # GIVEN a model with two inputs, and calibration items that are dicts of those inputs
    model = KwargForward().eval()
    calibration = [{"x": torch.randn(2, 4), "scale": torch.randn(2, 4)} for _ in range(3)]
    received: dict[nn.Module, torch.Tensor] = {}

    def record(module: nn.Module, total: torch.Tensor) -> None:
        received[module] = total

    spec = AlgorithmSpec(
        fn=record,
        selector=normalize([model.fc]),
        flows=[FlowOp(_total, combine=operator.add, activations=InputActivations("original"))],
    )

    # WHEN the model is optimized
    with torch.no_grad():
        ff.layerwise_optimize(
            model,
            calibration,
            spec,
            sample_args=(calibration[0]["x"],),
            sample_kwargs={"scale": calibration[0]["scale"]},
            config=ff.OrchestrationConfig(chunk_size=chunk_size),
        )

    # THEN the layer read the `x` of every item
    torch.testing.assert_close(received[model.fc], sum(item["x"].sum() for item in calibration))


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_orchestration_config_rejects_a_chunk_size_below_one(chunk_size: int) -> None:
    # WHEN a config is made with a chunk size below one
    # THEN it raises a ValueError
    with pytest.raises(ValueError, match="chunk_size must be at least one"):
        ff.OrchestrationConfig(chunk_size=chunk_size)


def test_layerwise_optimize_rejects_chunking_one_tensor(two_linear: TwoLinear) -> None:
    # GIVEN one tensor as calibration data
    model = two_linear.eval()
    data = torch.randn(4, 8)
    spec = AlgorithmSpec(
        fn=lambda module, total: None,
        selector=normalize([model.fc1]),
        flows=[FlowOp(_total, combine=operator.add, activations=InputActivations("original"))],
    )

    # WHEN it is optimized in chunks
    # THEN it raises a TypeError
    with pytest.raises(TypeError, match="Cannot chunk one tensor"):
        ff.layerwise_optimize(
            model,
            data,
            spec,
            sample_args=(data,),
            config=ff.OrchestrationConfig(chunk_size=2),
        )
