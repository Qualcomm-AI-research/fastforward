# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

# pylint: disable=missing-function-docstring
import copy
import dataclasses
import functools

from pathlib import Path

import fastforward as ff
import pytest
import torch

from fastforward._orchestration import registry
from fastforward._orchestration.graph_module import Region, Span
from fastforward._orchestration.instruction_engine import ActivationBundle
from fastforward._orchestration.location import DiskLocation, LocationLike
from fastforward._orchestration.registry import AlgorithmSpec, normalize
from fastforward._orchestration.trace import _MIN_TORCH_VERSION, trace
from fastforward.orchestration import Offload
from packaging.version import Version
from torch import nn
from typing_extensions import override

from ._models import KwargForward, TwoLinear
from .conftest import make_flows, sgd_step

pytestmark = pytest.mark.skipif(
    Version(torch.__version__.split("+", 1)[0]) < _MIN_TORCH_VERSION,
    reason=f"requires PyTorch >= {_MIN_TORCH_VERSION}",
)


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
        offloading=Offload(compute=cpu, weights=cpu),
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
            offloading=Offload(compute=cpu, weights=cpu, activations=activation_location),
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
            offloading=Offload(compute=cpu, weights=weight_location),
        )

    # WHEN we optimize both
    run(in_memory_model, cpu)
    run(on_disk_model, tmp_path)

    # THEN resting weights in files leaves the model weights unchanged
    assert torch.equal(on_disk_model.fc1.weight.data, in_memory_model.fc1.weight.data)

    # THEN the weights read from the files when the run ends, which holds the memory that
    # the caller asked to save
    assert on_disk_model.fc1.weight.is_shared()
    assert list(tmp_path.iterdir()) != []


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


def test_layerwise_optimize_overlapping_specs_raises(two_linear: TwoLinear) -> None:
    # GIVEN a model and two specs that both target fc1
    model = two_linear.eval()
    calibration = [torch.randn(2, 8) for _ in range(2)]

    spec1 = AlgorithmSpec(fn=sgd_step, selector=normalize([model.fc1]), flows=make_flows())
    spec2 = AlgorithmSpec(fn=sgd_step, selector=normalize([model.fc1]), flows=make_flows())

    # WHEN we pass overlapping specs
    # THEN a ValueError is raised
    with pytest.raises(ValueError, match="Overlapping nodes"):
        ff.layerwise_optimize(model, calibration, [spec1, spec2], sample_args=(calibration[0],))


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
