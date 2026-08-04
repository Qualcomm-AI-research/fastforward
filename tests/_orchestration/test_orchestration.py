# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

# pylint: disable=missing-function-docstring
import functools

import fastforward as ff
import pytest
import torch

from fastforward._orchestration import registry
from fastforward._orchestration.instruction_engine import ActivationBundle, OffloadEverything
from fastforward._orchestration.registry import AlgorithmSpec, normalize
from fastforward._orchestration.trace import _MIN_TORCH_VERSION, trace
from packaging.version import Version
from torch import nn

from ._models import TwoLinear
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
    ff.layerwise_optimize(model, calibration, spec)

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
        offloading=OffloadEverything(compute_device=cpu, storage_device=cpu),
    )

    # THEN only the targeted layer changed
    assert not torch.allclose(initial_w1, model.fc1.weight.data)
    assert torch.allclose(initial_w2, model.fc2.weight.data)


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
    ff.layerwise_optimize(model, calibration, spec)

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
        ff.layerwise_optimize(model, calibration, algorithm, targets=[model.fc1])

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
    ff.layerwise_optimize(model, calibration, spec)

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
    ff.layerwise_optimize(model, calibration, [spec_fc1, spec_fc2])

    # THEN both modules were optimized
    assert not torch.allclose(initial_w1, model.fc1.weight.data)
    assert not torch.allclose(initial_w2, model.fc2.weight.data)
