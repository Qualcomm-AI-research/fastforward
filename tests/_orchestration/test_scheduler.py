# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import dataclasses
import pathlib

from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from typing import Any, ContextManager

import attrs
import pytest
import syrupy
import torch

from fastforward._orchestration.data_flow import (
    ANY,
    ORIGINAL,
    QUANTIZED,
    DataFlow,
    FlowGenerator,
    FlowOp,
    Gradient,
    InputActivations,
    ModelOutputs,
    OutputActivations,
    register_generator,
)
from fastforward._orchestration.graph_module import (
    FlowOpRef,
    GraphModule,
    InputRef,
    NodeRef,
    SubgraphSpec,
    _BaseRef,
    reduce_resolution,
)
from fastforward._orchestration.instruction_engine import (
    ActivationBundle,
    CallModule,
    CallVJP,
    DeleteRegisterEntries,
    InstructionEngine,
    InstructionProgram,
    LoadAttribute,
    ReturnOutputs,
    RunDelegate,
    StoreValue,
    _fmt_callable,
    _offloading_pass,
    lifetime_management_pass,
)
from fastforward._orchestration.location import DeviceLocation, DiskLocation
from fastforward._orchestration.scheduler import (
    OptimizeIntervention,
    Scheduler,
    bind_flows,
    schedule,
)
from fastforward._orchestration.trace import _MIN_TORCH_VERSION, trace
from packaging.version import Version

from ._models import (
    ResidualStack,
    ReturnTuple,
    TiedBranches,
    TiedHead,
    TiedMatmul,
    ToyAttention,
    TwoLayerModel,
    TwoLinear,
)

pytestmark = pytest.mark.skipif(
    Version(torch.__version__.split("+", 1)[0]) < _MIN_TORCH_VERSION,
    reason=f"requires PyTorch >= {_MIN_TORCH_VERSION}",
)


def _traced_two_linear(model: TwoLinear) -> GraphModule:
    """Trace `model` so its nodes can be referenced by `bind_flows`."""
    return trace(model.eval(), torch.randn(2, 8))


def _noop(module: torch.nn.Module, *bundles: object) -> None:
    """An intervention that changes nothing; these tests read the schedule, not weights."""


@dataclasses.dataclass
class _UnhashableSelect:
    """A callable object with the default unhashable dataclass behavior."""

    def __call__(self, output: torch.Tensor) -> torch.Tensor:
        return output


def _spec(region: torch.nn.Module, *flows: DataFlow) -> SubgraphSpec:
    """A spec that runs `_noop` on `region` and declares `flows`."""
    return SubgraphSpec(region=region, fn=_noop, flows=list(flows))


def _named_optimize(name: str) -> Callable[..., None]:
    """Build a no-op intervention that names its own region.

    `CallDelegate` prints the class of its module and the name of its function,
    never the region. Without a distinct name, the interventions of `q_proj`,
    `k_proj` and `v_proj` all print the same text, and a snapshot could not tell
    them apart.
    """

    def optimize(module: torch.nn.Module, *bundles: object) -> None:
        pass

    optimize.__name__ = f"optimize<{name}>"
    return optimize


def _linear_specs(model: torch.nn.Module, flow: DataFlow) -> list[SubgraphSpec]:
    """One spec per `Linear` leaf of `model`, each declaring `flow`."""
    return [
        SubgraphSpec(region=module, fn=_named_optimize(name), flows=[flow])
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    ]


def _fold_specs(model: TwoLayerModel, flow: DataFlow) -> list[SubgraphSpec]:
    """One spec per attention/MLP fold, a resolution coarser than the Linear leaves."""
    return [
        SubgraphSpec(region=region, fn=_named_optimize(name), flows=[flow])
        for name, region in (
            ("layer_0.attn", model.layer_0.attn),
            ("layer_0.mlp", model.layer_0.mlp),
            ("layer_1.attn", model.layer_1.attn),
        )
    ]


def _region_name(instruction: object) -> str:
    """Name the region an instruction acts on, the same for every build of a graph."""
    match instruction:
        case CallModule():
            return repr(instruction.target)
        case RunDelegate():
            return _fmt_callable(instruction.fn)
        case _:
            return type(instruction).__name__


def _calls(program: InstructionProgram, generator: FlowGenerator | None = None) -> list[CallModule]:
    """The calls of `program`, of one stream if `generator` is given."""
    return [
        instruction
        for instruction in program.instructions
        if isinstance(instruction, CallModule)
        and (generator is None or generator.context == instruction.context)
    ]


def _optimizes(program: InstructionProgram) -> list[RunDelegate]:
    """The interventions of `program`, in the order they run."""
    return [i for i in program.instructions if isinstance(i, RunDelegate)]


def _stream_sequence(program: InstructionProgram) -> list[object]:
    """The streams of `program`, in the order their first call appears."""
    sequence: list[object] = []
    for call in _calls(program):
        if call.context not in sequence:
            sequence.append(call.context)
    return sequence


def _assert_reads_produced(program: InstructionProgram) -> None:
    """Assert every NodeRef a CallDelegate reads was produced by a prior CallModule."""
    produced: set[tuple[_BaseRef, object]] = set()
    for instruction in program.instructions:
        match instruction:
            case CallModule(target=target, context=context):
                produced.add((target, context))
            case RunDelegate(bundles=bundles):
                missing = [
                    (ref, bundle.context)
                    for bundle in bundles
                    for ref in (*bundle.args, *bundle.kwargs.values())
                    if isinstance(ref, NodeRef) and (ref, bundle.context) not in produced
                ]
                assert not missing, f"refs not in register: {missing}"


def _plain_context() -> Callable[[torch.nn.Module | None], ContextManager[None]]:
    """A context factory that constrains nothing, with an identity of its own.

    The scheduler keys a stream on the factory object, so two generators that
    share one factory are one stream. Each test generator needs its own.
    """

    def context(_: torch.nn.Module | None) -> ContextManager[None]:
        return nullcontext()

    return context


# Custom generators that only these tests use. `order` is what places a stream in
# the sequence: 0 is below ORIGINAL (2), 3 is between ORIGINAL and QUANTIZED (5),
# and 99 is above ANY (10), the highest built-in.
BELOW_ORIGINAL = register_generator(FlowGenerator("test_order_0", _plain_context(), order=0))
BETWEEN_BUILTINS = register_generator(FlowGenerator("test_order_3", _plain_context(), order=3))
ABOVE_ANY = register_generator(FlowGenerator("test_order_99", _plain_context(), order=99))

# Two generators that declare one and the same order.
TIE_FIRST = register_generator(FlowGenerator("test_tie_first", _plain_context(), order=7))
TIE_SECOND = register_generator(FlowGenerator("test_tie_second", _plain_context(), order=7))


def _input_flow(generator: FlowGenerator) -> InputActivations:
    """Input activations of `generator`, cached only where the generator demands it."""
    return InputActivations(generator, cache=generator.pinned)


def test_input_activations_on_first_layer_read_the_graph_input(two_linear: TwoLinear) -> None:
    # GIVEN a traced model and the FIRST layer (fc1), which has no predecessors
    graph = _traced_two_linear(two_linear)
    region = graph.node_ref(two_linear.fc1)

    # WHEN binding an InputActivations flow on that region
    [plan] = bind_flows(graph, region, [InputActivations(ORIGINAL)])

    # THEN it reads the graph's own input, so no module has to run for it
    [read] = plan.reads
    assert isinstance(read, InputRef)
    assert read.name == "x"
    assert plan.read_kwargs == {}


def test_input_activations_reads_are_region_args(two_linear: TwoLinear) -> None:
    # GIVEN a traced model and a region with predecessors (fc2)
    graph = _traced_two_linear(two_linear)
    region = graph.node_ref(two_linear.fc2)
    node = graph.node(region)

    # WHEN generating a plan to optimize that region w.r.t. its inputs
    [plan] = bind_flows(graph, region, [InputActivations(ORIGINAL)])

    # THEN the plan gives us the args/kwargs arriving at the region
    assert plan.reads == tuple(node.args)
    assert plan.read_kwargs == node.kwargs


def test_output_activations_reads_are_region_itself(two_linear: TwoLinear) -> None:
    # GIVEN a traced model and a region (fc2)
    graph = _traced_two_linear(two_linear)
    region = graph.node_ref(two_linear.fc2)

    # WHEN generating a plan to optimize that region w.r.t. its output
    [plan] = bind_flows(graph, region, [OutputActivations(ORIGINAL)])

    # THEN the plan gives us the region's own output ref
    assert plan.reads == (region,)
    assert plan.read_kwargs == {}


def test_bind_flows_rejects_unknown_dataflow_type(two_linear: TwoLinear) -> None:
    # GIVEN a custom DataFlow subclass not handled by bind_flows
    @attrs.define(frozen=True)
    class UnknownFlow(DataFlow):
        pass

    graph = _traced_two_linear(two_linear)
    region = graph.node_ref(two_linear.fc2)

    # WHEN / THEN planning for an unknown flow type raises TypeError
    with pytest.raises(TypeError, match="unsupported DataFlow type"):
        bind_flows(graph, region, [UnknownFlow(ORIGINAL)])


def test_interventions_do_not_return_outputs(two_linear: TwoLinear) -> None:
    # GIVEN a graph scheduled with interventions vs. without
    graph = _traced_two_linear(two_linear)
    specs = [_spec(two_linear.fc2, InputActivations(ORIGINAL))]

    # THEN a plain forward pass returns outputs
    assert any(isinstance(i, ReturnOutputs) for i in schedule(graph).instructions)

    # AND an intervention schedule does not
    assert not any(isinstance(i, ReturnOutputs) for i in schedule(graph, specs=specs).instructions)


@pytest.mark.parametrize(
    "generators",
    [
        (ORIGINAL, QUANTIZED),
        (BELOW_ORIGINAL, ORIGINAL),
        (ORIGINAL, BETWEEN_BUILTINS, QUANTIZED),
        (ORIGINAL, QUANTIZED, ABOVE_ANY),
        (BELOW_ORIGINAL, BETWEEN_BUILTINS, ABOVE_ANY),
    ],
    ids=["builtins", "below_original", "between_builtins", "above_any", "custom_only"],
)
def test_streams_run_in_generator_order(
    two_linear: TwoLinear, generators: tuple[FlowGenerator, ...]
) -> None:
    # GIVEN one region that needs the same activations from several streams, with the
    # flows declared in the reverse of the order the generators ask for
    graph = _traced_two_linear(two_linear)
    specs = [_spec(two_linear.fc2, *(_input_flow(g) for g in reversed(generators)))]

    # WHEN scheduling
    program = schedule(graph, specs=specs)

    # THEN the streams come out by ascending `order`. Declaration order does not
    # place a stream, and no built-in is treated as the first or the last.
    assert _stream_sequence(program) == [generator.context for generator in generators]


def test_generators_tying_on_order_are_rejected(two_linear: TwoLinear) -> None:
    # GIVEN two distinct generators that declare the same order
    graph = _traced_two_linear(two_linear)
    specs = [_spec(two_linear.fc2, _input_flow(TIE_FIRST), _input_flow(TIE_SECOND))]

    # WHEN scheduling
    # THEN it is refused. `order` alone places a stream, so a tie has no meaning, and
    # any sequence the scheduler picked would be one neither flow asked for.
    with pytest.raises(ValueError, match="order=7"):
        schedule(graph, specs=specs)


def test_any_flow_in_spec_is_rejected(two_linear: TwoLinear) -> None:
    # ANY is reserved for plain forward passes and cannot appear in optimization specs.
    graph = _traced_two_linear(two_linear)
    specs = [_spec(two_linear.fc1, InputActivations(ANY, cache=False), InputActivations(ORIGINAL))]

    with pytest.raises(ValueError, match="ANY flow"):
        schedule(graph, specs=specs)


@pytest.mark.parametrize(
    ("generator", "cache"),
    [(ORIGINAL, True), (QUANTIZED, True), (QUANTIZED, False)],
    ids=["pinned_cached", "live_cached", "live_uncached"],
)
def test_cache_of_the_flow_reaches_every_call(
    two_linear: TwoLinear, generator: FlowGenerator, cache: bool
) -> None:
    # GIVEN a flow that declares whether its work may be reused
    graph = _traced_two_linear(two_linear)
    specs = [_spec(two_linear.fc2, InputActivations(generator, cache=cache))]

    # WHEN scheduling
    program = schedule(graph, specs=specs)

    # THEN every call of that stream carries the value the flow declared
    calls = _calls(program, generator)
    assert calls
    assert all(call.cache is cache for call in calls)


def test_placement_does_not_depend_on_cache(two_layer_model: TwoLayerModel) -> None:
    # GIVEN the same live flow declared cached and uncached. Only `CallModule.cache`
    # may differ, so the programs are compared by region instead of by repr.
    def shape(cache: bool) -> list[tuple[str, str]]:
        graph = two_layer_model.to_graph_module()
        specs = _linear_specs(two_layer_model, InputActivations(QUANTIZED, cache=cache))
        program = schedule(graph, specs=specs)
        return [
            (type(instruction).__name__, _region_name(instruction))
            for instruction in program.instructions
        ]

    # WHEN scheduling both
    cached, uncached = shape(cache=True), shape(cache=False)

    # THEN the sequence is the same for both. `cache` only says whether work that is
    # already done may be reused; the generator alone decides where a call goes.
    assert cached == uncached
    # AND both hold work, so two empty programs cannot pass the comparison.
    assert sum(1 for kind, _ in cached if kind == "RunDelegate") == 12


def test_input_and_output_same_stream_merges_into_single_pass(two_linear: TwoLinear) -> None:
    # GIVEN a region (fc2) needing both input and output under the same stream
    graph = _traced_two_linear(two_linear)
    specs = [_spec(two_linear.fc2, InputActivations(ORIGINAL), OutputActivations(ORIGINAL))]

    # WHEN scheduling
    program = schedule(graph, specs=specs)

    # THEN a single stream covers the predecessors AND the region itself
    calls = _calls(program)
    assert {call.target for call in calls} == {
        graph.node_ref(two_linear.fc1),
        graph.node_ref(two_linear.act),
        graph.node_ref(two_linear.fc2),
    }

    # AND all are under one context (no duplication)
    assert {call.context for call in calls} == {ORIGINAL.context}

    # AND the intervention fires after the full stream
    opt_index = next(i for i, x in enumerate(program.instructions) if isinstance(x, RunDelegate))
    last_call = max(i for i, x in enumerate(program.instructions) if isinstance(x, CallModule))
    assert opt_index > last_call


def test_input_and_output_on_different_streams(two_linear: TwoLinear) -> None:
    # GIVEN inputs from one stream and the output of the same region from another
    graph = _traced_two_linear(two_linear)
    specs = [
        _spec(
            two_linear.fc2,
            InputActivations(ORIGINAL),
            OutputActivations(QUANTIZED, cache=False),
        )
    ]

    # WHEN scheduling
    program = schedule(graph, specs=specs)

    # THEN the input stream stops before the region, and the output stream includes it
    fc1, fc2 = graph.node_ref(two_linear.fc1), graph.node_ref(two_linear.fc2)
    input_targets = {call.target for call in _calls(program, ORIGINAL)}
    output_targets = {call.target for call in _calls(program, QUANTIZED)}
    assert fc1 in input_targets
    assert fc2 not in input_targets
    assert fc2 in output_targets


def test_same_stream_and_revision_share_one_call(two_linear: TwoLinear) -> None:
    # GIVEN two specs on the same region under the same generator, one declared
    # cached and one uncached
    graph = _traced_two_linear(two_linear)
    specs = [
        _spec(two_linear.fc2, InputActivations(QUANTIZED, cache=True)),
        _spec(two_linear.fc2, InputActivations(QUANTIZED, cache=False)),
    ]

    # WHEN scheduling
    program = schedule(graph, specs=specs)

    # THEN one set of calls serves both specs. Only fc2 changed between them, and
    # neither spec reads fc2, so `cache` does not split the register.
    assert [call.target for call in _calls(program, QUANTIZED)] == [
        graph.node_ref(two_linear.fc1),
        graph.node_ref(two_linear.act),
    ]

    # AND each spec still gets its own intervention
    optimizes = _optimizes(program)
    assert len(optimizes) == 2
    assert all(tuple(b.context for b in opt.bundles) == (QUANTIZED.context,) for opt in optimizes)


def test_duplicate_flow_on_same_region_deduplicates(two_linear: TwoLinear) -> None:
    # GIVEN the same flow declared twice in one spec
    graph = _traced_two_linear(two_linear)
    specs = [_spec(two_linear.fc2, InputActivations(ORIGINAL), InputActivations(ORIGINAL))]

    # WHEN scheduling
    program = schedule(graph, specs=specs)

    # THEN the stream does not run a node twice
    targets = [call.target for call in _calls(program, ORIGINAL)]
    assert targets
    assert len(targets) == len(set(targets))


def test_interventions_run_in_graph_order(two_linear: TwoLinear) -> None:
    # GIVEN specs on both layers, declared in the reverse of the graph order
    graph = _traced_two_linear(two_linear)
    specs = [
        _spec(two_linear.fc2, InputActivations(ORIGINAL)),
        _spec(two_linear.fc1, OutputActivations(ORIGINAL)),
    ]

    # WHEN scheduling
    program = schedule(graph, specs=specs)

    # THEN the interventions still run in graph order: a region reads what the
    # regions before it produce, so it cannot be optimized first
    optimizes = _optimizes(program)
    assert [opt.module for opt in optimizes] == [two_linear.fc1, two_linear.fc2]


def test_pinned_flow_calls_a_node_before_the_intervention_on_it(two_linear: TwoLinear) -> None:
    # GIVEN two regions that both ask for the data from before any weight changed,
    # so fc2's flow needs `act`, whose value depends on fc1's weights
    graph = _traced_two_linear(two_linear)
    specs = [
        _spec(two_linear.fc1, InputActivations(ORIGINAL)),
        _spec(two_linear.fc2, InputActivations(ORIGINAL)),
    ]

    # WHEN scheduling
    program = schedule(graph, specs=specs)
    instructions = list(program.instructions)

    # THEN every call of the pinned stream runs before the intervention on the node it
    # produces. A later call would read changed weights and store them under the name
    # of the original data.
    optimize_at: dict[_BaseRef, int] = {
        graph.node_ref(instruction.module): i
        for i, instruction in enumerate(instructions)
        if isinstance(instruction, RunDelegate) and instruction.module is not None
    }
    pinned_calls = [
        (i, instruction)
        for i, instruction in enumerate(instructions)
        if isinstance(instruction, CallModule) and ORIGINAL.context == instruction.context
    ]
    assert pinned_calls
    for index, call in pinned_calls:
        assert index < optimize_at.get(call.target, len(instructions))

    # AND it does not have to precede EVERY intervention: nothing optimizes `act`, so
    # its call waits until fc2's flow asks for it.
    act_index = next(i for i, call in pinned_calls if call.target == graph.node_ref(two_linear.act))
    assert act_index > optimize_at[graph.node_ref(two_linear.fc1)]


@pytest.mark.parametrize("cache", [True, False], ids=["cached", "uncached"])
def test_live_flow_calls_a_node_after_the_intervention_on_it(
    two_linear: TwoLinear, cache: bool
) -> None:
    # GIVEN two regions that both ask for the data the earlier interventions leave
    # behind, so fc2's flow needs fc1's output as it is after fc1 changed
    graph = _traced_two_linear(two_linear)
    specs = [
        _spec(two_linear.fc1, InputActivations(QUANTIZED, cache=cache)),
        _spec(two_linear.fc2, InputActivations(QUANTIZED, cache=cache)),
    ]

    # WHEN scheduling
    program = schedule(graph, specs=specs)
    instructions = list(program.instructions)

    # THEN every call that produces fc1's output runs after the intervention on fc1,
    # cached or not. An earlier call would give fc2 the weights that intervention
    # already replaced.
    optimize_fc1 = next(
        i
        for i, instruction in enumerate(instructions)
        if isinstance(instruction, RunDelegate) and instruction.module is two_linear.fc1
    )
    fc1_ref = graph.node_ref(two_linear.fc1)
    fc1_calls = [
        i
        for i, instruction in enumerate(instructions)
        if isinstance(instruction, CallModule) and instruction.target == fc1_ref
    ]
    assert fc1_calls
    assert all(i > optimize_fc1 for i in fc1_calls)


def test_optimize_reads_are_in_the_register_when_it_runs(two_linear: TwoLinear) -> None:
    # GIVEN a spec that pairs fc1's inputs from before any change with fc1's output
    # from after its own change, so one flow names a later model state than the other
    graph = _traced_two_linear(two_linear)
    specs = [
        _spec(
            two_linear.fc1,
            InputActivations(ORIGINAL),
            OutputActivations(QUANTIZED, cache=False),
        )
    ]

    # WHEN scheduling
    program = schedule(graph, specs=specs)

    # THEN every ref the intervention reads was produced by a prior call
    _assert_reads_produced(program)

    # AND the program runs; reading a value the register does not hold would raise.
    InstructionEngine().run(program, torch.randn(2, 8))


def test_unproducible_pinned_demand_is_refused(two_linear: TwoLinear) -> None:
    # GIVEN a scheduler whose timeline has already passed the mutation of fc1
    graph = _traced_two_linear(two_linear)
    fc1 = graph.node_ref(two_linear.fc1)
    fc2 = graph.node_ref(two_linear.fc2)

    spec = _spec(two_linear.fc2, InputActivations(ORIGINAL))
    intervention = OptimizeIntervention(
        region=fc2,
        module=two_linear.fc2,
        spec=spec,
        flows=bind_flows(graph, fc2, spec.flows),
    )
    scheduler = Scheduler(graph)
    scheduler.timeline.advance(fc1)

    # WHEN an intervention still asks for data from before that mutation
    # THEN it is refused. The weights that made the data are gone, so any value the
    # scheduler emitted instead would silently be the wrong data.
    with pytest.raises(RuntimeError, match="Cannot produce"):
        scheduler.schedule_intervention(intervention)


@pytest.mark.parametrize(
    ("model_type", "region", "sample"),
    [
        (TiedHead, "emb", torch.tensor([[1, 2]])),
        (TiedHead, "head", torch.tensor([[1, 2]])),
        (TiedMatmul, "emb", torch.tensor([[1, 2]])),
        (TiedBranches, "left", torch.randn(2, 4)),
    ],
)
def test_region_that_shares_a_weight_outside_the_region_is_rejected(
    model_type: type[torch.nn.Module], region: str, sample: torch.Tensor
) -> None:
    # GIVEN a region whose weight another node also reads, as a module or as a
    # plain tensor
    model = model_type().eval()
    graph = trace(model, sample)

    # WHEN scheduling an intervention on the region
    # THEN it fails, because changing the weight also changes the other node
    with pytest.raises(ValueError, match="shares a weight"):
        schedule(graph, specs=[_spec(model.get_submodule(region), InputActivations(ORIGINAL))])


def test_region_that_shares_a_weight_inside_the_region_is_scheduled() -> None:
    # GIVEN a region containing both readers of a tied weight, with no sharing
    # outside the region
    region = TiedHead()
    model = torch.nn.Sequential(region, torch.nn.Linear(10, 4)).eval()
    sample = torch.tensor([[1, 2]])
    graph = trace(model, sample)
    reduced, specs = reduce_resolution(
        graph, [_spec(region, InputActivations(ORIGINAL), OutputActivations(ORIGINAL))]
    )

    # WHEN scheduling an intervention on the whole region
    program = schedule(reduced, specs=specs)

    # THEN the intervention is scheduled and the program runs, including the
    # region's forward pass to produce its requested output activations
    [optimize] = _optimizes(program)
    assert optimize.module is region
    InstructionEngine().run(program, sample)


def test_region_without_shared_weights_in_a_tied_model_is_scheduled() -> None:
    # GIVEN a model with tied weights, and a region that does not read them
    model = TiedHead().eval()
    graph = trace(model, torch.tensor([[1, 2]]))

    # WHEN scheduling an intervention on the region
    program = schedule(graph, specs=[_spec(model.fc, InputActivations(ORIGINAL))])

    # THEN the program runs
    InstructionEngine().run(program, torch.tensor([[1, 2]]))


def test_gradient_flow_on_a_region_that_shares_a_weight_is_rejected() -> None:
    # GIVEN a gradient flow on a region whose weight another node also reads
    model = TiedHead().eval()
    graph = trace(model, torch.tensor([[1, 2]]))
    gradient = Gradient(_square_loss, outputs=ModelOutputs(ORIGINAL))

    # WHEN scheduling, THEN the shared weight is still found. The gradient step
    # itself is no region, so the check must skip it instead of failing on it.
    with pytest.raises(ValueError, match="shares a weight"):
        schedule(graph, specs=[_spec(model.emb, gradient)])


def test_gradient_flow_on_a_region_without_shared_weights_is_scheduled() -> None:
    # GIVEN a gradient flow on a region that does not read the tied weights
    model = TiedHead().eval()
    sample = torch.tensor([[1, 2]])
    graph = trace(model, sample)
    gradient = Gradient(_square_loss, outputs=ModelOutputs(ORIGINAL))

    # WHEN scheduling
    program = schedule(graph, specs=[_spec(model.fc, gradient)])

    # THEN the program runs
    InstructionEngine().run(program, sample)


def test_gradient_flow_with_unhashable_select_is_scheduled() -> None:
    # GIVEN a gradient whose model output selector is an unhashable callable object
    model = TiedHead().eval()
    sample = torch.tensor([[1, 2]])
    graph = trace(model, sample)
    gradient = Gradient(
        _square_loss,
        outputs=ModelOutputs(ORIGINAL, select=_UnhashableSelect()),
    )

    # WHEN scheduling
    program = schedule(graph, specs=[_spec(model.fc, gradient)])

    # THEN the gradient reference can be used as a scheduler register key
    InstructionEngine().run(program, sample)


def test_output_activations_delivers_region_output(two_linear: TwoLinear) -> None:
    # GIVEN a region optimized against its own inputs (x) and its own output (Wx),
    # both from before any weight changed
    model = two_linear.eval()
    with torch.no_grad():
        model.fc1.weight.copy_(torch.eye(8) * 2.0)
        model.fc2.weight.zero_()
        model.fc2.weight[:, :].copy_(torch.ones(4, 8))
        model.fc2.bias.zero_()

    graph = _traced_two_linear(model)

    x = torch.ones(1, 8)
    region_input = model.act(model.fc1(x)).detach().clone()
    region_output = model.fc2(region_input).detach().clone()

    received: list[torch.Tensor] = []

    def capture(_module: torch.nn.Module, *bundles: ActivationBundle) -> None:
        for bundle in bundles:
            for args, _ in bundle:
                received.append(args[0].detach().clone())
                break

    specs = [
        SubgraphSpec(
            region=model.fc2,
            fn=capture,
            flows=[InputActivations(ORIGINAL), OutputActivations(ORIGINAL)],
        )
    ]

    # WHEN the scheduled program runs
    program = schedule(graph, specs=specs)
    InstructionEngine().run(program, x)

    # THEN the intervention gets one bundle per declared flow, in declaration order:
    # the region's inputs first, its own output second. Two input bundles would
    # silently give the intervention x where it asked for Wx.
    assert len(received) == 2
    assert torch.allclose(received[0], region_input)
    assert torch.allclose(received[1], region_output)


def test_two_streams_deliver_the_same_reads_twice(two_linear: TwoLinear) -> None:
    # GIVEN a region (fc2) needing the same input activations from two streams
    graph = _traced_two_linear(two_linear)
    specs = [
        _spec(
            two_linear.fc2,
            InputActivations(ORIGINAL),
            InputActivations(QUANTIZED, cache=False),
        )
    ]

    # WHEN scheduling
    program = schedule(graph, specs=specs)

    # THEN both streams run the predecessors of the region, each in its own context
    predecessors = {graph.node_ref(two_linear.fc1), graph.node_ref(two_linear.act)}
    assert {call.target for call in _calls(program, ORIGINAL)} == predecessors
    assert {call.target for call in _calls(program, QUANTIZED)} == predecessors

    # AND the intervention receives a bundle from each of them
    [optimize] = _optimizes(program)
    assert {bundle.context for bundle in optimize.bundles} == {ORIGINAL.context, QUANTIZED.context}


def test_single_use_context_entered_once_per_call(two_linear: TwoLinear) -> None:
    # GIVEN a generator whose context is a generator-based (single-use) manager
    entries: list[int] = []

    @contextmanager
    def single_use(_: torch.nn.Module | None) -> Iterator[None]:
        entries.append(1)
        yield

    generator = register_generator(FlowGenerator("test_single_use", single_use, order=6))

    graph = _traced_two_linear(two_linear)
    specs = [_spec(two_linear.fc2, InputActivations(generator, cache=False))]
    program = schedule(graph, specs=specs)

    # WHEN running a stream that contains more than one call
    calls = _calls(program, generator)
    assert len(calls) > 1
    InstructionEngine().run(program, torch.randn(2, 8))

    # THEN the context was rebuilt per call instead of one instance being re-entered
    assert len(entries) == len(calls)


def test_returned_attribute_ref_is_extracted_once() -> None:
    # GIVEN a graph where one AttributeRef is both a node input and a graph output
    graph = GraphModule()
    inputs = graph.add_input("x")
    tuple_node = graph.add_node("tuple_node", ReturnTuple(), [inputs])
    identity = graph.add_node("identity", torch.nn.Identity(), [tuple_node[0]])
    graph.add_output(tuple_node[0], identity)

    # WHEN scheduling a plain forward pass
    program = schedule(graph)

    # THEN the attribute is extracted one time: the node and the output share the slot
    loads = [i for i in program.instructions if isinstance(i, LoadAttribute)]
    assert len(loads) == 1
    assert loads[0].target == tuple_node[0]

    # AND the program returns both outputs, the shared slot among them
    assert program.instructions[-1] == ReturnOutputs(outputs=[tuple_node[0], identity])


def test_flow_reading_an_attribute_of_a_node_schedules_the_node() -> None:
    # GIVEN a graph whose node returns a tuple, and a region that reads one element
    graph = GraphModule()
    inputs = graph.add_input("x")
    tuple_node = graph.add_node("tuple_node", ReturnTuple(), [inputs])
    consumer = torch.nn.Identity()
    identity = graph.add_node("identity", consumer, [tuple_node[0]])
    graph.add_output(identity)

    # WHEN a flow reads that attribute as the input of the region it optimizes
    program = schedule(graph, specs=[_spec(consumer, InputActivations(ORIGINAL))])

    # THEN the node behind the attribute is called first: an attribute is a view on
    # a value, so the value has to be in the register before `LoadAttribute` takes it.
    kinds = [type(instruction).__name__ for instruction in program.instructions]
    assert kinds.index("CallModule") < kinds.index("LoadAttribute")
    assert [call.target for call in _calls(program)] == [tuple_node]
    InstructionEngine().run(program, torch.randn(2, 8))


def _square_loss(outputs: ActivationBundle) -> Iterator[torch.Tensor]:
    """Yield a different nonlinear loss of the model outputs for each batch."""
    for idx, ((output,), _) in enumerate(outputs):
        yield _square_reference(idx, output)


def _square_reference(idx: int, output: torch.Tensor) -> torch.Tensor:
    return ((idx + 1) * output.square() + output).sum()


def _autograd(
    model: torch.nn.Module,
    regions: list[torch.nn.Module],
    batches: list[torch.Tensor],
    loss: Callable[[int, torch.Tensor], torch.Tensor] = _square_reference,
) -> dict[torch.nn.Module, list[torch.Tensor]]:
    """The gradient of `loss` at the output of each region, by full-model autograd."""
    grads: dict[torch.nn.Module, list[torch.Tensor]] = {region: [] for region in regions}
    for idx, batch in enumerate(batches):
        outputs: dict[torch.nn.Module, torch.Tensor] = {}
        hooks = [
            region.register_forward_hook(lambda m, _, out: outputs.__setitem__(m, out))
            for region in regions
        ]
        with torch.enable_grad():
            values = torch.autograd.grad(loss(idx, model(batch)), [outputs[r] for r in regions])
        for hook in hooks:
            hook.remove()
        for region, value in zip(regions, values):
            grads[region].append(value)
    return grads


@attrs.define
class _Recorder:
    """An intervention that records the gradients it reads, then changes its weights."""

    reads: dict[torch.nn.Module, list[Any]] = attrs.field(factory=dict)

    def __call__(self, module: torch.nn.Module, gradients: ActivationBundle) -> None:
        self.reads[module] = [args[0] for args, _ in gradients]
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.add_(1.0)


def _run(program: InstructionProgram, *args: Any) -> None:
    instructions = lifetime_management_pass(program.instructions)
    InstructionEngine().run(dataclasses.replace(program, instructions=instructions), *args)


def _assert_close(left: list[torch.Tensor], right: list[torch.Tensor]) -> None:
    assert len(left) == len(right)
    for one, two in zip(left, right):
        torch.testing.assert_close(one, two)


@pytest.mark.parametrize(
    ("make", "regions", "width"),
    [
        (TwoLinear, ["fc1", "fc2"], 8),
        (ResidualStack, ["layers.0.attn.out_proj", "layers.1.mlp.up", "layers.2.attn.q_proj"], 8),
        pytest.param(
            TwoLayerModel,
            ["layer_0.attn.v_proj", "layer_0.mlp.down", "layer_1.attn.q_proj", "layer_1.mlp.up"],
            8,
            marks=pytest.mark.slow,
        ),
    ],
    ids=["chain", "residual", "decoder"],
)
def test_gradients_agree_with_autograd_after_interventions(
    make: Callable[[], torch.nn.Module], regions: list[str], width: int
) -> None:
    # GIVEN regions that read the gradients at their outputs, and change their weights.
    # In the residual model, an output reaches the loss through more than one consumer.
    model = make().eval()
    modules = [model.get_submodule(name) for name in regions]
    batches = [torch.randn(2, 3, width) for _ in range(2)]
    expected = _autograd(model, modules, batches)
    recorder = _Recorder()
    flow = Gradient(_square_loss, outputs=ModelOutputs(ORIGINAL))
    specs = [SubgraphSpec(region=module, fn=recorder, flows=[flow]) for module in modules]

    # WHEN scheduling and running
    _run(schedule(trace(model, batches[0]), specs), batches)

    # THEN every region reads the gradients of the original model, also after an
    # earlier region changed a weight
    for module in modules:
        _assert_close(recorder.reads[module], expected[module])


@pytest.mark.parametrize("storage", ["cpu", "disk"])
def test_gradients_with_offloading(storage: str, tmp_path: pathlib.Path) -> None:
    # GIVEN a residual model, and a storage location for weights and activations
    model = ResidualStack(depth=2).eval()
    region = model.get_submodule("layers.0.mlp.up")
    batches = [torch.randn(2, 3, 8) for _ in range(2)]
    expected = _autograd(model, [region], batches)
    recorder = _Recorder()
    flow = Gradient(_square_loss, outputs=ModelOutputs(ORIGINAL))
    graph = trace(model, batches[0])
    location = DiskLocation(tmp_path) if storage == "disk" else DeviceLocation(torch.device("cpu"))
    program = schedule(graph, [SubgraphSpec(region=region, fn=recorder, flows=[flow])])

    # WHEN running with offloading
    compute = DeviceLocation(torch.device("cpu"))
    instructions = _offloading_pass(
        program.instructions, compute, location, lambda _: location, graph
    )
    _run(dataclasses.replace(program, instructions=instructions), batches)

    # THEN the region reads the gradients of the original model
    _assert_close(recorder.reads[region], expected[region])


class _WithAuxiliaryOutput(torch.nn.Module):
    """Returns the output of `model` next to a second tensor that the loss does not read."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: (model(x), sum of model(x))."""
        out = self.model(x)
        return out, out.sum()


def test_gradients_with_a_select_agree_with_autograd_of_the_selected_output() -> None:
    # GIVEN a model with two outputs, and a gradient that reads only the first
    inner = TwoLinear().eval()
    modules: list[torch.nn.Module] = [inner.fc1, inner.fc2]
    batches = [torch.randn(2, 3, 8) for _ in range(2)]
    expected = _autograd(inner, modules, batches)
    recorder = _Recorder()
    flow = Gradient(_square_loss, outputs=ModelOutputs(ORIGINAL, select=lambda output: output[0]))
    specs = [SubgraphSpec(region=module, fn=recorder, flows=[flow]) for module in modules]

    # WHEN scheduling and running
    _run(schedule(trace(_WithAuxiliaryOutput(inner).eval(), batches[0]), specs), batches)

    # THEN every region reads the gradients of the loss of the selected output
    for module in modules:
        _assert_close(recorder.reads[module], expected[module])


class _LastOfTuple(torch.nn.Module):
    """`fc2` reads the last item of a tuple, so the gradient reaches `fc1` through `[-1]`."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(8, 8)
        self.pair = ReturnTuple()
        self.fc2 = torch.nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: fc2(pair(fc1(x))[-1])."""
        return self.fc2(self.pair(self.fc1(x))[-1])  # type: ignore[no-any-return]


def test_gradients_through_a_negative_index() -> None:
    # GIVEN a graph where a node reads the last item of a tuple, and a region before the tuple
    model = _LastOfTuple().eval()
    graph = GraphModule()
    fc1 = graph.add_node("fc1", model.fc1, [graph.add_input("x")])
    pair = graph.add_node("pair", model.pair, [fc1])
    graph.add_output(graph.add_node("fc2", model.fc2, [pair[-1]]))
    batches = [torch.randn(2, 3, 8) for _ in range(2)]
    expected = _autograd(model, [model.fc1], batches)
    recorder = _Recorder()
    flow = Gradient(_square_loss, outputs=ModelOutputs(ORIGINAL))

    # WHEN scheduling and running
    _run(schedule(graph, [SubgraphSpec(region=model.fc1, fn=recorder, flows=[flow])]), batches)

    # THEN the region reads the gradients of the original model
    _assert_close(recorder.reads[model.fc1], expected[model.fc1])


def test_equal_gradients_call_the_objective_once(two_linear: TwoLinear) -> None:
    # GIVEN two equal gradients with a label value on two regions, and a gradient
    # of another objective that the second region also reads
    model = two_linear.eval()
    batches = [torch.randn(2, 8) for _ in range(2)]
    calls: list[tuple[str, object]] = []
    labels = object()

    def square(outputs: ActivationBundle, labels: object) -> Iterator[torch.Tensor]:
        calls.append(("square", labels))
        yield from _square_loss(outputs)

    def total(outputs: ActivationBundle) -> Iterator[torch.Tensor]:
        calls.append(("total", None))
        for (output,), _ in outputs:
            yield output.sum()

    shared = Gradient(square, outputs=ModelOutputs(ORIGINAL), labels=labels)
    equal = Gradient(square, outputs=ModelOutputs(ORIGINAL), labels=labels)
    other = Gradient(total, outputs=ModelOutputs(ORIGINAL))
    reads: dict[torch.nn.Module, list[list[torch.Tensor]]] = {}

    def record(module: torch.nn.Module, *gradients: ActivationBundle) -> None:
        reads[module] = [[args[0] for args, _ in flow] for flow in gradients]

    specs = [
        SubgraphSpec(region=model.fc1, fn=record, flows=[shared]),
        SubgraphSpec(region=model.fc2, fn=record, flows=[equal, other]),
    ]

    # WHEN scheduling and running
    _run(schedule(trace(model, batches[0]), specs), batches)

    # THEN the equal gradients share one call
    assert sorted(name for name, _ in calls) == ["square", "total"]

    # THEN the label value reaches the objective unchanged
    assert ("square", labels) in calls

    # THEN each gradient is the gradient of its own objective
    expected = _autograd(model, [model.fc1, model.fc2], batches)
    totals = _autograd(model, [model.fc2], batches, lambda _, output: output.sum())
    _assert_close(reads[model.fc1][0], expected[model.fc1])
    _assert_close(reads[model.fc2][0], expected[model.fc2])
    _assert_close(reads[model.fc2][1], totals[model.fc2])


def test_only_gradient_flows_require_eval_mode(two_linear: TwoLinear) -> None:
    # GIVEN a model in training mode
    graph = _traced_two_linear(two_linear)
    two_linear.train()
    gradient = Gradient(_square_loss, outputs=ModelOutputs(ORIGINAL))

    # WHEN scheduling activation flows, THEN it succeeds
    schedule(graph, [_spec(two_linear.fc1, InputActivations(ORIGINAL))])

    # WHEN scheduling a gradient flow, THEN it fails, because a VJP would not see the
    # same forward pass
    with pytest.raises(ValueError, match="eval mode"):
        schedule(graph, [_spec(two_linear.fc1, gradient)])


def _register_size(program: InstructionProgram) -> int:
    """The largest number of refs the register holds at one time.

    Runs the lifetime pass, then counts stores against frees. `DeleteRegisterEntries`
    frees a ref under every context, so a ref is the unit here.
    """
    instructions = lifetime_management_pass(program.instructions)
    live: set[_BaseRef] = set()
    peak = 0
    for instruction in instructions:
        match instruction:
            case CallModule(target=target) | StoreValue(target=target):
                live.add(target)
            case DeleteRegisterEntries(targets=targets):
                live.difference_update(targets)
        peak = max(peak, len(live))
    return peak


@pytest.mark.parametrize("generator", [ORIGINAL, QUANTIZED], ids=["pinned", "live"])
def test_residual_stack_schedules_one_call_per_region(generator: FlowGenerator) -> None:
    # GIVEN a stack of residual layers. A residual makes a node read both its
    # predecessor and the value before it, so the number of paths from the input to
    # the output doubles per layer while the number of nodes only grows by a constant.
    depth = 6
    model = ResidualStack(depth).eval()
    graph = model.to_graph_module()
    specs = _linear_specs(model, InputActivations(generator))

    # WHEN scheduling
    program = schedule(graph, specs=specs)

    # THEN each intervention appears once, and the number of calls stays proportional
    # to the graph. Work that grows with the number of paths, instead of with the
    # number of nodes, makes a real model of 32 layers impossible to schedule.
    assert len(_optimizes(program)) == len(specs)
    assert len(_calls(program)) <= 2 * len(graph.topo_order)


def test_residual_stack_register_stays_bounded() -> None:
    # GIVEN two residual stacks of different depth, each Linear optimized against
    # the inputs it had before any weights changed
    programs = []
    for depth in (3, 9):
        model = ResidualStack(depth).eval()
        graph = model.to_graph_module()
        programs.append(schedule(graph, specs=_linear_specs(model, InputActivations(ORIGINAL))))

    # THEN the register holds the same number of entries in both. A pinned value is
    # taken as late as the mutation allows, so the schedule carries the frontier of
    # the graph and not its history, however deep the stack is.
    shallow, deep = (_register_size(program) for program in programs)
    assert shallow == deep


def test_snapshot_input_activations_original_and_quantized(
    two_linear: TwoLinear, snapshot: syrupy.assertion.SnapshotAssertion
) -> None:
    # GIVEN one region that needs the same inputs from a pinned and a live stream
    graph = _traced_two_linear(two_linear)
    specs = [
        _spec(
            two_linear.fc2,
            InputActivations(ORIGINAL),
            InputActivations(QUANTIZED, cache=False),
        )
    ]

    # WHEN scheduling
    program = schedule(graph, specs=specs)

    # THEN the snapshot holds the whole sequence: the pinned stream first and cached,
    # the live stream after it and uncached, then one intervention reading both.
    assert snapshot == "\n".join(repr(instruction) for instruction in program.instructions)


def test_snapshot_input_activations_original_on_every_linear(
    two_layer_model: TwoLayerModel, snapshot: syrupy.assertion.SnapshotAssertion
) -> None:
    # GIVEN a two-layer transformer with every Linear optimized against the inputs
    # it had before any weights changed
    graph = two_layer_model.to_graph_module()
    specs = _linear_specs(two_layer_model, InputActivations(ORIGINAL))

    # WHEN scheduling
    program = schedule(graph, specs=specs)

    # THEN each Linear runs right before its own intervention, and one time only.
    # The snapshot shows the rest: `sdpa` and `act` run for a region downstream but
    # nothing optimizes them, the three projections share one register entry, and
    # `layer_1.mlp.down` is optimized but never called because no flow needs the
    # output of the last node.
    assert snapshot == "\n".join(repr(instruction) for instruction in program.instructions)


def test_snapshot_input_activations_quantized_on_every_linear(
    two_layer_model: TwoLayerModel, snapshot: syrupy.assertion.SnapshotAssertion
) -> None:
    # GIVEN a two-layer transformer with every Linear optimized against the inputs
    # that the already-optimized Linears produce
    graph = two_layer_model.to_graph_module()
    specs = _linear_specs(two_layer_model, InputActivations(QUANTIZED))

    # WHEN scheduling
    program = schedule(graph, specs=specs)

    # THEN each Linear runs after its own intervention, so a region calibrates
    # against what the regions before it now produce. The three projections that
    # share an input make this visible: their interventions come together, before
    # any of the three calls.
    assert snapshot == "\n".join(repr(instruction) for instruction in program.instructions)


def test_snapshot_input_activations_original_on_reduced_folds(
    two_layer_model: TwoLayerModel, snapshot: syrupy.assertion.SnapshotAssertion
) -> None:
    # GIVEN specs on the attention and MLP folds instead of on the Linear leaves. A
    # fold is not a node of the multi-resolution graph, so `reduce_resolution` first
    # rebuilds the graph at the resolution the specs ask for, and anchors the specs
    # on it. This is the path `layerwise_optimize` takes.
    graph = two_layer_model.to_graph_module()
    reduced, specs = reduce_resolution(
        graph, _fold_specs(two_layer_model, InputActivations(ORIGINAL))
    )

    # WHEN scheduling the reduced graph
    program = schedule(reduced, specs=specs)

    # THEN there is one call per fold, right before its own intervention. No leaf
    # appears; the whole attention is a single `SmallAttn` call. `layer_1.mlp` is
    # absent, because no flow needs the output of the last fold.
    assert snapshot == "\n".join(repr(instruction) for instruction in program.instructions)


def test_snapshot_input_activations_quantized_on_reduced_folds(
    two_layer_model: TwoLayerModel, snapshot: syrupy.assertion.SnapshotAssertion
) -> None:
    # GIVEN specs on the attention and MLP folds, each optimized against the inputs
    # that the already-optimized folds produce
    graph = two_layer_model.to_graph_module()
    reduced, specs = reduce_resolution(
        graph, _fold_specs(two_layer_model, InputActivations(QUANTIZED))
    )

    # WHEN scheduling the reduced graph
    program = schedule(reduced, specs=specs)

    # THEN each fold runs after its own intervention, exactly as a Linear leaf would.
    # The resolution of the graph decides what a call covers, not where it goes.
    assert snapshot == "\n".join(repr(instruction) for instruction in program.instructions)


def _delegate_kinds_of(instruction: RunDelegate) -> str:
    """Tell the call of a `FlowOp` from an intervention or an objective."""
    if any(isinstance(ref, FlowOpRef) for ref, _ in instruction.targets):
        return "op"
    return "objective" if instruction.targets else "consumer"


def _delegate_kinds(program: InstructionProgram) -> list[str]:
    """Tell the calls of `FlowOp`s from the interventions, in the order they run."""
    return [
        _delegate_kinds_of(instruction)
        for instruction in program.instructions
        if isinstance(instruction, RunDelegate)
    ]


def _mean_square(x: ActivationBundle) -> torch.Tensor:
    """The mean over the batches of the mean square of the first arg."""
    return torch.stack([args[0].square().mean() for args, _ in x]).mean()


def _expected_mean_square(inputs: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([x.square().mean() for x in inputs]).mean()


def _record_and_mutate(
    reads: dict[torch.nn.Module, list[Any]],
) -> Callable[..., None]:
    """An intervention that records what it reads, then changes weights.

    It records the first arg of every batch of a flow, and the value of an op.
    """

    def record(module: torch.nn.Module, *inputs: Any) -> None:
        reads[module] = [
            [args[0] for args, _ in data] if isinstance(data, ActivationBundle) else data
            for data in inputs
        ]
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.add_(1.0)

    return record


@pytest.mark.parametrize("generator", [ORIGINAL, QUANTIZED], ids=["pinned", "live"])
def test_flow_op_schedules_just_before_its_first_consumer(
    two_linear: TwoLinear, generator: FlowGenerator
) -> None:
    # GIVEN an op over the inputs of two regions in a chain
    model = two_linear.eval()
    batches = [torch.randn(2, 8) for _ in range(3)]
    reads: dict[torch.nn.Module, list[Any]] = {}
    op = FlowOp(_mean_square, x=InputActivations(generator))
    consumer = _record_and_mutate(reads)
    specs = [SubgraphSpec(region=r, fn=consumer, flows=[op]) for r in (model.fc1, model.fc2)]

    # WHEN scheduling
    program = schedule(trace(model, batches[0]), specs)

    # THEN each op runs right before its own consumer
    assert _delegate_kinds(program) == ["op", "consumer", "op", "consumer"]


def test_regions_that_read_equal_data_share_one_flow_op_call() -> None:
    # GIVEN an attention block
    model = ToyAttention(8).eval()
    batches = [torch.randn(2, 3, 8) for _ in range(2)]
    projections = [model.q_proj, model.k_proj, model.v_proj]
    calls: list[int] = []

    def counted_mean_square(x: ActivationBundle) -> torch.Tensor:
        calls.append(1)
        return _mean_square(x)

    # GIVEN a FlowOp on all projections QKV and O
    reads: dict[torch.nn.Module, list[Any]] = {}
    op = FlowOp(counted_mean_square, x=InputActivations(ORIGINAL))
    regions = [*projections, model.o_proj]
    specs = [SubgraphSpec(region=r, fn=_record_and_mutate(reads), flows=[op]) for r in regions]

    # WHEN scheduling
    program = schedule(trace(model, batches[0]), specs)

    # THEN the QKV inputs share the same FlowOp
    assert _delegate_kinds(program) == ["op", "consumer", "consumer", "consumer", "op", "consumer"]

    # WHEN running
    _run(program, batches)

    # THEN the FlowOp function runs only twice, QKV all share the same input
    assert len(calls) == 2
    for region in projections:
        torch.testing.assert_close(reads[region][0], _expected_mean_square(batches))


def test_regions_share_a_flow_op_call_only_if_their_settings_are_equal() -> None:
    # GIVEN an attention block
    model = ToyAttention(8).eval()
    batches = [torch.randn(2, 3, 8) for _ in range(2)]

    # GIVEN the user makes a distinction between QK and V projections
    tags: dict[torch.nn.Module, str] = {model.q_proj: "a", model.k_proj: "a", model.v_proj: "b"}
    calls: list[str] = []

    def tag_of_call(_x: ActivationBundle, tag: str) -> str:
        calls.append(tag)
        return tag

    # GIVEN The user provides these distinctions to `from_module`
    reads: dict[torch.nn.Module, list[Any]] = {}
    op = FlowOp(
        tag_of_call,
        from_module=lambda module: {"tag": tags[module]},
        _x=InputActivations(ORIGINAL),
    )
    specs = [SubgraphSpec(region=r, fn=_record_and_mutate(reads), flows=[op]) for r in tags]

    # WHEN scheduling
    program = schedule(trace(model, batches[0]), specs)

    # THEN QK shares a call and V has a call of its own
    assert _delegate_kinds(program) == ["op", "consumer", "consumer", "op", "consumer"]

    # WHEN running
    _run(program, batches)

    # THEN the FlowOp runs twice, once for QK and once for V
    assert calls == ["a", "b"]
    assert [reads[region][0] for region in tags] == ["a", "a", "b"]


def _sum_of_gradients(g: ActivationBundle) -> torch.Tensor:
    """The sum over the batches of the first arg."""
    return torch.stack([args[0].sum() for args, _ in g]).sum()


def test_flow_op_on_a_gradient_runs_once_its_gradient_is_made(two_linear: TwoLinear) -> None:
    # GIVEN an op over the gradient at the output of each region in a chain
    model = two_linear.eval()
    batches = [torch.randn(2, 8) for _ in range(3)]
    reads: dict[torch.nn.Module, list[Any]] = {}
    op = FlowOp(_sum_of_gradients, g=Gradient(_square_loss, outputs=ModelOutputs(ORIGINAL)))
    regions: list[torch.nn.Module] = [model.fc1, model.fc2]
    specs = [SubgraphSpec(region=r, fn=_record_and_mutate(reads), flows=[op]) for r in regions]
    expected = _autograd(model, regions, batches)

    # WHEN scheduling
    program = schedule(trace(model, batches[0]), specs)

    # THEN the op of fc2 runs before the backward pass goes on through fc2, and the op of
    # fc1 runs after it
    steps = [
        f"vjp {instruction.caller is model.fc2}"
        if isinstance(instruction, CallVJP)
        else _delegate_kinds_of(instruction)
        for instruction in program.instructions
        if isinstance(instruction, CallVJP | RunDelegate)
    ]
    first_op = steps.index("op")
    assert first_op < steps.index("vjp True") < steps.index("op", first_op + 1)
    assert steps[-2:] == ["consumer", "consumer"]

    # WHEN running
    _run(program, batches)

    # THEN each region reads the sum of the gradients of the original model
    for region in regions:
        torch.testing.assert_close(reads[region][0], torch.stack(expected[region]).sum())


def test_nested_flow_ops(two_linear: TwoLinear) -> None:
    # GIVEN a two linear layer model
    model = two_linear.eval()
    batches = [torch.randn(2, 8) for _ in range(3)]
    seen: list[Any] = []

    def doubled(mean_square: torch.Tensor) -> torch.Tensor:
        seen.append(mean_square)
        return 2 * mean_square

    # GIVEN a nested flow op, the first calculating the mean square, the second doubling it
    reads: dict[torch.nn.Module, list[Any]] = {}
    op = FlowOp(doubled, mean_square=FlowOp(_mean_square, x=InputActivations(ORIGINAL)))
    specs = [SubgraphSpec(region=model.fc1, fn=_record_and_mutate(reads), flows=[op])]

    # WHEN scheduling and running
    _run(schedule(trace(model, batches[0]), specs), batches)

    expected = _expected_mean_square(batches)
    # THEN the inner function gets the mean square
    torch.testing.assert_close(seen[0], expected)
    # THEN the outer function gets the doubled mean square
    torch.testing.assert_close(reads[model.fc1][0], 2 * expected)
