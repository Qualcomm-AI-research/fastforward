# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from typing import ContextManager

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
    InputActivations,
    OutputActivations,
    register_generator,
)
from fastforward._orchestration.graph_module import (
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
    DeleteRegisterEntries,
    InstructionEngine,
    InstructionProgram,
    LoadAttribute,
    OptimizeModule,
    ReturnOutputs,
    StoreValue,
    lifetime_management_pass,
)
from fastforward._orchestration.scheduler import (
    OptimizeIntervention,
    Scheduler,
    bind_flows,
    schedule,
)
from fastforward._orchestration.trace import _MIN_TORCH_VERSION, trace
from packaging.version import Version

from ._models import ResidualStack, ReturnTuple, TwoLayerModel, TwoLinear

pytestmark = pytest.mark.skipif(
    Version(torch.__version__.split("+", 1)[0]) < _MIN_TORCH_VERSION,
    reason=f"requires PyTorch >= {_MIN_TORCH_VERSION}",
)


def _traced_two_linear(model: TwoLinear) -> GraphModule:
    """Trace `model` so its nodes can be referenced by `bind_flows`."""
    return trace(model.eval(), torch.randn(2, 8))


def _noop(module: torch.nn.Module, *bundles: object) -> None:
    """An intervention that changes nothing; these tests read the schedule, not weights."""


def _spec(region: torch.nn.Module, *flows: DataFlow) -> SubgraphSpec:
    """A spec that runs `_noop` on `region` and declares `flows`."""
    return SubgraphSpec(region=region, fn=_noop, flows=list(flows))


def _named_optimize(name: str) -> Callable[..., None]:
    """Build a no-op intervention that names its own region.

    `OptimizeModule` prints the class of its module and the name of its function,
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
        case OptimizeModule():
            return instruction.fn.__name__
        case _:
            return type(instruction).__name__


def _calls(program: InstructionProgram, generator: FlowGenerator | None = None) -> list[CallModule]:
    """The calls of `program`, of one stream if `generator` is given."""
    return [
        instruction
        for instruction in program.instructions
        if isinstance(instruction, CallModule)
        and (generator is None or generator.context in instruction.contexts)
    ]


def _optimizes(program: InstructionProgram) -> list[OptimizeModule]:
    """The interventions of `program`, in the order they run."""
    return [i for i in program.instructions if isinstance(i, OptimizeModule)]


def _stream_sequence(program: InstructionProgram) -> list[object]:
    """The streams of `program`, in the order their first call appears."""
    sequence: list[object] = []
    for call in _calls(program):
        if call.contexts[0] not in sequence:
            sequence.append(call.contexts[0])
    return sequence


def _assert_reads_produced(program: InstructionProgram) -> None:
    """Assert every NodeRef an OptimizeModule reads was produced by a prior CallModule."""
    produced: set[tuple[_BaseRef, object]] = set()
    for instruction in program.instructions:
        match instruction:
            case CallModule(target=target, contexts=contexts):
                produced.update((target, ctx) for ctx in contexts)
            case OptimizeModule(bundles=bundles):
                missing = [
                    (ref, bundle.context)
                    for bundle in bundles
                    for ref in (*bundle.args, *bundle.kwargs.values())
                    if isinstance(ref, NodeRef) and (ref, bundle.context) not in produced
                ]
                assert not missing, f"refs not in register: {missing}"


def _plain_context() -> Callable[[torch.nn.Module], ContextManager[None]]:
    """A context factory that constrains nothing, with an identity of its own.

    The scheduler keys a stream on the factory object, so two generators that
    share one factory are one stream. Each test generator needs its own.
    """

    def context(_: torch.nn.Module) -> ContextManager[None]:
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
    assert sum(1 for kind, _ in cached if kind == "OptimizeModule") == 12


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
    assert {call.contexts[0] for call in calls} == {ORIGINAL.context}

    # AND the intervention fires after the full stream
    opt_index = next(i for i, x in enumerate(program.instructions) if isinstance(x, OptimizeModule))
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
        if isinstance(instruction, OptimizeModule)
    }
    pinned_calls = [
        (i, instruction)
        for i, instruction in enumerate(instructions)
        if isinstance(instruction, CallModule) and ORIGINAL.context in instruction.contexts
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
        if isinstance(instruction, OptimizeModule) and instruction.module is two_linear.fc1
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
    def single_use(_: torch.nn.Module) -> Iterator[None]:
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
