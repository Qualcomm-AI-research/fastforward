# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Compile a GraphModule and its specs into an InstructionProgram.

Given a `GraphModule` and an optional list of specs (each an algorithm to run
at a region, together with the data it needs to see), the scheduler produces
a sequence of instructions that executes the graph and dispatches the
interventions at the right points. When no specs are supplied, the program is
just a plain forward pass.

For example, take `fc2(act(fc1(x)))` and one spec on `fc1` and one on `fc2`. Each
spec asks for its own inputs twice: once under a pinned stream and once under a live
one. The scheduler emits:

    CallModule(fc1, stream=pinned)                # fc1 is the next to change, so
                                                  # take it now
    OptimizeModule(fc1, streams=[pinned, live])   # fc1 reads x: no call
    CallModule(act, stream=pinned)                # nothing optimizes act, so it
                                                  # can wait
    CallModule(fc1, stream=live)                  # fc1 changed, so the inputs of
    CallModule(act, stream=live)                  # fc2 are stale
    OptimizeModule(fc2, streams=[pinned, live])

The pinned pair runs on the weights from before the interventions, so one pair
serves both specs. The live pair reads the newest weights, so it runs again once
`fc1` holds new ones.
"""

from __future__ import annotations

import abc
import collections
import dataclasses

from collections.abc import Iterator, Mapping, Sequence
from typing import override

import attrs
import torch

from fastforward._orchestration.data_flow import (
    ANY,
    DataFlow,
    FlowGenerator,
    InputActivations,
    OutputActivations,
)
from fastforward._orchestration.graph_module import (
    AttributeRef,
    Const,
    GraphModule,
    InputRef,
    NodeRef,
    Op,
    SubgraphSpec,
    _BaseRef,
    descendants,
)
from fastforward._orchestration.instruction_engine import (
    BundleSpec,
    CallFunction,
    CallMethod,
    CallModule,
    Instruction,
    InstructionProgram,
    Instructions,
    LoadAttribute,
    OptimizeModule,
    ReturnOutputs,
    StoreValue,
    StreamKey,
)


@dataclasses.dataclass(frozen=True)
class FlowPlan:
    """A flow resolved against a concrete graph.

    Wraps the original `flow` alongside the refs the algorithm reads. A flow can
    have various outputs - InputActivations specify the total args/kwargs it should
    produce, OutputActivations specify only a single output value. These values are
    captured in `reads` and `read_kwargs`.

    Which calls have to run for those refs to hold a value is not part of the plan.
    The scheduler walks the graph from the refs, and it works out the calls together
    with the model state each one must show.

    Args:
        flow: The original declaration this plan resolves.
        reads: Positional refs the algorithm receives.
        read_kwargs: Keyword refs the algorithm receives.
    """

    flow: DataFlow
    reads: tuple[_BaseRef, ...] = ()
    read_kwargs: Mapping[str, _BaseRef] = dataclasses.field(default_factory=dict)

    @property
    def generator(self) -> FlowGenerator:
        """The flow generator that defines execution context for this plan."""
        return self.flow.generator

    @property
    def order(self) -> int:
        """Tie-break order among flows that can run at the same moment."""
        return self.generator.order

    @property
    def pinned(self) -> bool:
        """Whether this flow shows the weights from before every mutation."""
        return self.generator.pinned

    @property
    def cache(self) -> bool:
        """Whether the work done to produce this data may be reused."""
        return self.flow.cache

    def refs(self) -> Iterator[_BaseRef]:
        """Every ref this plan reads, positional first."""
        yield from self.reads
        yield from self.read_kwargs.values()


def bind_flows(graph: GraphModule, region: NodeRef, flows: Sequence[DataFlow]) -> list[FlowPlan]:
    """Bind each declared flow to the refs it reads on `graph`.

    Args:
        graph: The graph the flows are bound against.
        region: The node the flows are declared on.
        flows: The declared flows to bind.

    Raises:
        TypeError: If a flow is not a supported `DataFlow` type.
    """
    node = graph.node(region)

    plans: list[FlowPlan] = []
    for flow in flows:
        read_kwargs: Mapping[str, _BaseRef] = {}
        match flow:
            case InputActivations():
                # What arrives at the region: its own arguments.
                reads = tuple(node.args)
                read_kwargs = node.kwargs
            case OutputActivations():
                # What leaves the region: the region's own output.
                reads = (region,)
            case _:
                msg = f"unsupported DataFlow type: {type(flow).__name__}"
                raise TypeError(msg)

        plans.append(FlowPlan(flow=flow, reads=reads, read_kwargs=read_kwargs))
    return plans


def ref_load_instructions(ref: _BaseRef, context: StreamKey) -> list[StoreValue | LoadAttribute]:
    """Return the instructions needed to load a reference into the register.

    Node references and inputs should be available and need no instructions.
    Constant values and attribute accesses need a StoreValue/LoadAttribute instruction.

    Args:
        ref: The reference to load.
        context: The stream context under which the load runs.

    Raises:
        TypeError: If the reference is not a supported `_BaseRef` type.
    """
    match ref:
        case NodeRef() | InputRef():
            return []
        case Const():
            return [StoreValue(target=ref, value=ref, contexts=[context])]
        case AttributeRef(reference=attr_ref, attribute=attr):
            base_instructions = ref_load_instructions(attr_ref, context)
            return [*base_instructions, LoadAttribute(source=attr_ref, target=ref, attribute=attr)]
        case _BaseRef():
            msg = f"Unsupported reference type: {type(ref).__name__}"
            raise TypeError(msg)


@dataclasses.dataclass(frozen=True)
class _Intervention(abc.ABC):
    """A scheduled graph intervention that might mutate model state.

    An intervention could be an algorithm that the scheduler inserts (e.g. for
    optimizing a `module`, or performing a statistical analysis on data `flows`).
    The scheduler ensures each intervention's data flows are satisfied before emitting it.
    """

    region: NodeRef
    module: torch.nn.Module
    spec: SubgraphSpec
    flows: list[FlowPlan]

    @abc.abstractmethod
    def schedule(self, scheduler: Scheduler) -> None: ...


@dataclasses.dataclass(frozen=True)
class OptimizeIntervention(_Intervention):
    """An intervention that optimizes a region's weights in-place."""

    @override
    def schedule(self, scheduler: Scheduler) -> None:
        bundles: list[BundleSpec] = []
        for plan in self.flows:
            scheduler.schedule_loads(list(plan.refs()), plan.generator)
            bundles.append(
                BundleSpec(
                    context=plan.generator.context,
                    args=list(plan.reads),
                    kwargs=dict(plan.read_kwargs),
                )
            )

        scheduler.instructions.append(
            OptimizeModule(module=self.module, fn=self.spec.fn, bundles=tuple(bundles))
        )

        # The weights of this region are now different from the ones every value
        # in flight was made with.
        scheduler.timeline.advance(self.region)


@attrs.define(frozen=True, cache_hash=True, repr=False)
class Value:
    """One version of one register entry under one flow generator.

    The scheduler compares these to decide which instructions it must emit, and in
    which order. `generation` separates the weight eras: two values with the same
    ref and generator but a different generation need separate instructions. At run
    time only `ref` and the generator's context remain, as a register key.

    Args:
        ref: The reference that names the value. A `NodeRef` for a value a call
            produces; a `Const` or `AttributeRef` for one a load puts there.
        generator: The stream the node runs under.
        generation: The number of mutations the weights of this value include. A
            pinned stream always names generation 0, the state from before every
            mutation.
    """

    ref: _BaseRef
    generator: FlowGenerator
    generation: int

    def __repr__(self) -> str:
        return f"{self.ref!r}@{self.generation}:{self.generator.key}"


class Timeline:
    """Which mutations each value includes, and which values are still producible.

    When an intervention (potentially) changes the weights of a region, the data of
    that region and of everything downstream of it changes with it. This Timeline
    keeps track of it all.
    """

    def __init__(self, graph: GraphModule) -> None:
        self.graph = graph
        self.mutations = 0
        self._mutated: collections.defaultdict[_BaseRef, int] = collections.defaultdict(int)
        self._generation: collections.defaultdict[_BaseRef, int] = collections.defaultdict(int)
        self._invalidated: dict[NodeRef, frozenset[NodeRef]] = {}

    def invalidated_by(self, region: NodeRef) -> frozenset[NodeRef]:
        """Return every node that a mutation of `region` invalidates, `region` included.

        A node is invalidated when it would produce different data than before, so
        every value it has already produced belongs to an earlier generation and
        cannot be produced again.
        """
        if region not in self._invalidated:
            nodes, _ = descendants(self.graph, region)
            self._invalidated[region] = frozenset(nodes)
        return self._invalidated[region]

    def advance(self, region: NodeRef) -> None:
        """Record one mutation of `region`, and invalidate everything it reaches."""
        self.mutations += 1
        self._mutated[region] = self.mutations
        for node in self.invalidated_by(region):
            self._generation[node] = self.mutations

    def value(self, ref: _BaseRef, generator: FlowGenerator) -> Value:
        """Return the `Value` for `ref` under `generator`, at the current generation."""
        generation = 0 if generator.pinned else self._generation[ref.unwrap_ref()]
        return Value(ref=ref, generator=generator, generation=generation)

    def input_values(self, value: Value) -> Iterator[Value]:
        """Return the values `value` reads, each under the same generator as `value`."""
        assert isinstance(value.ref, NodeRef)
        for arg in self.graph.node_inputs(value.ref):
            yield self.value(arg, value.generator)

    def is_stale(self, value: Value) -> bool:
        """Return whether the weights `value` shows are gone, so it cannot be produced."""
        base = value.ref.unwrap_ref()
        if value.generator.pinned:
            return self._mutated[base] != 0
        return value.generation != self._generation[base]


class Scheduler:
    """Compiles a graph and its interventions into instructions."""

    def __init__(self, graph: GraphModule) -> None:
        self.graph = graph
        self.instructions: list[Instruction] = []
        self.register: set[Value] = set()
        self.timeline = Timeline(graph)
        self._pending: list[tuple[FlowPlan, Value]] = []

    def production_order(self, value: Value) -> list[Value]:
        """The values to call, inputs first, to put `value` in the register.

        Walks upstream from `value` and stops at every value the register already
        holds. The walk visits each value once, so a diamond in the graph costs one
        visit and not one per path through it. Inputs go on the stack in reverse, so
        that the calls come out in declaration order.
        """
        order: list[Value] = []
        done: set[Value] = set()
        stack: list[tuple[Value, bool]] = [(value, False)]

        while stack:
            current, expanded = stack.pop()
            if current in self.register or current in done:
                continue
            if expanded:
                done.add(current)
                order.append(current)
                continue
            stack.append((current, True))
            reads = list(self.timeline.input_values(current))
            stack.extend((read, False) for read in reversed(reads))

        return order

    def can_produce(self, value: Value) -> bool:
        """Return whether the program can still produce `value`.

        Every call the register does not already cover has to run on the weights it
        names, so none of them may be stale.
        """
        return not any(self.timeline.is_stale(item) for item in self.production_order(value))

    def schedule_loads(self, refs: Sequence[_BaseRef], generator: FlowGenerator) -> None:
        """Schedule load instructions for references not yet in the register.

        Args:
            refs: References to check and load.
            generator: The flow generator whose context the loads run under.
        """
        for ref in refs:
            for instruction in ref_load_instructions(ref, generator.context):
                value = self.timeline.value(instruction.target, generator)
                if value not in self.register:
                    self.register.add(value)
                    self.instructions.append(instruction)

    def schedule_value(self, value: Value, *, cache: bool) -> None:
        """Schedule the calls that put `value` in the register, inputs first.

        Args:
            value: The value to produce.
            cache: Whether the results may be reused by later reads.
        """
        for item in self.production_order(value):
            assert isinstance(item.ref, NodeRef)
            node = self.graph.node(item.ref)
            self.schedule_loads([*node.args, *node.kwargs.values()], item.generator)
            common = dict(
                args=list(node.args),
                kwargs=dict(node.kwargs),
                target=item.ref,
                contexts=[item.generator.context],
                cache=cache,
            )
            instruction: Instruction
            match node.op:
                case Op.torch_module:
                    instruction = CallModule(module=node.target, **common)  # type: ignore[arg-type]
                case Op.call_function | Op.get_attr:
                    instruction = CallFunction(fn=node.target, **common)  # type: ignore[arg-type]
                case Op.call_method:
                    instruction = CallMethod(method=node.target, **common)  # type: ignore[arg-type]
            self.instructions.append(instruction)
            self.register.add(item)

    def demands_of(self, plan: FlowPlan) -> list[Value]:
        """Return the values `plan` needs in the register, named as of right now.

        A read of an attribute of a node needs the node itself, so the base ref is
        what has to be produced; `schedule_loads` adds the `LoadAttribute` on top.
        """
        values: list[Value] = []
        for ref in plan.refs():
            if isinstance(base := ref.unwrap_ref(), NodeRef):
                values.append(self.timeline.value(base, plan.generator))
        return values

    def schedule_expiring_values(self, region: NodeRef) -> None:
        """Schedule the values that depend on the current weights of `region`.

        Call this before an intervention changes `region`. Any flow that still waits for
        a value below `region` gets the output of `region` now, and computes the rest
        from it later.
        """
        invalidated = self.timeline.invalidated_by(region)
        scheduled: set[FlowGenerator] = set()

        for plan, demand in self._pending:
            if plan.generator in scheduled or demand.ref not in invalidated:
                continue
            if demand in self.register:
                continue
            scheduled.add(plan.generator)
            self.schedule_value(self.timeline.value(region, plan.generator), cache=plan.cache)

    def schedule_intervention(self, intervention: _Intervention) -> None:
        """Satisfy every data flow of one intervention, then emit it.

        Args:
            intervention: The graph intervention to schedule.

        Raises:
            RuntimeError: If a flow asks for data the program can no longer produce.
        """
        self.schedule_expiring_values(intervention.region)

        for plan in sorted(intervention.flows, key=lambda plan: plan.order):
            for demand in self.demands_of(plan):
                if not self.can_produce(demand):
                    msg = (
                        f"Cannot produce {demand!r} for the {plan.generator.key!r} flow on "
                        f"{intervention.region!r}: an earlier intervention has already replaced "
                        f"the weights it reads."
                    )
                    raise RuntimeError(msg)
                self.schedule_value(demand, cache=plan.cache)

        intervention.schedule(self)
        self._pending = [entry for entry in self._pending if entry[1] not in self.register]

    def schedule_outputs(self, generator: FlowGenerator) -> None:
        """Schedule the `ReturnOutputs` instruction that returns the graph outputs.

        Args:
            generator: The flow generator whose context the output refs are loaded under.
        """
        if not self.graph._outputs:
            return

        self.schedule_loads(list(self.graph._outputs), generator)
        self.instructions.append(ReturnOutputs(outputs=list(self.graph._outputs)))

    def run(self, interventions: Sequence[_Intervention]) -> Instructions:
        """Schedule the instructions needed to run a GraphModule.

        When no interventions are provided the scheduler produces a full forward
        pass that returns the graph outputs.  When interventions are provided
        no ``ReturnOutputs`` is emitted — interventions mutate weights in-place
        and do not produce a model output.

        Args:
            interventions: The graph interventions to schedule.

        Returns:
            A sequence of instructions that satisfy the interventions if
                provided, otherwise instructions that satisfy a forward pass
                of the original GraphModule.
        """
        if not interventions:
            for node in self.graph.topo_order:
                self.schedule_value(self.timeline.value(node, ANY), cache=False)
            self.schedule_outputs(ANY)
            return self.instructions

        self._pending = [
            (plan, demand)
            for intervention in interventions
            for plan in intervention.flows
            if plan.pinned
            for demand in self.demands_of(plan)
        ]
        self._pending.sort(key=lambda entry: entry[0].order)

        for intervention in interventions:
            self.schedule_intervention(intervention)

        return self.instructions


def _check_order_unambiguous(interventions: Sequence[_Intervention]) -> None:
    """Ensure that no two distinct flows have the same `order` value.

    The order value defines order-of-operations, which flow must run first.
    If a (probably user-specified) flow has an order that is equal to another
    existing flow, this creates a contradiction and should be raised.
    """
    keys_by_order: dict[int, set[str]] = collections.defaultdict(set)
    for intervention in interventions:
        for plan in intervention.flows:
            keys_by_order[plan.order].add(plan.generator.key)

    for order, keys in sorted(keys_by_order.items()):
        if len(keys) > 1:
            listed = ", ".join(repr(key) for key in sorted(keys))
            msg = (
                f"Flow generators {listed} each declare order={order}. "
                f"`order` decides which stream runs first, so distinct "
                f"generators need distinct values."
            )
            raise ValueError(msg)


def schedule(graph: GraphModule, specs: Sequence[SubgraphSpec] = ()) -> InstructionProgram:
    """Compile `graph` and its `specs` into an `InstructionProgram`.

    With no `specs`, the program is a plain forward pass. With `specs`, it holds the
    `CallModule`s each declared `DataFlow` needs and the `OptimizeModule`s that
    consume them, in an order that honours `order` and the mutations the
    interventions perform.

    Args:
        graph: The graph to schedule.
        specs: The specs to schedule, each declaring a region, a function, and the
            data flows it requires. Empty means a plain forward pass.

    Returns:
        The instruction program to run.

    Raises:
        ValueError: If two generators in use share an `order`, or if a spec declares
            an `ANY` flow.
    """
    interventions: list[_Intervention] = []

    # Collect the requirements of every spec into interventions.
    for spec in specs:
        module = spec.resolve(graph).target
        region = graph.node_ref(module)

        if any(flow.generator is ANY for flow in spec.flows):
            msg = f"Spec on {region!r} declares an ANY flow. ANY is reserved for plain forward passes."
            raise ValueError(msg)

        interventions.append(
            OptimizeIntervention(
                region=region, module=module, spec=spec, flows=bind_flows(graph, region, spec.flows)
            )
        )

    # Interventions run in graph order, a region's inputs must already be generated.
    topo_index = {ref: i for i, ref in enumerate(graph.topo_order)}
    interventions.sort(key=lambda intervention: topo_index[intervention.region])
    _check_order_unambiguous(interventions)

    instructions = Scheduler(graph).run(interventions)

    return InstructionProgram(instructions=instructions, input_refs=graph._inputs)
