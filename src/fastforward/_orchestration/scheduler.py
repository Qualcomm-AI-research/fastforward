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
    CallDelegate(fc1, streams=[pinned, live])     # fc1 reads x: no call
    CallModule(act, stream=pinned)                # nothing optimizes act, so it
                                                  # can wait
    CallModule(fc1, stream=live)                  # fc1 changed, so the inputs of
    CallModule(act, stream=live)                  # fc2 are stale
    CallDelegate(fc2, streams=[pinned, live])

The pinned pair runs on the weights from before the interventions, so one pair
serves both specs. The live pair reads the newest weights, so it runs again once
`fc1` holds new ones.
"""

from __future__ import annotations

import abc
import collections
import dataclasses
import functools

from collections.abc import Hashable, Iterator, Mapping, Sequence
from functools import cached_property

import attrs
import torch

from typing_extensions import override

from fastforward._orchestration.data_flow import (
    ANY,
    FlowGenerator,
    FlowInput,
    FlowOp,
    Gradient,
    InputActivations,
    OutputActivations,
)
from fastforward._orchestration.graph_module import (
    AttributePath,
    AttributeRef,
    Const,
    FlowOpRef,
    GradientRef,
    GraphModule,
    InputRef,
    LossGradRef,
    Node,
    NodeRef,
    Op,
    SubgraphSpec,
    VJPRef,
    _BaseRef,
    descendants,
)
from fastforward._orchestration.instruction_engine import (
    ActivationDataset,
    BundleSpec,
    CallFunction,
    CallMethod,
    CallModule,
    CallVJP,
    Instruction,
    InstructionProgram,
    Instructions,
    LoadAttribute,
    ReturnOutputs,
    RunDelegate,
    StoreValue,
    StreamKey,
    assemble_grad_output,
    call_flow_op,
    loss_grad_outputs,
    sum_grad_outputs,
)


@dataclasses.dataclass(frozen=True)
class FlowPlan:
    """A flow resolved against a concrete graph.

    Wraps the original `flow` alongside the refs the algorithm reads. A flow can
    have various outputs - InputActivations specify the total args/kwargs it should
    produce, OutputActivations specify only a single output value. These values are
    captured in `reads` and `read_kwargs`. A `FlowOp` reads one ref, a `FlowOpRef`,
    which holds the plans of the inputs of the op.

    Which calls have to run for those refs to hold a value is not part of the plan.
    The scheduler walks the graph from the refs, and it works out the calls together
    with the model state each one must show.

    Args:
        flow: The original declaration this plan resolves.
        reads: Positional refs the algorithm receives.
        read_kwargs: Keyword refs the algorithm receives.
    """

    flow: FlowInput
    reads: tuple[_BaseRef, ...] = ()
    read_kwargs: Mapping[str, _BaseRef] = dataclasses.field(default_factory=dict)

    def __hash__(self) -> int:
        return hash((self.flow, self.reads, frozenset(self.read_kwargs.items())))

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

    @property
    def op_ref(self) -> FlowOpRef | None:
        """The ref to the result of the op this plan resolves, or `None` for a plain flow."""
        if not isinstance(self.flow, FlowOp):
            return None
        (ref,) = self.reads
        assert isinstance(ref, FlowOpRef)
        return ref

    def refs(self) -> Iterator[_BaseRef]:
        """Every ref this plan reads, positional first."""
        yield from self.reads
        yield from self.read_kwargs.values()

    def leaves(self) -> Iterator[FlowPlan]:
        """The plans that read the graph: this plan, or the leaves of the inputs of an op."""
        if (ref := self.op_ref) is None:
            yield self
        else:
            for plan in ref.inputs:
                yield from plan.leaves()


def bind_flows(graph: GraphModule, region: NodeRef, flows: Sequence[FlowInput]) -> list[FlowPlan]:
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
            case Gradient():
                # The total gradient of the region's own output.
                reads = (GradientRef(region, flow),)
            case FlowOp():
                # The result of the op, on what its inputs read at this region, and the
                # settings of the module of this region.
                op_inputs = bind_flows(graph, region, [value for _, value in flow.inputs])
                settings: dict[str, Hashable] = {}
                if flow.from_module is not None:
                    assert isinstance(node.target, torch.nn.Module)
                    settings = dict(flow.from_module(node.target))
                reads = (FlowOpRef(flow, tuple(op_inputs), tuple(sorted(settings.items()))),)
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
        case NodeRef() | InputRef() | GradientRef() | LossGradRef() | VJPRef() | FlowOpRef():
            return []
        case Const():
            return [StoreValue(target=ref, value=ref, context=context)]
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
    optimizing a `module`, or performing a statistical analysis on data `flows`), or
    a step that produces data for later interventions, such as a backward pass. Each
    intervention satisfies its data flows through the scheduler, then emits itself.

    Args:
        flows: The data the intervention reads.
    """

    flows: list[FlowPlan]

    @abc.abstractmethod
    def schedule(self, scheduler: Scheduler) -> None: ...

    def leaf_plans(self) -> Iterator[FlowPlan]:
        """The plans that read the graph: the flows, and the inputs of the ops among them."""
        for plan in self.flows:
            yield from plan.leaves()


@dataclasses.dataclass(frozen=True)
class OptimizeIntervention(_Intervention):
    """An intervention that optimizes a region's weights in-place."""

    region: NodeRef
    module: torch.nn.Module
    spec: SubgraphSpec

    @override
    def schedule(self, scheduler: Scheduler) -> None:
        scheduler.schedule_expiring_values(self.region)
        scheduler.schedule_flows(self.flows, f"on {self.region!r}")

        inputs = scheduler.delegate_inputs(self.flows)
        scheduler.instructions.append(
            RunDelegate(fn=self.spec.fn, bundles=tuple(inputs), module=self.module)
        )

        # The weights of this region are now different from the ones every value
        # in flight was made with.
        scheduler.timeline.advance(self.region)


@dataclasses.dataclass(frozen=True)
class GradientIntervention(_Intervention):
    """An intervention that runs the backward pass, and changes no weight.

    The gradients are made from the last region back to the first. Each op in `ops` runs as
    soon as the gradients it reads are made, so the data it reads can be freed before the
    backward pass is done.

    Args:
        ops: The plans of the pinned `FlowOp`s that read a gradient.
    """

    ops: list[FlowPlan] = dataclasses.field(default_factory=list)

    @override
    def schedule(self, scheduler: Scheduler) -> None:
        topo_index: dict[NodeRef | InputRef, int] = {
            ref: i for i, ref in enumerate(scheduler.graph.topo_order)
        }
        gradients = [
            (base, plan.generator)
            for plan in self.flows
            for ref in plan.refs()
            if isinstance(base := ref.unwrap_ref(), GradientRef)
        ]
        gradients.sort(key=lambda item: topo_index.get(item[0].reference, -1), reverse=True)

        waiting = list(self.ops)
        for gradient, generator in gradients:
            scheduler.schedule_gradient(gradient, generator)
            ready = [op for op in waiting if _gradients_made(op, scheduler)]
            scheduler.schedule_flows(ready, "in the backward pass")
            waiting = [op for op in waiting if op not in ready]


def _gradients_made(plan: FlowPlan, scheduler: Scheduler) -> bool:
    """Return whether the register holds every gradient that `plan` reads."""
    return all(
        scheduler.timeline.value(base, leaf.generator) in scheduler.register
        for leaf in plan.leaves()
        for ref in leaf.refs()
        if isinstance(base := ref.unwrap_ref(), GradientRef)
    )


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

    @cached_property
    def _output_uses(self) -> dict[NodeRef | InputRef, list[tuple[int, AttributePath]]]:
        """The position of each graph output, and the part of the value it is."""
        uses: dict[NodeRef | InputRef, list[tuple[int, AttributePath]]] = collections.defaultdict(
            list
        )
        for position, output in enumerate(self.graph._outputs):
            base, path = output.unwrap_path()
            if isinstance(base, NodeRef | InputRef):
                uses[base].append((position, path))
        return uses

    @cached_property
    def _differentiable(self) -> frozenset[NodeRef | InputRef]:
        """The node outputs and graph inputs that reach the graph outputs, so get a gradient."""
        reached: set[NodeRef | InputRef] = set()
        bases: list[NodeRef | InputRef] = [
            *reversed(self.graph.topo_order),
            *self.graph._inputs.values(),
        ]
        for base in bases:
            if base in self._output_uses or any(
                node in reached for node, _, _ in self.graph.readers(base)
            ):
                reached.add(base)
        return frozenset(reached)

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
                context=item.generator.context,
                cache=cache,
            )
            instruction: Instruction
            match node.op:
                case Op.torch_module:
                    instruction = CallModule(caller=node.target, **common)  # type: ignore[arg-type]
                case Op.call_function | Op.get_attr:
                    instruction = CallFunction(caller=node.target, **common)  # type: ignore[arg-type]
                case Op.call_method:
                    instruction = CallMethod(caller=node.target, **common)  # type: ignore[arg-type]
            self.instructions.append(instruction)
            self.register.add(item)

    def delegate_inputs(self, plans: Sequence[FlowPlan]) -> list[BundleSpec]:
        """Schedule the loads that `plans` read, and return the data each one hands over.

        Args:
            plans: The plans a delegate reads, in the order it receives them.
        """
        inputs: list[BundleSpec] = []
        for plan in plans:
            self.schedule_loads(list(plan.refs()), plan.generator)
            inputs.append(
                BundleSpec(
                    context=plan.generator.context,
                    args=list(plan.reads),
                    kwargs=dict(plan.read_kwargs),
                    unwrap=plan.op_ref is not None,
                )
            )
        return inputs

    def schedule_flow_op(self, ref: FlowOpRef, generator: FlowGenerator, owner: str) -> None:
        """Schedule the call of a `FlowOp` that puts its result in the register.

        The call runs once, and changes no weight, so the timeline does not advance.

        Args:
            ref: The ref to the result of the op.
            generator: The flow generator that the result is stored under.
            owner: Names the intervention in an error, for example `"on fc1"`.
        """
        value = self.timeline.value(ref, generator)
        if value in self.register:
            return

        self.schedule_flows(ref.inputs, owner)
        names = [name for name, _ in ref.op.inputs]
        fn = functools.partial(ref.op.fn, **dict(ref.settings))
        self.instructions.append(
            RunDelegate(
                fn=functools.partial(call_flow_op, fn, names),
                bundles=self.delegate_inputs(ref.inputs),
                targets=[(ref, generator.context)],
            )
        )
        self.register.add(value)

    def schedule_gradient(self, ref: GradientRef, generator: FlowGenerator) -> None:
        """Schedule the backward steps that put the gradient `ref` in the register.

        The gradient of a value needs the gradient of each node that reads it, so the
        steps run from the graph outputs back to `ref`. Each step that the register
        already holds is used again, so objectives and VJPs run once.

        Args:
            ref: The gradient to produce.
            generator: The flow generator that the forward values and the steps use.
        """
        needed: set[NodeRef | InputRef] = set()
        stack = [ref.reference]
        while stack:
            base = stack.pop()
            gradient = GradientRef(base, ref.gradient)
            if base in needed or self.timeline.value(gradient, generator) in self.register:
                continue
            needed.add(base)
            stack.extend(
                node for node, _, _ in self.graph.readers(base) if node in self._differentiable
            )

        # A gradient needs the gradients of the nodes that read it, so they go first.
        topo_index: dict[NodeRef | InputRef, int] = {
            node: idx for idx, node in enumerate(self.graph.topo_order)
        }
        for base in sorted(needed, key=lambda base: topo_index.get(base, -1), reverse=True):
            self._schedule_total_gradient(GradientRef(base, ref.gradient), generator)

    def _schedule_total_gradient(self, ref: GradientRef, generator: FlowGenerator) -> None:
        """Schedule the step that adds every contribution to the gradient `ref`.

        Each graph output and each node that reads the value sends a contribution to
        one part of it. A value that does not reach the graph outputs gets `None`.
        """
        context = generator.context
        base = ref.reference
        parts: list[tuple[AttributePath, _BaseRef]] = []
        if base in self._differentiable:
            for position, path in self._output_uses[base]:
                loss_grads = self._schedule_loss_grads(ref.gradient, generator, position)
                parts.extend((path, loss_grad) for loss_grad in loss_grads)
            for node, argument, read in self.graph.readers(base):
                if node in self._differentiable:
                    vjp = self._schedule_vjp(GradientRef(node, ref.gradient), generator)
                    _, path = read.unwrap_path()
                    parts.append((path, vjp[argument]))

        instruction: Instruction
        paths = [path for path, _ in parts]
        grad_outputs = [grad_output for _, grad_output in parts]
        self.schedule_loads(grad_outputs, generator)
        if not parts:
            # The loss does not depend on this value, so its gradient is None.
            instruction = StoreValue(target=ref, value=ActivationDataset([None]), context=context)
        elif all(path == () for path in paths):
            # Each contribution is for the full value, so only add them.
            instruction = CallFunction(
                caller=sum_grad_outputs,
                args=grad_outputs,
                kwargs={},
                target=ref,
                context=context,
                cache=True,
            )
        else:
            # Some contributions are for a part only, so place each one in the value.
            # A node output must be in the register; a graph input always is.
            if isinstance(base, NodeRef):
                self.schedule_value(self.timeline.value(base, generator), cache=True)
            instruction = CallFunction(
                caller=functools.partial(assemble_grad_output, paths),
                args=[base, *grad_outputs],
                kwargs={},
                target=ref,
                context=context,
                cache=True,
            )
        self.instructions.append(instruction)
        self.register.add(self.timeline.value(ref, generator))

    def _schedule_loss_grads(
        self, gradient: Gradient, generator: FlowGenerator, position: int
    ) -> list[LossGradRef]:
        """Return the gradients of the loss at the graph output at `position`.

        The gradients of the loss with respect to the graph output at `position`, one
        for each name that the objective of `gradient` reads the model outputs under.
        The objective is called once for all graph outputs.
        """
        names = list(gradient.outputs)
        loss_grads = [LossGradRef(gradient, name, position) for name in names]
        # The objective already ran for another graph output, so do not call it again.
        if self.timeline.value(loss_grads[0], generator) in self.register:
            return loss_grads

        outputs = list(self.graph._outputs)
        for output in outputs:
            if isinstance(base := output.unwrap_ref(), NodeRef):
                self.schedule_value(self.timeline.value(base, generator), cache=True)
        self.schedule_loads(outputs, generator)

        targets = [
            LossGradRef(gradient, name, idx) for name in names for idx in range(len(outputs))
        ]
        self.instructions.append(
            RunDelegate(
                fn=functools.partial(loss_grad_outputs, gradient.f, names, gradient.values),
                bundles=[BundleSpec(generator.context, outputs) for _ in names],
                targets=[(target, generator.context) for target in targets],
            )
        )
        self.register.update(self.timeline.value(target, generator) for target in targets)
        return loss_grads

    def _schedule_vjp(self, grad_output: GradientRef, generator: FlowGenerator) -> VJPRef:
        """Schedule the VJP that runs `grad_output` back through its node, and return its ref.

        The VJP gives the gradient of each argument of the node, keyed by position or keyword.
        """
        node_ref = grad_output.reference
        assert isinstance(node_ref, NodeRef)
        target = VJPRef(node_ref, grad_output.gradient)
        if self.timeline.value(target, generator) in self.register:
            return target

        node = self.graph.node(node_ref)
        for arg in self.timeline.input_values(self.timeline.value(node_ref, generator)):
            self.schedule_value(arg, cache=True)
        self.schedule_loads([*node.args, *node.kwargs.values()], generator)
        self.instructions.append(
            CallVJP(
                caller=node.target,
                args=[*node.args, grad_output],
                kwargs=dict(node.kwargs),
                target=target,
                context=generator.context,
                cache=True,
            )
        )
        self.register.add(self.timeline.value(target, generator))
        return target

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

    def schedule_flows(self, flows: Sequence[FlowPlan], owner: str) -> None:
        """Schedule the calls that put every value `flows` read in the register.

        Args:
            flows: The flows of one intervention.
            owner: Names the intervention in an error, for example `"on fc1"`.

        Raises:
            RuntimeError: If a flow asks for data the program can no longer produce.
        """
        for plan in sorted(flows, key=lambda plan: plan.order):
            if (op_ref := plan.op_ref) is not None:
                self.schedule_flow_op(op_ref, plan.generator, owner)
            for demand in self.demands_of(plan):
                if not self.can_produce(demand):
                    msg = (
                        f"Cannot produce {demand!r} for the {plan.generator.key!r} flow "
                        f"{owner}: an earlier intervention has already replaced the weights "
                        f"it reads."
                    )
                    raise RuntimeError(msg)
                self.schedule_value(demand, cache=plan.cache)

    def schedule_intervention(self, intervention: _Intervention) -> None:
        """Schedule one intervention, and drop the pending demands it has produced.

        Args:
            intervention: The graph intervention to schedule.
        """
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
            for plan in intervention.leaf_plans()
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
        for plan in intervention.leaf_plans():
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


def _node_parameters(node: Node) -> frozenset[int]:
    """Return the identities of the parameters that `node` reads.

    A module node reads the parameters of its module. A `get_attr` node reads the
    tensor it returns, which is a parameter when a model uses its weight directly.
    """
    if isinstance(node.target, torch.nn.Module):
        return frozenset(id(param) for param in node.target.parameters())
    if node.op is Op.get_attr and isinstance(value := node.target(), torch.nn.Parameter):
        return frozenset({id(value)})
    return frozenset()


def _check_no_shared_weights(graph: GraphModule, regions: Sequence[NodeRef]) -> None:
    """Ensure that no node outside a region reads a weight of that region.

    An intervention on a tied weight also changes every other node that reads it, so
    the result of the optimization is ambiguous.
    """
    parameters = {node: _node_parameters(graph.node(node)) for node in graph.topo_order}
    for region in regions:
        owned = _node_parameters(graph.node(region))
        for node, params in parameters.items():
            if node != region and params & owned:
                msg = f"Region {region!r} shares a weight with {node!r}. Untie the weights or remove the region from the targets."
                raise ValueError(msg)


def _check_eval_mode(graph: GraphModule) -> None:
    """Ensure that the user hands over the model in eval mode.

    A backward step runs its node a second time. In training mode, dropout and batch
    norm give a different result on the second call, so the gradient would not match
    the forward pass.

    The check reads module flags only. It cannot see a functional call, like
    `F.dropout(x, p, training=...)`.

    Raises:
        ValueError: If a module of `graph` is in training mode.
    """
    for ref in graph.topo_order:
        target = graph.node(ref).target
        if not isinstance(target, torch.nn.Module):
            continue
        for module in target.modules():
            if module.training and not isinstance(module, GraphModule):
                msg = (
                    f"Gradient flows need the model in eval mode, but {type(module).__name__} "
                    f"in {ref!r} is in training mode. Call model.eval() first."
                )
                raise ValueError(msg)


def schedule(graph: GraphModule, specs: Sequence[SubgraphSpec] = ()) -> InstructionProgram:
    """Compile `graph` and its `specs` into an `InstructionProgram`.

    With no `specs`, the program is a plain forward pass. With `specs`, it holds the
    `CallModule`s each declared `DataFlow` needs and the `CallDelegate`s that
    consume them, in an order that honours `order` and the mutations the
    interventions perform.

    Args:
        graph: The graph to schedule.
        specs: The specs to schedule, each declaring a region, a function, and the
            data flows it requires. Empty means a plain forward pass.

    Returns:
        The instruction program to run.

    Raises:
        ValueError: If two generators in use share an `order`, if a spec declares
            an `ANY` flow, or if a node outside a region reads a weight of that region.
    """
    optimizations: list[OptimizeIntervention] = []
    # Collect the requirements of every spec into interventions.
    for spec in specs:
        module = spec.resolve(graph).target
        region = graph.node_ref(module)

        flows = bind_flows(graph, region, spec.flows)
        if any(leaf.generator is ANY for plan in flows for leaf in plan.leaves()):
            msg = f"Spec on {region!r} declares an ANY flow. ANY is reserved for plain forward passes."
            raise ValueError(msg)

        optimizations.append(
            OptimizeIntervention(region=region, module=module, spec=spec, flows=flows)
        )

    # Optimizations run in graph order, a region's inputs must already be generated.
    topo_index = {ref: i for i, ref in enumerate(graph.topo_order)}
    optimizations.sort(key=lambda intervention: topo_index[intervention.region])

    # Gradients are only implemented on the weights from before every mutation, so
    # the backward pass runs before the first optimization.
    interventions: list[_Intervention] = list(optimizations)
    if gradients := [
        plan
        for optimization in optimizations
        for plan in optimization.leaf_plans()
        if isinstance(plan.flow, Gradient)
    ]:
        _check_eval_mode(graph)
        ops = [
            plan
            for optimization in optimizations
            for plan in optimization.flows
            if plan.op_ref is not None
            and plan.pinned
            and any(isinstance(leaf.flow, Gradient) for leaf in plan.leaves())
        ]
        interventions.insert(0, GradientIntervention(flows=gradients, ops=ops))
    _check_order_unambiguous(interventions)
    _check_no_shared_weights(graph, [optimization.region for optimization in optimizations])

    instructions = Scheduler(graph).run(interventions)

    return InstructionProgram(instructions=instructions, input_refs=graph._inputs)
