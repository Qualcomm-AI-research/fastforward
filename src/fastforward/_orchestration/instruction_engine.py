# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


import abc
import dataclasses
import functools
import itertools

from collections import defaultdict
from typing import (
    Any,
    Callable,
    Collection,
    ContextManager,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    TypeAlias,
)

import torch

from torch.utils.data import DataLoader

from fastforward._orchestration.graph_module import (
    Const,
    GraphModule,
    InputRef,
    Op,
    _BaseRef,
)
from fastforward._orchestration.location import DeviceLocation, Location

# Distinguishes data produced under different execution conditions for the same node.
# A node without a module (a free function or a method call) passes None.
StreamKey: TypeAlias = Callable[[torch.nn.Module | None], ContextManager[None]]

# Ordered sequence of context managers that an instruction executes under.
Contexts: TypeAlias = Sequence[Callable[[torch.nn.Module | None], ContextManager[None]]]


def _fmt_module(module: torch.nn.Module | Callable[..., Any]) -> str:
    """Format a module/callable target without dumping parameters or a memory address."""
    if isinstance(module, torch.nn.Module):
        return type(module).__name__
    return _fmt_callable(module)


def _fmt_callable(fn: Callable[..., Any]) -> str:
    """Format a callable by name, without its memory address."""
    while isinstance(fn, functools.partial):
        fn = fn.func
    return getattr(fn, "__name__", type(fn).__name__)


def _fmt_contexts(contexts: Contexts) -> str:
    """Format execution contexts by callable name, without their memory addresses."""
    return "[" + ", ".join(_fmt_callable(context) for context in contexts) + "]"


def _fmt_value(value: Any) -> str:
    """Format a register value, showing shape/dtype instead of tensor contents."""
    if isinstance(value, torch.Tensor):
        return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype})"
    return repr(value)


@dataclasses.dataclass(frozen=True)
class ActivationDataset(Collection[Any]):
    """A batched collection that wraps all data flowing through the InstructionEngine."""

    batches: list[Any]

    def __iter__(self) -> Iterator[Any]:
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)

    def __contains__(self, item: Any) -> bool:
        return item in self.batches

    @classmethod
    def from_value(cls, value: Any) -> "ActivationDataset":
        """Create an ActivationDataset from 'value'.

        If the value itself is an ActivationDataset we return it unchanged.

        Args:
            value: anything you want wrapped in an ActivationDataset.

        Returns:
            ActivationDataset with value(s) as batches.
        """
        if isinstance(value, cls):
            return value

        batches = list(value) if isinstance(value, (list, DataLoader)) else [value]
        return cls(batches)

    @classmethod
    def merge(cls, datasets: Sequence["ActivationDataset"]) -> "ActivationDataset":
        """Zip multiple ActivationDatasets.

        All datasets must have the same length.

        Args:
            datasets: Non-empty sequence of ActivationDatasets to merge.

        Returns:
            ActivationDataset where each batch is a tuple of corresponding elements.

        Raises:
            ValueError: If datasets have different lengths.
        """
        if len(datasets) == 0:
            msg = "ActivationDataset.merge expects at least one dataset"
            raise ValueError(msg)

        if len(datasets) == 1:
            return datasets[0]

        lengths = [len(ds) for ds in datasets]
        if len(set(lengths)) > 1:
            msg = f"Cannot merge datasets of different sizes: {lengths}"
            raise ValueError(msg)

        batches: list[tuple[Any, ...]] = []
        for i in range(lengths[0]):
            tpl = tuple(ds.batches[i] for ds in datasets)
            batches.append(tpl)

        return ActivationDataset(batches)

    @staticmethod
    def broadcast(datasets: Sequence["ActivationDataset"]) -> list["ActivationDataset"]:
        """Broadcast datasets to a common length.

        Datasets of length 1 are repeated to match N-length datasets.

        Args:
            datasets: Datasets to broadcast.

        Returns:
            List of datasets all at the same length.

        Raises:
            ValueError: If there are multiple non-1 lengths.
        """
        if not datasets:
            return []
        lengths = {len(ds) for ds in datasets}
        if len(lengths) <= 1:
            return list(datasets)
        lengths.discard(1)
        if len(lengths) != 1:
            msg = (
                "Dataset length mismatch: broadcasting supports 1->N but not "
                f"arbitrary mismatches. Lengths: {[len(ds) for ds in datasets]}."
            )
            raise ValueError(msg)
        target = lengths.pop()
        return [ActivationDataset(ds.batches * target) if len(ds) == 1 else ds for ds in datasets]


class ActivationRegister:
    """Stores per-node, per-stream activation data for the instruction engine."""

    def __init__(self) -> None:
        self._data: dict[_BaseRef, dict[StreamKey, ActivationDataset]] = {}

    def store(self, ref: _BaseRef, context: StreamKey, data: ActivationDataset) -> None:
        """Store activation data for a ref under a specific context."""
        if ref not in self._data:
            self._data[ref] = {}
        self._data[ref][context] = data

    def store_all(self, ref: _BaseRef, mapping: dict[StreamKey, ActivationDataset]) -> None:
        """Merge activation data for a ref, so multiple producers accumulate streams."""
        self._data.setdefault(ref, {}).update(mapping)

    def load(self, ref: _BaseRef, context: StreamKey) -> ActivationDataset:
        """Load activation data for a ref under a specific context."""
        return self._data[ref][context]

    def items_for(self, ref: _BaseRef) -> Iterator[tuple[StreamKey, ActivationDataset]]:
        """Iterate over (context, dataset) pairs for a ref."""
        return iter(self._data[ref].items())

    def contexts_for(self, ref: _BaseRef) -> Iterator[StreamKey]:
        """Iterate over contexts that have data stored for a ref."""
        return iter(self._data[ref].keys())

    def delete(self, ref: _BaseRef, context: StreamKey | None = None) -> None:
        """Remove data for `ref`.

        With `context` omitted, drops every stream stored for `ref`. With
        `context` given, drops only that stream; if it was the last, `ref`
        itself is dropped so `ref in register` becomes False.
        """
        if context is None:
            self._data.pop(ref, None)
        elif ref in self._data:
            self._data[ref].pop(context, None)
            if not self._data[ref]:
                self._data.pop(ref)

    def __contains__(self, ref: _BaseRef) -> bool:
        return ref in self._data


@dataclasses.dataclass(frozen=True)
class ActivationBundle:
    """Per-context view of all activations needed to reproduce one call.

    An `ActivationBundle` packages the positional and keyword `ActivationDataset`s
    that together describe a single node's input under one execution context.
    Iteration yields `(args_tuple, kwargs_dict)` per batch — the exact shape needed
    to invoke the underlying callable as `module(*args, **kwargs)`.

    Constructed via `gather`, which resolves a node's `_BaseRef` args and kwargs
    against the register, broadcasts singletons against N-length streams, and
    bundles them. `Const` refs are wrapped on the fly.
    """

    args: tuple[ActivationDataset, ...]
    kwargs: Mapping[str, ActivationDataset]

    @classmethod
    def gather(
        cls,
        register: ActivationRegister,
        context: StreamKey,
        args: Sequence[_BaseRef],
        kwargs: Mapping[str, _BaseRef],
    ) -> "ActivationBundle":
        """Resolve refs from the register under one context key, broadcast, and bundle.

        Args:
            register: The activation register mapping refs to per-context datasets.
            context: The context key under which to resolve each ref.
            args: Positional input refs in declaration order.
            kwargs: Keyword input refs keyed by parameter name.

        Returns:
            An `ActivationBundle` whose positional and keyword streams are broadcast
            to a common batch length and ready for iteration as `(args, kwargs)` pairs.
        """

        def get_dataset(ref: _BaseRef) -> ActivationDataset:
            if isinstance(ref, Const):
                return ActivationDataset([ref.value])
            return register.load(ref, context)

        arg_datasets = [get_dataset(ref) for ref in args]
        kwarg_datasets = {key: get_dataset(ref) for key, ref in kwargs.items()}

        broadcasted = ActivationDataset.broadcast([*arg_datasets, *kwarg_datasets.values()])
        n_args = len(arg_datasets)
        return cls(
            args=tuple(broadcasted[:n_args]),
            kwargs=dict(zip(kwarg_datasets.keys(), broadcasted[n_args:])),
        )

    def __len__(self) -> int:
        """Return the number of batches in this bundle.

        Returns:
            The length of the first positional or keyword stream, or 0 if empty.
        """
        for ds in self.args:
            return len(ds)
        for ds in self.kwargs.values():
            return len(ds)
        return 0

    def __iter__(self) -> Iterator[tuple[tuple[Any, ...], dict[str, Any]]]:
        """Iterate over batches, yielding one `(args, kwargs)` pair per batch.

        Each yielded tuple contains the positional activations as a tuple and
        the keyword activations as a dict, aligned at the same batch index across
        all streams. The result can be passed directly to a module as
        ``module(*args, **kwargs)``.

        Yields:
            Tuples of ``(args_tuple, kwargs_dict)`` for each batch index.
        """
        length = len(self)
        for i in range(length):
            args = tuple(ds.batches[i] for ds in self.args)
            kwargs = {name: ds.batches[i] for name, ds in self.kwargs.items()}
            yield args, kwargs


@dataclasses.dataclass(frozen=True)
class Instruction(abc.ABC):
    """Base class for all instructions in the execution engine.

    Each instruction must implement the execute method to define its behavior
    when executed by the InstructionEngine.
    """

    @abc.abstractmethod
    def execute(self, register: ActivationRegister) -> Any:
        """Execute this instruction.

        Args:
            register: The execution register mapping references to values.

        Returns:
            The result of executing this instruction.
        """

    def uses(self) -> Iterator[_BaseRef]:
        """Return references this instruction uses from the register."""
        return iter(())

    def produces(self) -> Iterator[_BaseRef]:
        """Return references this instruction writes to the register."""
        return iter(())


@dataclasses.dataclass(frozen=True)
class StoreValue(Instruction):
    """Store a value in the register."""

    target: _BaseRef
    value: Any
    contexts: Contexts

    def __repr__(self) -> str:
        return (
            f"StoreValue(target={self.target!r}, value={_fmt_value(self.value)}, "
            f"contexts={_fmt_contexts(self.contexts)})"
        )

    def execute(self, register: ActivationRegister) -> None:  # noqa: D102
        for context in self.contexts:
            register.store(self.target, context, self.value)

    def uses(self) -> Iterator[_BaseRef]:  # noqa: D102
        return iter([self.target])


@dataclasses.dataclass(frozen=True)
class LoadAttribute(Instruction):
    """Load an attribute/index from a register value and store result."""

    source: _BaseRef
    target: _BaseRef
    attribute: str | int

    def execute(self, register: ActivationRegister) -> None:  # noqa: D102
        for context, dataset in register.items_for(self.source):
            register.store(self.target, context, self._extract_attribute(dataset))

    def _extract_attribute(self, dataset: ActivationDataset) -> ActivationDataset:
        """Extract attribute/item from each batch in the dataset."""
        match self.attribute:
            case int() as idx:
                batches = [batch[idx] for batch in dataset]
            case str() as key:
                # Item access for mappings (dict, etc.)
                try:
                    batches = [batch[key] for batch in dataset]
                except (KeyError, TypeError):
                    # Attribute access for structured objects (dataclass, namedtuple, etc.)
                    batches = [getattr(batch, key) for batch in dataset]

        return ActivationDataset(batches)

    def uses(self) -> Iterator[_BaseRef]:  # noqa: D102
        return iter([self.source])

    def produces(self) -> Iterator[_BaseRef]:  # noqa: D102
        return iter([self.target])


@dataclasses.dataclass(frozen=True)
class Call(Instruction):
    """Execute one graph node on batched data from the register.

    Handles the per-context loop, bundle gathering, and result storage.

    Args:
        args: Positional input refs in declaration order.
        kwargs: Keyword input refs keyed by parameter name.
        target: Ref the outputs are stored under.
        contexts: Execution conditions to run under, one pass each.
        caller: Callable invoked for each batch.
        cache: Whether the outputs may be reused by a later reader.
        input_location: Where to place each batch's arguments before invocation.
        output_location: Where to place each result before collecting the next batch.
    """

    args: Sequence[_BaseRef]
    kwargs: dict[str, _BaseRef]
    target: _BaseRef
    contexts: Contexts
    caller: Callable[..., Any]
    cache: bool = dataclasses.field(default=False, kw_only=True)
    input_location: Location | None = dataclasses.field(default=None, kw_only=True)
    output_location: Location | None = dataclasses.field(default=None, kw_only=True)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(caller={_fmt_module(self.caller)}, args={list(self.args)!r}, "
            + f"kwargs={dict(self.kwargs)!r}, target={self.target!r}, "
            + f"contexts={_fmt_contexts(self.contexts)}, cache={self.cache}, "
            + f"input_location={self.input_location!r}, output_location={self.output_location!r})"
        )

    def _batches(
        self, bundle: ActivationBundle
    ) -> Iterator[tuple[tuple[Any, ...], dict[str, Any]]]:
        batches: Iterable[tuple[tuple[Any, ...], dict[str, Any]]] = bundle
        if not bundle:
            batches = [((), {})]
        for args, kwargs in batches:
            if self.input_location is not None:
                args, kwargs = self.input_location.place((args, kwargs))
            yield args, kwargs
            del args, kwargs

    def execute(self, register: ActivationRegister) -> None:  # noqa: D102
        for context in self.contexts:
            bundle = ActivationBundle.gather(register, context, self.args, self.kwargs)
            outputs = []
            module = self.caller if isinstance(self, CallModule) else None
            with context(module):
                for args, kwargs in self._batches(bundle):
                    output = self.caller(*args, **kwargs)
                    if self.output_location is not None:
                        output = self.output_location.place(output)
                    outputs.append(output)
                    del args, kwargs, output
            register.store(self.target, context, ActivationDataset(outputs))

    def uses(self) -> Iterator[_BaseRef]:  # noqa: D102
        yield from self.args
        yield from self.kwargs.values()

    def produces(self) -> Iterator[_BaseRef]:  # noqa: D102
        return iter([self.target])


@dataclasses.dataclass(frozen=True, repr=False)
class CallModule(Call):
    """Call a module on batched data, entering the context around each invocation."""

    caller: torch.nn.Module


@dataclasses.dataclass(frozen=True, repr=False)
class CallFunction(Call):
    """Call a free function on batched data."""


@dataclasses.dataclass(frozen=True, repr=False)
class CallMethod(Call):
    """Call a bound method on batched data."""


@dataclasses.dataclass(frozen=True)
class BundleSpec:
    """One bundle handed to a delegate: which refs to gather, under which context.

    A data-flow declaration names both halves, and they vary together: one flow
    may want the region's inputs under an unquantized context while the next wants
    the region's output under a quantized one. Pairing them here keeps a ref from
    being gathered under a context that never produced it.

    Args:
        context: The context to resolve the refs under.
        args: Positional refs to gather.
        kwargs: Keyword refs to gather.
    """

    context: StreamKey
    args: Sequence[_BaseRef]
    kwargs: Mapping[str, _BaseRef] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class RunDelegate(Instruction):
    """Run a function once on a set of ordered data flows and optionally store the result.

    The function `fn` can be any callable that expects an optional module as first argument
    and then a sequence of data flows (in order) that are converted by the scheduler
    to `bundles`. This instruction can be used to optimize the module with an algorithm,
    calculate statistics, etc.

    If you want to store data to the register, make sure the function outputs
    exactly as much items as there are `targets`.

    Args:
        fn: The function to call.
        bundles: The bundles to gather for `fn`.
        targets: The ref and context that each returned dataset is stored under.
        module: The module whose weights must be on the compute location during the
            call, if any.
    """

    fn: Callable[..., Sequence[ActivationDataset] | None]
    bundles: Sequence[BundleSpec]
    targets: Sequence[tuple[_BaseRef, StreamKey]] = ()
    module: torch.nn.Module | None = None

    def __repr__(self) -> str:
        args = [ref for bundle in self.bundles for ref in bundle.args]
        kwargs = {key: ref for bundle in self.bundles for key, ref in bundle.kwargs.items()}
        contexts = [bundle.context for bundle in self.bundles]
        module = None if self.module is None else _fmt_module(self.module)
        targets = [f"{ref!r}@{_fmt_callable(context)}" for ref, context in self.targets]
        return (
            f"CallDelegate(fn={_fmt_callable(self.fn)}, module={module}, args={args!r}, "
            f"kwargs={kwargs!r}, contexts={_fmt_contexts(contexts)}, "
            f"targets=[{', '.join(targets)}])"
        )

    def execute(self, register: ActivationRegister) -> None:  # noqa: D102
        fn_inputs: list[torch.nn.Module | ActivationBundle] = [
            ActivationBundle.gather(register, spec.context, spec.args, spec.kwargs)
            for spec in self.bundles
        ]
        if self.module is not None:
            fn_inputs = [self.module, *fn_inputs]

        results = self.fn(*fn_inputs)
        datasets = () if results is None else results
        if len(datasets) != len(self.targets):
            msg = (
                f"Delegate {_fmt_callable(self.fn)} returned {len(datasets)} datasets for "
                f"{len(self.targets)} targets."
            )
            raise ValueError(msg)

        for (ref, context), dataset in zip(self.targets, datasets):
            register.store(ref, context, dataset)

    def uses(self) -> Iterator[_BaseRef]:  # noqa: D102
        for spec in self.bundles:
            yield from spec.args
            yield from spec.kwargs.values()

    def produces(self) -> Iterator[_BaseRef]:  # noqa: D102
        for ref, _ in self.targets:
            yield ref


@dataclasses.dataclass(frozen=True)
class ReturnOutputs(Instruction):
    """Return output values from the register.

    Returns a dict mapping each execution context to its output values.
    Output values are automatically unpacked based on batch/output count.
    """

    outputs: Sequence[_BaseRef]

    def execute(self, register: ActivationRegister) -> Any:  # noqa: D102
        # Invert register[output_ref][context] to contexts[context] = [ds1, ds2, ...],
        # and use this to create a single ActivationDataset per context.
        context_outputs: dict[StreamKey, list[ActivationDataset]] = defaultdict(list)
        for output_ref in self.outputs:
            for context, dataset in register.items_for(output_ref):
                context_outputs[context].append(dataset)

        context_datasets = {
            context: tuple(ActivationDataset.merge(datasets).batches)
            for context, datasets in context_outputs.items()
        }

        return context_datasets

    def uses(self) -> Iterator[_BaseRef]:  # noqa: D102
        return iter(self.outputs)


@dataclasses.dataclass(frozen=True)
class DeleteRegisterEntries(Instruction):
    """Delete specified register entries to free memory."""

    targets: Sequence[_BaseRef]

    def execute(self, register: ActivationRegister) -> None:  # noqa: D102
        for target_id in self.targets:
            register.delete(target_id)


@dataclasses.dataclass(frozen=True)
class MoveModule(Instruction):
    """Move module parameters and buffers to a target location.

    Args:
        location: Target location for all parameters and buffers, or a mapping from name
            to location. When a mapping is provided, each named parameter/buffer is moved
            to its corresponding location.
        module: Module whose parameters and buffers will be moved.
    """

    location: Location | Mapping[str, Location]
    module: torch.nn.Module

    def __repr__(self) -> str:
        return f"MoveModule(location={self.location!r}, module={_fmt_module(self.module)})"

    def execute(self, register: ActivationRegister) -> None:  # noqa: D102, ARG002
        if isinstance(self.location, Mapping):
            for name, parameter in self.module.named_parameters():
                if (location := self.location.get(name)) is not None:
                    parameter.data = location.receive(parameter.data)
            for name, buffer in self.module.named_buffers():
                if (location := self.location.get(name)) is not None:
                    buffer.data = location.receive(buffer.data)
        else:
            for parameter in self.module.parameters():
                parameter.data = self.location.receive(parameter.data)
            for buffer in self.module.buffers():
                buffer.data = self.location.receive(buffer.data)


Instructions: TypeAlias = Sequence[Instruction]
InstructionPass: TypeAlias = Callable[[Instructions], Instructions]


@dataclasses.dataclass(frozen=True)
class InstructionProgram:
    """A scheduled program consisting of instructions and input metadata.

    Args:
        instructions: Sequence of instructions to execute.
        input_refs: Mapping from input names to InputRef objects.
    """

    instructions: Instructions
    input_refs: dict[str, InputRef]

    @property
    def contexts(self) -> Contexts:
        """All contexts used in program."""
        all_contexts: set[StreamKey] = set()

        for instruction in self.instructions:
            match instruction:
                case Call(contexts=contexts) | StoreValue(contexts=contexts):
                    all_contexts.update(contexts)
                case RunDelegate(bundles=bundles, targets=targets):
                    all_contexts.update(spec.context for spec in bundles)
                    all_contexts.update(context for _, context in targets)
                case _:
                    pass

        return list(all_contexts)


@dataclasses.dataclass(frozen=True)
class InstructionEngine:
    """Executes pre-scheduled instruction sequences.

    This engine runs instructions in order using a register-based approach
    to track intermediate values and activations. Instructions are generated
    by the scheduler during the scheduling phase.
    """

    @staticmethod
    def prepare_input_register(
        input_refs: dict[str, InputRef],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        contexts: Contexts,
    ) -> ActivationRegister:
        """Prepare execution register from user inputs.

        Args:
            input_refs: Mapping from input names to InputRef objects.
            args: Positional arguments from user.
            kwargs: Keyword arguments from user.
            contexts: Each input will be replicated across all provided contexts.

        Returns:
            register mapping InputRefs to ActivationDataset values.

        Raises:
            TypeError: If arguments don't match graph inputs.
        """
        input_names = list(input_refs.keys())

        if len(args) > len(input_names):
            msg = f"Expected {len(input_names)} positional arguments, got {len(args)}"
            raise TypeError(msg)

        inputs: dict[str, Any] = dict(zip(input_names[: len(args)], args))

        for name, value in kwargs.items():
            if name not in input_refs:
                msg = f"Unexpected keyword argument: '{name}'"
                raise TypeError(msg)
            if name in inputs:
                msg = f"Multiple values for argument: '{name}'"
                raise TypeError(msg)
            inputs[name] = value

        if missing := set(input_names) - inputs.keys():
            msg = f"Missing required inputs: {sorted(missing)}"
            raise TypeError(msg)

        register = ActivationRegister()
        for input_name, value in inputs.items():
            ref = input_refs[input_name]
            dataset = ActivationDataset.from_value(value)
            for context in contexts:
                register.store(ref, context, dataset)
        return register

    @staticmethod
    def run_instructions(instructions: Instructions, register: ActivationRegister) -> Any:
        """Run a sequence of instructions with the given register.

        Args:
            instructions: Sequence of instructions to execute.
            register: The execution register.

        Returns:
            Result from RETURN instruction, or None if no RETURN is executed.
        """
        for instruction in instructions:
            if (result := instruction.execute(register)) is not None:
                return result

    def run(self, program: InstructionProgram, *args: Any, **kwargs: Any) -> Any:
        """Run instructions with provided inputs.

        Args:
            program: InstructionProgram containing instructions and input metadata.
            *args: Positional inputs for graph.
            **kwargs: Keyword inputs for graph.

        Returns:
            Result from instruction execution.
        """
        register = self.prepare_input_register(program.input_refs, args, kwargs, program.contexts)
        return self.run_instructions(program.instructions, register)


class InstructionPasses:
    """Applies a sequence of instruction passes to a program.

    Each pass is a pure function transforming an instruction sequence into a new one.
    Passes are applied in order.
    """

    @staticmethod
    def apply(
        program: InstructionProgram, passes: Sequence[InstructionPass] | None = None
    ) -> InstructionProgram:
        """Apply the given passes to the program's instruction sequence.

        Args:
            program: The instruction program to transform.
            passes: Passes to apply in order. If None, the program is returned unchanged.

        Returns:
            A new program with the transformed instruction sequence.
        """
        instructions = program.instructions
        for pass_fn in passes or []:
            instructions = pass_fn(instructions)
        return dataclasses.replace(program, instructions=instructions)


def lifetime_management_pass(instructions: Instructions) -> Instructions:
    """Insert DeleteRegisterEntries instructions to free memory when values are no longer needed.

    Args:
        instructions: Sequence of instructions to analyze.

    Returns:
        New instruction sequence with DeleteRegisterEntries instructions inserted.
    """
    keep_alive: set[_BaseRef] = set()
    for instruction in instructions:
        if isinstance(instruction, ReturnOutputs):
            keep_alive.update(instruction.uses())

    last_use = {}

    # Iterate in reverse to find the last instruction that depends on each register slot
    for idx in range(len(instructions) - 1, -1, -1):
        instruction = instructions[idx]
        for uuid_id in itertools.chain(instruction.uses(), instruction.produces()):
            if uuid_id not in last_use:
                last_use[uuid_id] = idx

    new_instructions: list[Instruction] = []
    for idx, instruction in enumerate(instructions):
        new_instructions.append(instruction)

        # Delete any register entry that has no dependent instructions after this point
        to_delete = [
            uuid_id
            for uuid_id, last_idx in last_use.items()
            if last_idx == idx and uuid_id not in keep_alive
        ]

        if to_delete:
            new_instructions.append(DeleteRegisterEntries(targets=to_delete))

    return tuple(new_instructions)


def _weight_offloading_pass(
    instructions: Instructions,
    compute: Location,
    storage: Location,
    graph: GraphModule,
) -> Instructions:
    """Insert `MoveModule` instructions to move module weights between locations.

    First, record the original device placement of all weights so the model can be restored
    to its initial state after the pass (`post_restore`). Next, move each weight to `storage`
    to perform the actual offload. Finally, wrap each `CallModule`/`OptimizeModule` with the appropriate
    placement: `compute` for execution and `storage` for storage.

    If we have a instruction stream that goes through two linear layers L1 -> L2, the pass would add
    offload(L1), Offload(L2), Load(L1), Call(L1), Offload(L1), Load(L2), Call(L2), offload(L2), Load(L1), Load(L2).

    NB: We need access to `GraphModule` because during optimization not all parameters have
    to be present in the instruction stream even if they are still possibly on `compute`.

    Args:
        instructions: Sequence of instructions to analyze.
        compute: Where `CallModule`/`OptimizeModule` execution happens.
        storage: Where we 'offload' to.
        graph: Original GraphModule — all node modules are pre- and post-offloaded.

    Returns:
        New instruction sequence with `MoveModule` instructions inserted.
    """
    all_modules = list(
        dict.fromkeys(
            node.target
            for node in graph._nodes.values()
            if node.op is Op.torch_module and isinstance(node.target, torch.nn.Module)
        )
    )

    # Ensure `post_restore` maps each parameter back to its individual original location.
    original_locations: dict[torch.nn.Module, dict[str, Location]] = {}
    for m in all_modules:
        locations: dict[str, Location] = {
            name: DeviceLocation(param.device) for name, param in m.named_parameters()
        }
        locations.update({name: DeviceLocation(buf.device) for name, buf in m.named_buffers()})
        if locations:
            original_locations[m] = locations

    post_restore = [
        MoveModule(location=locations, module=m) for m, locations in original_locations.items()
    ]

    pre_offload = [MoveModule(location=storage, module=m) for m in all_modules]

    new_instructions: list[Instruction] = [*pre_offload]

    for instruction in instructions:
        match instruction:
            case CallModule(caller=module) | RunDelegate(module=module) if isinstance(
                module, torch.nn.Module
            ):
                new_instructions.append(MoveModule(location=compute, module=module))
                new_instructions.append(instruction)
                new_instructions.append(MoveModule(location=storage, module=module))
            case _:
                new_instructions.append(instruction)

    new_instructions.extend(post_restore)
    return tuple(new_instructions)


def _activation_offloading_pass(
    instructions: Instructions, compute: Location, storage: Location
) -> Instructions:
    """Configure input and output locations on each `Call`.

    Before each invocation, moves input activations to `compute`. After each
    invocation, moves the output activation to `storage`, one batch at a time.
    Input datasets remain in the register at their existing locations.

    Args:
        instructions: Sequence of instructions to analyze.
        compute: Where activations are moved to before execution.
        storage: Where activations are moved to after execution.

    Returns:
        Instructions with per-batch placement configured on calls.
    """
    new_instructions: list[Instruction] = []

    for instruction in instructions:
        if isinstance(instruction, Call):
            new_instructions.append(
                dataclasses.replace(instruction, input_location=compute, output_location=storage)
            )
        else:
            new_instructions.append(instruction)

    return tuple(new_instructions)


def _offloading_pass(
    instructions: Instructions,
    compute: Location,
    storage: Location,
    graph: GraphModule,
) -> Instructions:
    """Configure module placement and per-batch activation offloading.

    Composes `_weight_offloading_pass` and `_activation_offloading_pass` to handle both
    module weight movement and per-batch activation movement.

    Args:
        instructions: Original instruction sequence.
        compute: Where data is moved to before execution.
        storage: Where data is moved to after execution.
        graph: Original GraphModule — all node modules are pre- and post-offloaded.

    Returns:
        New instruction sequence with module and activation placement configured.
    """
    instructions = _weight_offloading_pass(instructions, compute, storage, graph)
    instructions = _activation_offloading_pass(instructions, compute, storage)

    # Cancel out compute(M1) -> storage(M1) -> compute(M1) placements for any module M1.
    instructions = _cancel_module_round_trips(instructions, compute, storage)
    return instructions


def _cancel_module_round_trips(
    instructions: Instructions, compute: Location, storage: Location
) -> Instructions:
    """Cancel adjacent MoveModule round-trips."""
    location_pair = {compute, storage}

    def _is_round_trip(left: Instruction, right: Instruction) -> bool:
        return (
            isinstance(left, MoveModule)
            and isinstance(right, MoveModule)
            and left.module is right.module
            and isinstance(left.location, Location)
            and isinstance(right.location, Location)
            and {left.location, right.location} == location_pair
        )

    result: list[Instruction] = []
    i = 0
    while i < len(instructions):
        if i + 1 < len(instructions) and _is_round_trip(instructions[i], instructions[i + 1]):
            i += 2
        else:
            result.append(instructions[i])
            i += 1

    return tuple(result)


class OffloadingStrategy(abc.ABC):
    """Abstract base for offloading strategies.

    An offloading strategy controls how module weights and activations are moved
    between locations during graph execution. Implement `create_instruction_pass` to
    configure module and activation placement.
    """

    @abc.abstractmethod
    def create_instruction_pass(self, graph: GraphModule) -> InstructionPass:
        """Return an instruction pass that configures placement.

        Args:
            graph: The GraphModule being scheduled.

        Returns:
            An `InstructionPass` that configures placement in the instruction sequence.
        """


@dataclasses.dataclass(frozen=True)
class OffloadEverything(OffloadingStrategy):
    """Offload all module weights and activations between a compute device and a storage device.

    Moves every module's weights to `storage_device` when idle, and back to
    `compute_device` just before execution. Activations are loaded and offloaded
    one batch at a time.

    Args:
        compute_device: Device where computation happens (e.g. `cuda`).
        storage_device: Device where idle data is stored (e.g. `cpu`).
    """

    compute_device: torch.device
    storage_device: torch.device = dataclasses.field(default_factory=lambda: torch.device("cpu"))

    def create_instruction_pass(self, graph: GraphModule) -> InstructionPass:  # noqa: D102
        compute = DeviceLocation(self.compute_device)
        storage = DeviceLocation(self.storage_device)

        def _pass(instructions: Instructions) -> Instructions:
            return _offloading_pass(instructions, compute, storage, graph)

        return _pass
