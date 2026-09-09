# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


import abc
import dataclasses
import itertools

from collections import defaultdict
from typing import (
    Any,
    Callable,
    Collection,
    ContextManager,
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

# Distinguishes data produced under different execution conditions for the same node.
StreamKey: TypeAlias = Callable[[torch.nn.Module], ContextManager[None]]

# Ordered sequence of context managers that an instruction executes under.
Contexts: TypeAlias = Sequence[Callable[[torch.nn.Module], ContextManager[None]]]


def _fmt_module(module: torch.nn.Module | Callable[..., Any]) -> str:
    """Format a module/callable target without dumping parameters or a memory address."""
    if isinstance(module, torch.nn.Module):
        return type(module).__name__
    return _fmt_callable(module)


def _fmt_callable(fn: Callable[..., Any]) -> str:
    """Format a callable by name, without its memory address."""
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
class Call(Instruction, abc.ABC):
    """Execute one graph node on batched data from the register.

    Handles the per-context loop, bundle gathering, and result storage. A
    subclass adds one field for the callable it invokes and implements `_call`.

    Args:
        args: Positional input refs in declaration order.
        kwargs: Keyword input refs keyed by parameter name.
        target: Ref the outputs are stored under.
        contexts: Execution conditions to run under, one pass each.
        cache: Whether the outputs may be reused by a later reader.
    """

    args: Sequence[_BaseRef]
    kwargs: dict[str, _BaseRef]
    target: _BaseRef
    contexts: Contexts
    cache: bool = dataclasses.field(default=False, kw_only=True)

    @abc.abstractmethod
    def _call(
        self,
        context: Callable[[torch.nn.Module], ContextManager[None]],
        bundle: ActivationBundle,
    ) -> list[Any]:
        """Invoke the callable under the given context."""

    def execute(self, register: ActivationRegister) -> None:  # noqa: D102
        for context in self.contexts:
            bundle = ActivationBundle.gather(register, context, self.args, self.kwargs)
            outputs = self._call(context, bundle)
            register.store(self.target, context, ActivationDataset(outputs))

    def uses(self) -> Iterator[_BaseRef]:  # noqa: D102
        yield from self.args
        yield from self.kwargs.values()

    def produces(self) -> Iterator[_BaseRef]:  # noqa: D102
        return iter([self.target])


@dataclasses.dataclass(frozen=True)
class CallModule(Call):
    """Call a module on batched data, entering the context around each invocation."""

    module: torch.nn.Module

    def __repr__(self) -> str:
        return (
            f"CallModule(module={_fmt_module(self.module)}, args={list(self.args)!r}, "
            f"kwargs={dict(self.kwargs)!r}, target={self.target!r}, "
            f"contexts={_fmt_contexts(self.contexts)}, cache={self.cache})"
        )

    def _call(self, context: StreamKey, bundle: ActivationBundle) -> list[Any]:  # noqa: D102
        with context(self.module):
            if not bundle:
                return [self.module()]
            return [self.module(*args, **kwargs) for args, kwargs in bundle]


@dataclasses.dataclass(frozen=True)
class CallFunction(Call):
    """Call a free function on batched data."""

    fn: Callable[..., Any]

    def __repr__(self) -> str:
        return (
            f"CallFunction(fn={_fmt_callable(self.fn)}, args={list(self.args)!r}, "
            f"kwargs={dict(self.kwargs)!r}, target={self.target!r}, "
            f"contexts={_fmt_contexts(self.contexts)}, cache={self.cache})"
        )

    def _call(self, _context: StreamKey, bundle: ActivationBundle) -> list[Any]:  # noqa: D102
        if not bundle:
            return [self.fn()]
        return [self.fn(*args, **kwargs) for args, kwargs in bundle]


@dataclasses.dataclass(frozen=True)
class CallMethod(Call):
    """Call a bound method on batched data."""

    method: Callable[..., Any]

    def __repr__(self) -> str:
        return (
            f"CallMethod(method={_fmt_callable(self.method)}, args={list(self.args)!r}, "
            f"kwargs={dict(self.kwargs)!r}, target={self.target!r}, "
            f"contexts={_fmt_contexts(self.contexts)}, cache={self.cache})"
        )

    def _call(self, _context: StreamKey, bundle: ActivationBundle) -> list[Any]:  # noqa: D102
        if not bundle:
            return [self.method()]
        return [self.method(*args, **kwargs) for args, kwargs in bundle]


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
class OptimizeModule(Instruction):
    """Optimize a module in-place using batched data from the register.

    Invokes `fn(self.module, *bundles)`, one `ActivationBundle` per declared data
    flow, in declaration order. Each bundle carries the data that its flow asked
    for, gathered under the context that flow named.
    """

    module: torch.nn.Module
    fn: Callable[..., None]
    bundles: Sequence[BundleSpec]

    def __repr__(self) -> str:
        args = [ref for bundle in self.bundles for ref in bundle.args]
        kwargs = {key: ref for bundle in self.bundles for key, ref in bundle.kwargs.items()}
        contexts = [bundle.context for bundle in self.bundles]
        return (
            f"OptimizeModule(module={_fmt_module(self.module)}, args={args!r}, "
            f"kwargs={kwargs!r}, fn={_fmt_callable(self.fn)}, "
            f"contexts={_fmt_contexts(contexts)})"
        )

    def execute(self, register: ActivationRegister) -> None:  # noqa: D102
        bundles = [
            ActivationBundle.gather(register, spec.context, spec.args, spec.kwargs)
            for spec in self.bundles
        ]
        self.fn(self.module, *bundles)

    def uses(self) -> Iterator[_BaseRef]:  # noqa: D102
        for spec in self.bundles:
            yield from spec.args
            yield from spec.kwargs.values()


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
    """Move module parameters and buffers to a target device.

    Args:
        device: Target device for all parameters and buffers, or a mapping from name to device.
            When a mapping is provided, each named parameter/buffer is moved to its corresponding
            device.
        module: Module whose parameters and buffers will be moved.
    """

    device: torch.device | dict[str, torch.device]
    module: torch.nn.Module

    def __repr__(self) -> str:
        return f"MoveModule(device={self.device!r}, module={_fmt_module(self.module)})"

    def execute(self, register: ActivationRegister) -> None:  # noqa: D102, ARG002
        if isinstance(self.device, dict):
            for name, parameter in self.module.named_parameters():
                if name in self.device:
                    parameter.data = parameter.data.to(device=self.device[name])
            for name, buffer in self.module.named_buffers():
                if name in self.device:
                    buffer.data = buffer.data.to(device=self.device[name])
        else:
            for parameter in self.module.parameters():
                parameter.data = parameter.data.to(device=self.device)
            for buffer in self.module.buffers():
                buffer.data = buffer.data.to(device=self.device)


def _move_to_device(value: Any, device: torch.device) -> Any:
    """Recursively move tensors in nested structures to `device`.

    Args:
        value: A tensor, tuple, list, dict, or other value.
        device: Target device.

    Returns:
        The value with all tensors moved to `device`. Non-tensor leaves are returned as-is.
    """
    match value:
        case torch.Tensor():
            return value.to(device=device)
        case tuple():
            return tuple(_move_to_device(v, device) for v in value)
        case list():
            return [_move_to_device(v, device) for v in value]
        case dict() if type(value) is not dict:
            # Try to preserve true dict subclass types (e.g. HF outputs).
            moved = {k: _move_to_device(v, device) for k, v in value.items()}
            try:
                return type(value)(moved)
            except TypeError:
                pass
            return moved
        case dict():
            return {k: _move_to_device(v, device) for k, v in value.items()}
    return value


def _move_register_entries_to_device(
    register: ActivationRegister, ref: _BaseRef, device: torch.device
) -> None:
    """Move register entry for `ref` to `device` in-place.

    Args:
        register: The activation register.
        ref: Reference whose entry should be moved.
        device: Target device.
    """
    for context, dataset in register.items_for(ref):
        moved = dataclasses.replace(
            dataset, batches=[_move_to_device(batch, device) for batch in dataset.batches]
        )
        register.store(ref, context, moved)


@dataclasses.dataclass(frozen=True)
class MoveActivations(Instruction):
    """Move a single activation register entry to a target device.

    Args:
        device: Target device for the move.
        register_ref: Reference whose register entry will be moved.
    """

    device: torch.device
    register_ref: _BaseRef

    def execute(self, register: ActivationRegister) -> None:  # noqa: D102
        _move_register_entries_to_device(register, self.register_ref, self.device)

    def uses(self) -> Iterator[_BaseRef]:  # noqa: D102
        yield self.register_ref


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
                case OptimizeModule(bundles=bundles):
                    all_contexts.update(spec.context for spec in bundles)
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
    compute_device: torch.device,
    storage_device: torch.device,
    graph: GraphModule,
) -> Instructions:
    """Insert `MoveModule` instructions to move module weights between devices.

    First, record the original device placement of all weights so the model can be restored
    to its initial state after the pass (`post_restore`). Next, move each weight to `storage_device`
    to perform the actual offload. Finally, wrap each `CallModule`/`OptimizeModule` with the appropriate
    device placement: `compute_device` for execution and `storage_device` for storage.

    If we have a instruction stream that goes through two linear layers L1 -> L2, the pass would add
    offload(L1), Offload(L2), Load(L1), Call(L1), Offload(L1), Load(L2), Call(L2), offload(L2), Load(L1), Load(L2).

    NB: We need access to `GraphModule` because during optimization not all parameters have
    to be present in the instruction stream even if they are still possibly on `compute_device`.

    Args:
        instructions: Sequence of instructions to analyze.
        compute_device: Compute device, where `CallModule`/`OptimizeModule` execution happens.
        storage_device: Storage device, where we 'offload' to.
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

    # Ensure `post_restore` maps each parameter back to its individual original device.
    original_devices: dict[torch.nn.Module, dict[str, torch.device]] = {}
    for m in all_modules:
        param_devices = {name: param.device for name, param in m.named_parameters()}
        param_devices.update({name: buf.device for name, buf in m.named_buffers()})
        if param_devices:
            original_devices[m] = param_devices

    post_restore = [
        MoveModule(device=param_devices, module=m) for m, param_devices in original_devices.items()
    ]

    pre_offload = [MoveModule(device=storage_device, module=m) for m in all_modules]

    new_instructions: list[Instruction] = [*pre_offload]

    for instruction in instructions:
        match instruction:
            case CallModule(module=module) | OptimizeModule(module=module) if isinstance(
                module, torch.nn.Module
            ):
                new_instructions.append(MoveModule(device=compute_device, module=module))
                new_instructions.append(instruction)
                new_instructions.append(MoveModule(device=storage_device, module=module))
            case _:
                new_instructions.append(instruction)

    new_instructions.extend(post_restore)
    return tuple(new_instructions)


def _activation_offloading_pass(
    instructions: Instructions, compute_device: torch.device, storage_device: torch.device
) -> Instructions:
    """Insert `MoveActivations` instructions to move register entries between devices.

    Before each `CallModule`/`OptimizeModule`, moves input activations to `compute_device`.
    After each `CallModule`, moves the output activation to `storage_device`.

    If we have an instruction stream that goes through two linear layers L1 -> L2, the pass would add
    MoveAct(in, compute), Call(L1), MoveAct(out1, storage), MoveAct(out1, compute), Call(L2), MoveAct(out2, storage).

    Args:
        instructions: Sequence of instructions to analyze.
        compute_device: Device to move activations to before execution.
        storage_device: Device to move activations to after execution.

    Returns:
        New instruction sequence with `MoveActivations` instructions inserted.
    """
    new_instructions: list[Instruction] = []

    for instruction in instructions:
        if isinstance(instruction, (CallModule, OptimizeModule)):
            for ref in instruction.uses():
                if isinstance(ref.unwrap_ref(), Const):
                    continue
                new_instructions.append(MoveActivations(device=compute_device, register_ref=ref))

            new_instructions.append(instruction)

            # Only CallModule produces an ActivationDataset.
            if isinstance(instruction, CallModule):
                new_instructions.append(
                    MoveActivations(device=storage_device, register_ref=instruction.target)
                )
        else:
            new_instructions.append(instruction)

    return tuple(new_instructions)


def _offloading_pass(
    instructions: Instructions,
    compute_device: torch.device,
    storage_device: torch.device,
    graph: GraphModule,
) -> Instructions:
    """Insert device placement instructions around `CallModule` and `OptimizeModule`.

    Composes `_weight_offloading_pass` and `_activation_offloading_pass` to handle both
    module weight movement and activation register entry movement.

    Args:
        instructions: Original instruction sequence.
        compute_device: Device to move data to before execution.
        storage_device: Device to move data to after execution.
        graph: Original GraphModule — all node modules are pre- and post-offloaded.

    Returns:
        New instruction sequence with device placement instructions inserted.
    """
    instructions = _weight_offloading_pass(instructions, compute_device, storage_device, graph)
    instructions = _activation_offloading_pass(instructions, compute_device, storage_device)

    # Cancel out compute(M1) -> storage(M1) -> compute(M1) placements for any module M1.
    instructions = _cancel_module_round_trips(instructions, compute_device, storage_device)

    # Drop activation moves that the tracked device state proves redundant.
    instructions = _cancel_redundant_activation_moves(instructions, compute_device)
    return instructions


def _cancel_module_round_trips(
    instructions: Instructions, compute_device: torch.device, storage_device: torch.device
) -> Instructions:
    """Cancel adjacent MoveModule round-trips."""
    device_pair = {compute_device, storage_device}

    def _is_round_trip(left: Instruction, right: Instruction) -> bool:
        return (
            isinstance(left, MoveModule)
            and isinstance(right, MoveModule)
            and left.module is right.module
            and isinstance(left.device, torch.device)
            and isinstance(right.device, torch.device)
            and {left.device, right.device} == device_pair
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


def _cancel_redundant_activation_moves(
    instructions: Instructions, compute_device: torch.device
) -> Instructions:
    """Cancel redundant MoveActivations via buffer-and-flush.

    MoveActivations instructions are buffered instead of emitted immediately.
    Consecutive moves on the same ref overwrite each other in the buffer, so
    round-trips collapse naturally. When a non-move instruction consumes a ref
    (via ``uses()``), the pending move for that ref is flushed: emitted only if
    the ref's tracked device differs from the move's target.

    Device state is tracked for Call outputs:
    - ``CallModule`` outputs are on ``compute_device`` (MoveModule guarantees this).
    - ``CallFunction``/``CallMethod`` outputs inherit the device of their inputs
      when all inputs agree; otherwise the output device is left unknown.
    """
    ref_device: dict[_BaseRef, torch.device] = {}
    pending: dict[_BaseRef, MoveActivations] = {}
    result: list[Instruction] = []

    def _infer_output_device(instr: Instruction) -> None:
        if isinstance(instr, CallModule):
            ref_device[instr.target.unwrap_ref()] = compute_device
            return
        if not isinstance(instr, (CallFunction, CallMethod)):
            return
        key = instr.target.unwrap_ref()
        devs = {ref_device[a.unwrap_ref()] for a in instr.args if a.unwrap_ref() in ref_device}
        if len(devs) == 1:
            ref_device[key] = devs.pop()
        else:
            ref_device.pop(key, None)

    def _flush(ref: _BaseRef) -> None:
        key = ref.unwrap_ref()
        if (move := pending.pop(key, None)) is not None:
            if ref_device.get(key) != move.device:
                result.append(move)
                ref_device[key] = move.device

    def _is_redundant(move: MoveActivations) -> bool:
        return ref_device.get(move.register_ref.unwrap_ref()) == move.device

    for instr in instructions:
        if isinstance(instr, MoveActivations):
            pending[instr.register_ref.unwrap_ref()] = instr
        else:
            for ref in instr.uses():
                _flush(ref)
            result.append(instr)
            _infer_output_device(instr)

    for move in pending.values():
        if not _is_redundant(move):
            result.append(move)

    return tuple(result)


class OffloadingStrategy(abc.ABC):
    """Abstract base for offloading strategies.

    An offloading strategy controls how module weights and activations are moved
    between devices during graph execution. Implement `create_instruction_pass` to
    insert the appropriate `MoveModule` and `MoveActivations` instructions.
    """

    @abc.abstractmethod
    def create_instruction_pass(self, graph: GraphModule) -> InstructionPass:
        """Return an instruction pass that inserts device-movement instructions.

        Args:
            graph: The GraphModule being scheduled.

        Returns:
            An `InstructionPass` that wraps the instruction sequence with the
            appropriate placement instructions.
        """


@dataclasses.dataclass(frozen=True)
class OffloadEverything(OffloadingStrategy):
    """Offload all module weights and activations between a compute device and a storage device.

    Moves every module's weights and every activation to `storage_device` when idle,
    and back to `compute_device` just before execution.

    Args:
        compute_device: Device where computation happens (e.g. `cuda`).
        storage_device: Device where idle data is stored (e.g. `cpu`).
    """

    compute_device: torch.device
    storage_device: torch.device = dataclasses.field(default_factory=lambda: torch.device("cpu"))

    def create_instruction_pass(self, graph: GraphModule) -> InstructionPass:  # noqa: D102
        def _pass(instructions: Instructions) -> Instructions:
            return _offloading_pass(instructions, self.compute_device, self.storage_device, graph)

        return _pass
