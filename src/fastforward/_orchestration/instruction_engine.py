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
    DEFAULT_CONTEXT,
    AttributeRef,
    Const,
    Contexts,
    Delegate,
    GraphModule,
    InputRef,
    NodeRef,
    Op,
    _BaseRef,
    topological_sort,
)


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


ActivationRegister: TypeAlias = dict[_BaseRef, dict[ContextManager[None], ActivationDataset | Any]]


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
        context: ContextManager[None],
        args: Sequence[_BaseRef],
        kwargs: Mapping[str, _BaseRef],
    ) -> "ActivationBundle":
        """Resolve refs from the register under one context, broadcast, and bundle.

        Args:
            register: The activation register mapping refs to per-context datasets.
            context: The execution context under which to resolve each ref.
            args: Positional input refs in declaration order.
            kwargs: Keyword input refs keyed by parameter name.

        Returns:
            An `ActivationBundle` whose positional and keyword streams are broadcast
            to a common batch length and ready for iteration as `(args, kwargs)` pairs.
        """

        def get_dataset(ref: _BaseRef) -> ActivationDataset:
            if isinstance(ref, Const):
                return ActivationDataset([ref.value])
            return register[ref][context]

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

    def execute(self, register: ActivationRegister) -> None:  # noqa: D102
        register[self.target] = {context: self.value for context in self.contexts}

    def uses(self) -> Iterator[_BaseRef]:  # noqa: D102
        return iter([self.target])


@dataclasses.dataclass(frozen=True)
class LoadAttribute(Instruction):
    """Load an attribute/index from a register value and store result."""

    source: _BaseRef
    target: _BaseRef
    attribute: str | int

    def execute(self, register: ActivationRegister) -> None:  # noqa: D102
        source_contexts = register[self.source]
        register[self.target] = {
            context: self._extract_attribute(dataset)
            for context, dataset in source_contexts.items()
            if isinstance(dataset, ActivationDataset)
        }

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
class CallModule(Instruction):
    """Execute a module on batched data from the register.

    Merges data arguments (zipped) and constant arguments (broadcast),
    then calls the module on each batch with the provided kwargs.
    """

    module: torch.nn.Module | Callable[..., Any]
    args: Sequence[_BaseRef]
    kwargs: dict[str, _BaseRef]
    target: _BaseRef
    contexts: Contexts

    def execute(self, register: ActivationRegister) -> None:  # noqa: D102
        results: dict[ContextManager[None], ActivationDataset] = {}

        for context in self.contexts:
            bundle = ActivationBundle.gather(register, context, self.args, self.kwargs)
            with context:
                if not self.args and not self.kwargs:
                    # A node with no inputs (e.g. a buffer/constant producer) is
                    # called exactly once with no arguments.
                    outputs = [self.module()]
                else:
                    outputs = [self.module(*args, **kwargs) for args, kwargs in bundle]
            results[context] = ActivationDataset(outputs)

        register[self.target] = results

    def uses(self) -> Iterator[_BaseRef]:  # noqa: D102
        yield from self.args
        yield from self.kwargs.values()

    def produces(self) -> Iterator[_BaseRef]:  # noqa: D102
        return iter([self.target])


@dataclasses.dataclass(frozen=True)
class OptimizeModule(Instruction):
    """Optimize a module in-place using batched data from the register.

    Builds one `ActivationBundle` per context from the register and the node's
    positional/keyword inputs, then invokes the user-supplied delegate as
    `fn(module, *bundles_per_context)`. Inside the delegate, iterating each bundle
    yields `(args, kwargs)` per batch — typically called as `module(*args, **kwargs)`.
    """

    module: torch.nn.Module
    args: Sequence[_BaseRef]
    kwargs: Mapping[str, _BaseRef]
    delegate: Delegate

    def execute(self, register: ActivationRegister) -> None:  # noqa: D102
        bundles = [
            ActivationBundle.gather(register, context, self.args, self.kwargs)
            for context in self.delegate.contexts
        ]
        self.delegate.fn(self.module, *bundles)

    def uses(self) -> Iterator[_BaseRef]:  # noqa: D102
        yield from self.args
        yield from self.kwargs.values()


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
        context_outputs = defaultdict(list)
        for output_ref in self.outputs:
            for context, dataset in register[output_ref].items():
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
            if target_id in register:
                del register[target_id]


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

    def execute(self, register: ActivationRegister) -> None:  # noqa: D102, ARG002
        if isinstance(self.device, dict):
            for name, parameter in self.module.named_parameters():
                if name in self.device:
                    parameter.data = parameter.data.to(device=self.device[name], non_blocking=True)
            for name, buffer in self.module.named_buffers():
                if name in self.device:
                    buffer.data = buffer.data.to(device=self.device[name], non_blocking=True)
        else:
            for parameter in self.module.parameters():
                parameter.data = parameter.data.to(device=self.device, non_blocking=True)
            for buffer in self.module.buffers():
                buffer.data = buffer.data.to(device=self.device, non_blocking=True)


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
            return value.to(device=device, non_blocking=True)
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
    context_map = register[ref]
    for context, dataset in context_map.items():
        if isinstance(dataset, ActivationDataset):
            context_map[context] = dataclasses.replace(
                dataset, batches=[_move_to_device(batch, device) for batch in dataset.batches]
            )


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


def _propagate_contexts(graph: GraphModule, order: list[NodeRef]) -> Mapping[_BaseRef, Contexts]:
    """Determine which execution contexts each node needs based on downstream usage.

    Contexts flow backward through he graph, where the forward path is defined by `order`.t

    Args:
        graph: GraphModule to analyze.
        order: Node execution order.

    Returns:
        Mapping from references to their required execution contexts.
    """
    node_contexts: dict[_BaseRef, set[ContextManager[None]]] = defaultdict(set)

    # Seed outputs with the default context so the forward pass always produces results.
    if graph._outputs:
        for out in graph._outputs:
            node_contexts[out.unwrap_ref()] |= {DEFAULT_CONTEXT}

    # Propagate backwards to the graph and ensure if a child node depends on context X
    # the parent will also depend on context X.
    for node_ref in reversed(order):
        node = graph.node(node_ref)

        # Node arguments represent parents in the graph. Propagate contexts to them.
        for node_input in graph.node_inputs(node_ref):
            node_contexts[node_input] |= node_contexts[node_ref]

            if node.delegate is not None:
                node_contexts[node_input] |= set(node.delegate.contexts)

    return {ref: list(contexts) for ref, contexts in node_contexts.items()}


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
        all_contexts = set()

        for instruction in self.instructions:
            match instruction:
                case CallModule(contexts=contexts):
                    all_contexts.update(set(contexts))
                case OptimizeModule(delegate=delegate):
                    all_contexts.update(set(delegate.contexts))
                case StoreValue(contexts=contexts):
                    all_contexts.update(set(contexts))
                case _:
                    pass

        return list(all_contexts)


@dataclasses.dataclass(frozen=True)
class InstructionEngine:
    """Executes pre-scheduled instruction sequences.

    This engine runs instructions in order using a register-based approach
    to track intermediate values and activations. Instructions are generated
    by an InstructionScheduler during the scheduling phase.
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

        register: dict[_BaseRef, Any] = {}
        for input_name, value in inputs.items():
            key = input_refs[input_name]
            register[key] = {context: ActivationDataset.from_value(value) for context in contexts}
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


def optimization_only_pass(instructions: Instructions) -> Instructions:
    """Keep only instructions needed for optimization.

    This filters out any CallModule whose output is not needed for downstream optimization, which
    might include ReturnOutputs if upstream instructions have been removed.
    If no OptimizeModule instructions are present, returns instructions unchanged.

    Args:
        instructions: Sequence of instructions to analyze.

    Returns:
        Filtered instruction sequence.
    """
    has_optimize = any(isinstance(i, OptimizeModule) for i in instructions)
    if not has_optimize:
        return instructions

    required_values: set[_BaseRef] = set()
    retained_instructions: list[Instruction] = []

    # Instructions can only depend on outputs from earlier instructions.
    # Iterate over instructions in reverse to determine dependency relationships.
    for instruction in reversed(instructions):
        if isinstance(instruction, OptimizeModule):
            # Any dependency of OptimizeModule must be retained.
            retained_instructions.append(instruction)
            required_values.update(instruction.uses())
        else:
            outputs = set(instruction.produces())
            if outputs & required_values:
                # If this instruction produces any output required, directly or indirectly,
                # by an OptimizeModule, retain this instruction and its dependencies.
                retained_instructions.append(instruction)
                required_values.update(instruction.uses())

    return tuple(reversed(retained_instructions))


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

    Device state is tracked for CallModule outputs:
    - ``nn.Module`` calls produce on ``compute_device`` (MoveModule guarantees this).
    - Non-Module callables (aten ops) inherit the device of their inputs when
      all inputs agree; otherwise the output device is left unknown.
    """
    ref_device: dict[_BaseRef, torch.device] = {}
    pending: dict[_BaseRef, MoveActivations] = {}
    result: list[Instruction] = []

    def _infer_output_device(instr: Instruction) -> None:
        if not isinstance(instr, CallModule):
            return
        key = instr.target.unwrap_ref()
        if isinstance(instr.module, torch.nn.Module):
            ref_device[key] = compute_device
            return
        devs = {ref_device[a.unwrap_ref()] for a in instr.args if a.unwrap_ref() in ref_device}
        if len(devs) == 1:
            ref_device[key] = devs.pop()

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


class InstructionScheduler:
    """Schedules instruction sequences from GraphModule structure.

    Analyzes graph dependencies to determine node execution order, generates
    instructions for each node, and applies optional transformation passes.
    Produces an InstructionEngine that executes the scheduled instruction sequence.

    Args:
        passes: Optional sequence of transformation passes to apply to scheduled instructions.
        ordering_strategy: Strategy for determining node execution order. Defaults to topological sort.
    """

    def __init__(
        self,
        passes: Sequence[InstructionPass] | None = None,
        ordering_strategy: Callable[[GraphModule], list[NodeRef]] | None = None,
    ) -> None:
        self._passes = passes or []
        self._ordering_strategy = ordering_strategy or topological_sort
        self._node_contexts: Mapping[_BaseRef, Contexts] = {}

    def schedule(self, graph: GraphModule) -> InstructionProgram:
        """Schedule node execution and build an engine to run the graph.

        Args:
            graph: GraphModule to schedule execution for.

        Returns:
            InstructionEngine ready to execute the graph.
        """
        order = self._ordering_strategy(graph)

        self._node_contexts = _propagate_contexts(graph, order)
        try:
            instructions = self._schedule(graph, order)
        finally:
            self._node_contexts = {}

        for pass_fn in self._passes:
            instructions = pass_fn(instructions)

        return InstructionProgram(instructions=instructions, input_refs=graph._inputs)

    def _schedule(self, graph: GraphModule, order: list[NodeRef]) -> Instructions:
        """Schedule instructions from ordered nodes.

        Args:
            graph: GraphModule containing nodes.
            order: Node execution order.

        Returns:
            Sequence of Instructions in execution order.
        """
        instructions: list[Instruction] = []

        for node_ref in order:
            instructions.extend(self._schedule_node(node_ref, graph))

        if graph._outputs:
            return_instruction, output_prerequisites = self._schedule_return(graph._outputs)
            instructions.extend(output_prerequisites + [return_instruction])

        return tuple(instructions)

    def _schedule_node(self, node_ref: NodeRef, graph: GraphModule) -> list[Instruction]:
        """Schedule instructions for a single node.

        Args:
            node_ref: reference to node to be scheduled.
            graph: GraphModule containing nodes.

        Returns:
            List of instructions to execute this node.
        """
        instructions: list[Instruction] = []

        node = graph.node(node_ref)

        args = []
        for arg in node.args:
            ref_id, new_instructions = self._schedule_ref_id(arg)
            instructions.extend(new_instructions)
            args.append(ref_id)

        kwargs = {}
        for key, arg in node.kwargs.items():
            ref_id, new_instructions = self._schedule_ref_id(arg)
            instructions.extend(new_instructions)
            kwargs[key] = ref_id

        # Inject optimization if specified
        if node.delegate is not None and isinstance(node.target, torch.nn.Module):
            instructions.append(
                OptimizeModule(module=node.target, args=args, kwargs=kwargs, delegate=node.delegate)
            )

        # Always execute the module to cache activations
        instructions.append(
            CallModule(
                module=node.target,
                args=args,
                kwargs=kwargs,
                target=node_ref,
                contexts=self._node_contexts.get(node_ref, []),
            )
        )

        return instructions

    def _schedule_return(
        self, outputs: list[NodeRef | AttributeRef]
    ) -> tuple[ReturnOutputs, list[Instruction]]:
        """Schedule return instruction for graph outputs.

        Compiles all output references to register slots.

        Args:
            outputs: List of output references from the graph.

        Returns:
            Tuple of (ReturnOutputs instruction, prerequisite instructions for output preparation).
            Prerequisite instructions handle attribute extraction for AttributeRef outputs.
        """
        output_refs = []
        prerequisites: list[Instruction] = []

        for output_ref in outputs:
            ref, new_instructions = self._schedule_ref_id(output_ref)
            prerequisites.extend(new_instructions)
            output_refs.append(ref)

        return ReturnOutputs(outputs=output_refs), prerequisites

    def _schedule_ref_id(self, ref: _BaseRef) -> tuple[_BaseRef, list[Instruction]]:
        """Schedule instructions to resolve a reference to a register slot or constant value.

        NodeRef/InputRef are already usable register keys. Const adds a StoreConstant so the value
        is populated under that key. AttributeRef appends a LoadAttribute such that its register key won't
        overwrite the base reference slot.

        Args:
            ref: Reference whose runtime value must be resolved.

        Returns:
            The original reference and the instructions required to materialize its value.
        """
        match ref:
            case NodeRef() | InputRef():
                return ref, []
            case Const():
                # Store the value of the constant directly to the register.
                contexts = list(self._node_contexts.get(ref, set())) or [DEFAULT_CONTEXT]
                store_instruction = StoreValue(target=ref, value=ref, contexts=contexts)
                return ref, [store_instruction]
            case AttributeRef(reference=attr_ref, attribute=attr):
                base_ref, base_instructions = self._schedule_ref_id(attr_ref)
                # Allocate the extracted value under the AttributeRef so the base output stays intact.
                load_instruction = LoadAttribute(source=base_ref, target=ref, attribute=attr)
                return ref, [*base_instructions, load_instruction]
            case _BaseRef():
                msg = f"Unsupported reference type: {type(ref).__name__}"
                raise TypeError(msg)


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
