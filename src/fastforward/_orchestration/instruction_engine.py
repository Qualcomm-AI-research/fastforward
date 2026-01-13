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


ActivationRegister: TypeAlias = dict[_BaseRef, dict[ContextManager[None], ActivationDataset | Any]]


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

    module: torch.nn.Module
    args: Sequence[_BaseRef]
    kwargs: dict[str, _BaseRef]
    target: _BaseRef
    contexts: Contexts

    def execute(self, register: ActivationRegister) -> None:  # noqa: D102
        results: dict[ContextManager[None], ActivationDataset] = {}

        for context in self.contexts:
            args_datasets, kwargs_datasets = self._gather_datasets(register, context)
            with context:
                outputs = self._forward_batches(args_datasets, kwargs_datasets)
            results[context] = ActivationDataset(outputs)

        register[self.target] = results

    def _gather_datasets(
        self, register: ActivationRegister, context: ContextManager[None]
    ) -> tuple[list[ActivationDataset], dict[str, ActivationDataset]]:
        """Extract and validate positional and keyword argument datasets given a context name.

        Args:
            register: Register with contexts as keys and datasets as values.
            context: The context which has generated the required datasets.

        Returns:
            Tuple of (arg_datasets, kwarg_datasets).
        """
        # The batch size is the length of any data loader we catch that is not a Constant.
        # If there is no data loader the batch size is 1 (run once).
        batch_size = 1
        for ref in itertools.chain(self.args, self.kwargs.values()):
            if not isinstance(ref, Const):
                batch_size = len(register[ref][context])
                break

        def get_dataset(ref: _BaseRef) -> ActivationDataset:
            # Repeat constant values for the entire batch.
            if isinstance(ref, Const):
                return ActivationDataset([ref.value] * batch_size)
            return register[ref][context]

        arg_datasets = [get_dataset(ref) for ref in self.args]
        kwarg_datasets = {key: get_dataset(ref) for key, ref in self.kwargs.items()}

        return arg_datasets, kwarg_datasets

    def _forward_batches(
        self, arg_datasets: list[ActivationDataset], kwarg_datasets: dict[str, ActivationDataset]
    ) -> list[Any]:
        """Forward pass arg and kwarg datasets through the module.

        Args:
            arg_datasets: List of argument datasets.
            kwarg_datasets: Dictionary of keyword argument datasets.
            context: Context manager to use during execution.

        Returns:
            List of module outputs for each batch.
        """
        if not arg_datasets and not kwarg_datasets:
            return [self.module()]

        outputs = []
        kwarg_keys = list(kwarg_datasets.keys())
        num_args = len(arg_datasets)

        try:
            for batch_data in zip(*arg_datasets, *kwarg_datasets.values(), strict=True):
                batch_args = batch_data[:num_args]
                batch_kwargs = dict(zip(kwarg_keys, batch_data[num_args:]))
                outputs.append(self.module(*batch_args, **batch_kwargs))
        except ValueError as e:
            if "zip()" in str(e):
                msg = (
                    f"Dataset length mismatch in CallModule for {self.module.__class__.__name__}. "
                    f"All inputs must have the same number of batches. "
                    f"Please verify that all datasets passed to this module have matching lengths."
                )
                raise ValueError(msg) from e
            else:
                raise

        return outputs

    def uses(self) -> Iterator[_BaseRef]:  # noqa: D102
        yield from self.args
        yield from self.kwargs.values()

    def produces(self) -> Iterator[_BaseRef]:  # noqa: D102
        return iter([self.target])


@dataclasses.dataclass(frozen=True)
class OptimizeModule(Instruction):
    """Optimize a module in-place using batched data from the register.

    Execute a user-defined optimization function (delegate) on the module using context
    specific activations per arg.
    """

    module: torch.nn.Module
    args: Sequence[_BaseRef]
    delegate: Delegate

    def execute(self, register: ActivationRegister) -> None:  # noqa: D102
        delegate_args = []

        for context in self.delegate.contexts:
            # For each context, gather all arguments the module expects into an ActivationDataset.
            context_args = [register[arg][context] for arg in self.args]
            delegate_args.append(ActivationDataset.merge(context_args))

        # Run the delegate function with per-context arguments required for the module.
        self.delegate.fn(self.module, *delegate_args)

    def uses(self) -> Iterator[_BaseRef]:  # noqa: D102
        return iter(self.args)


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

    # If no delegates exist, we want to ensure the model still functions as expected (e.g. out = model(in)).
    # We do this here by adding the default context to the outputs, and this will propagate to it's
    # parents below.
    graph_has_delegates = any(graph.node(ref).delegate is not None for ref in order)
    if not graph_has_delegates and graph._outputs:
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
        if node.delegate is not None:
            instructions.append(
                OptimizeModule(module=node.module, args=args, delegate=node.delegate)
            )

        # Always execute the module to cache activations
        instructions.append(
            CallModule(
                module=node.module,
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
