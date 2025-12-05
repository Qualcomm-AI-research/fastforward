# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


import abc
import dataclasses
import itertools

from typing import (
    Any,
    Callable,
    Collection,
    Iterator,
    Mapping,
    Sequence,
    TypeAlias,
)

import torch

from torch.utils.data import DataLoader

from fastforward._orchestration.graph_module import (
    AttributeRef,
    Const,
    GraphModule,
    InputRef,
    NodeRef,
    _BaseRef,
    topological_sort,
)

Register: TypeAlias = dict[_BaseRef, Any]


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


@dataclasses.dataclass(frozen=True)
class Instruction(abc.ABC):
    """Base class for all instructions in the execution engine.

    Each instruction must implement the execute method to define its behavior
    when executed by the InstructionEngine.
    """

    @abc.abstractmethod
    def execute(self, register: Register) -> Any:
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
class StoreConstant(Instruction):
    """Store a constant value in the register."""

    target: _BaseRef
    value: Any

    def execute(self, register: Register) -> None:  # noqa: D102
        register[self.target] = ActivationDataset.from_value(self.value)

    def uses(self) -> Iterator[_BaseRef]:  # noqa: D102
        return iter([self.target])


@dataclasses.dataclass(frozen=True)
class LoadAttribute(Instruction):
    """Load an attribute/index from a register value and store result."""

    source: _BaseRef
    target: _BaseRef
    attribute: str | int

    def execute(self, register: Register) -> None:  # noqa: D102
        source_dataset = register[self.source]
        sample = next(iter(source_dataset))

        match self.attribute:
            case int() as idx:
                batches = [batch[idx] for batch in source_dataset]
            case str() as key if isinstance(sample, Mapping):
                batches = [batch[key] for batch in source_dataset]
            case str() as attr if hasattr(sample, self.attribute):
                batches = [getattr(batch, attr) for batch in source_dataset]
            case _:
                msg = f"Unsupported attribute type: {type(self.attribute).__name__}"
                raise ValueError(msg)

        register[self.target] = ActivationDataset(batches)

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

    def execute(self, register: Register) -> None:  # noqa: D102
        arg_datasets: list[ActivationDataset] = [register[arg] for arg in self.args]
        kwarg_datasets: dict[str, ActivationDataset] = {
            k: register[v] for k, v in self.kwargs.items()
        }

        all_datasets = list(arg_datasets) + list(kwarg_datasets.values())
        if not all_datasets:
            total_len = 1
        else:
            lengths = [len(ds) for ds in all_datasets]
            max_len = max(lengths)
            if len({length for length in lengths if length > 1}) > 1:
                msg = f"Incompatible batch sizes across args/kwargs: {lengths}"
                raise ValueError(msg)
            total_len = max_len

        outputs: list[Any] = []
        for i in range(total_len):
            args = tuple(ds.batches[0 if len(ds) == 1 else i] for ds in arg_datasets)
            kwargs = {k: ds.batches[0 if len(ds) == 1 else i] for k, ds in kwarg_datasets.items()}
            outputs.append(self.module(*args, **kwargs))

        register[self.target] = ActivationDataset(outputs)

    def uses(self) -> Iterator[_BaseRef]:  # noqa: D102
        yield from self.args
        yield from self.kwargs.values()

    def produces(self) -> Iterator[_BaseRef]:  # noqa: D102
        return iter([self.target])


@dataclasses.dataclass(frozen=True)
class OptimizeModule(Instruction):
    """Optimize a module in-place using batched data from the register."""

    module: torch.nn.Module
    args: Sequence[_BaseRef]
    fn: Callable[[torch.nn.Module, Collection[Any]], None]

    def execute(self, register: Register) -> None:  # noqa: D102
        arg_datasets = [register[arg] for arg in self.args]
        merged_dataset = ActivationDataset.merge(arg_datasets)
        self.fn(self.module, merged_dataset)

    def uses(self) -> Iterator[_BaseRef]:  # noqa: D102
        return iter(self.args)


@dataclasses.dataclass(frozen=True)
class ReturnOutputs(Instruction):
    """Merge and return output values from the register.

    The expected outputs are first gathered and merged into a single
    ActivationDataset. We then return the expected outputs (in batches)
    as standard Python types based on the number of batches and outputs.

    Args:
        register: The execution register.

    Returns:
        - Single batch, single output: value
        - Single batch, multiple outputs: tuple of values
        - Multiple batches, single output: tuple of values
        - Multiple batches, multiple outputs: tuple of tuples
    """

    outputs: Sequence[_BaseRef]

    def execute(self, register: Register) -> Any:  # noqa: D102
        datasets = [register[output_id] for output_id in self.outputs]
        merged = ActivationDataset.merge(datasets)

        num_outputs = len(self.outputs)

        # Single batch: return (either value or tuple of values).
        if len(merged) == 1:
            return merged.batches[0]

        # Multiple batches, single output: flatten.
        if num_outputs == 1:
            return tuple(batch[0] for batch in merged.batches)

        # Multiple batches, multiple outputs: tuple of tuples.
        return tuple(merged.batches)

    def uses(self) -> Iterator[_BaseRef]:  # noqa: D102
        return iter(self.outputs)


@dataclasses.dataclass(frozen=True)
class DeleteRegisterEntries(Instruction):
    """Delete specified register entries to free memory."""

    targets: Sequence[_BaseRef]

    def execute(self, register: Register) -> None:  # noqa: D102
        for target_id in self.targets:
            if target_id in register:
                del register[target_id]


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
    ) -> Register:
        """Prepare execution register from user inputs.

        Args:
            input_refs: Mapping from input names to InputRef objects.
            args: Positional arguments from user.
            kwargs: Keyword arguments from user.

        Returns:
            Register mapping InputRefs to ActivationDataset values.

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

        return {
            input_refs[name]: ActivationDataset.from_value(value) for name, value in inputs.items()
        }

    @staticmethod
    def run_instructions(instructions: Instructions, register: Register) -> Any:
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
        register = self.prepare_input_register(program.input_refs, args, kwargs)
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

    def schedule(self, graph: GraphModule) -> InstructionProgram:
        """Schedule node execution and build an engine to run the graph.

        Args:
            graph: GraphModule to schedule execution for.

        Returns:
            InstructionEngine ready to execute the graph.
        """
        order = self._ordering_strategy(graph)
        instructions = self._schedule(graph, order)

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

        node = graph._nodes[node_ref.id]

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
            instructions.append(OptimizeModule(module=node.module, args=args, fn=node.delegate))

        # Always execute the module to cache activations
        instructions.append(
            CallModule(module=node.module, args=args, kwargs=kwargs, target=node_ref)
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
            case Const(value=v):
                # Store constant as ActivationDataset
                store_instruction = StoreConstant(target=ref, value=v)
                return ref, [store_instruction]
            case AttributeRef(reference=attr_ref, attribute=attr):
                base_ref, base_instructions = self._schedule_ref_id(attr_ref)
                # Allocate the extracted value under the AttributeRef so the base output stays intact.
                load_instruction = LoadAttribute(source=base_ref, target=ref, attribute=attr)
                return ref, [*base_instructions, load_instruction]
            case _BaseRef():
                msg = f"Unsupported reference type: {type(ref).__name__}"
                raise TypeError(msg)
