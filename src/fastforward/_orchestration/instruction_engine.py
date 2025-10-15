# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


import abc
import dataclasses
import uuid

from typing import (
    Any,
    Callable,
    Collection,
    Iterator,
    Sequence,
    TypeAlias,
)

import torch

from fastforward._orchestration.graph_module import (
    AttributeRef,
    Const,
    GraphModule,
    InputRef,
    Node,
    NodeRef,
    _BaseRef,
    topological_sort,
)

Register: TypeAlias = dict[uuid.UUID, Any]


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

        batches = list(value) if isinstance(value, (list, tuple)) else [value]
        return cls(batches)

    @classmethod
    def merge(cls, datasets: Sequence["ActivationDataset"]) -> "ActivationDataset":
        """Zip multiple ActivationDatasets (with identical lengths).

        Args:
            datasets: Non-empty sequence of ActivationDatasets to merge.

        Returns:
            ActivationDataset where each batch is a tuple of corresponding elements.
        """
        if len(datasets) == 1:
            return datasets[0]

        try:
            batches = [tuple(vals) for vals in zip(*[ds.batches for ds in datasets], strict=True)]
        except ValueError as e:
            msg = "Tried to merge datasets of unequal size"
            raise ValueError(msg) from e

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
            register: The execution register mapping IDs to values.

        Returns:
            The result of executing this instruction.
        """


@dataclasses.dataclass(frozen=True)
class StoreConstant(Instruction):
    """Store a constant value in the register."""

    target: uuid.UUID
    value: Any

    def execute(self, register: Register) -> None:  # noqa: D102
        register[self.target] = ActivationDataset.from_value(self.value)


@dataclasses.dataclass(frozen=True)
class LoadAttribute(Instruction):
    """Load an attribute/index from a register value and store result."""

    source: uuid.UUID
    target: uuid.UUID
    attribute: str | int

    def execute(self, register: Register) -> None:  # noqa: D102
        source_dataset = register[self.source]

        if isinstance(self.attribute, int):
            batches = [batch[self.attribute] for batch in source_dataset]
        else:
            batches = [getattr(batch, self.attribute) for batch in source_dataset]

        register[self.target] = ActivationDataset(batches)


@dataclasses.dataclass(frozen=True)
class CallModule(Instruction):
    """Execute a module on batched data from the register.

    Merges data arguments (zipped) and constant arguments (broadcast),
    then calls the module on each batch with the provided kwargs.
    """

    module: torch.nn.Module
    args: Sequence[uuid.UUID]
    kwargs: dict[str, uuid.UUID]
    target: uuid.UUID

    def execute(self, register: Register) -> None:  # noqa: D102
        arg_datasets = [register[arg] for arg in self.args]
        merged_args = ActivationDataset.merge(arg_datasets)

        kwarg_datasets: dict[str, ActivationDataset] = {
            k: register[v] for k, v in self.kwargs.items()
        }

        outputs: list[Any] = []
        for batch in merged_args:
            batch_tuple = batch if isinstance(batch, tuple) else (batch,)
            batch_kwargs = {k: v.batches[len(outputs)] for k, v in kwarg_datasets.items()}
            outputs.append(self.module(*batch_tuple, **batch_kwargs))

        register[self.target] = ActivationDataset(outputs)


@dataclasses.dataclass(frozen=True)
class OptimizeModule(Instruction):
    """Optimize a module in-place using batched data from the register."""

    module: torch.nn.Module
    args: Sequence[uuid.UUID]
    fn: Callable[[torch.nn.Module, Collection[Any]], None]

    def execute(self, register: Register) -> None:  # noqa: D102
        arg_datasets = [register[arg] for arg in self.args]
        merged_dataset = ActivationDataset.merge(arg_datasets)
        self.fn(self.module, merged_dataset)


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

    outputs: Sequence[uuid.UUID]

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
            Register mapping InputRef UUIDs to ActivationDataset values.

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
            input_refs[name].id: ActivationDataset.from_value(value)
            for name, value in inputs.items()
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
            node = graph._nodes[node_ref.id]
            instructions.extend(self._schedule_node(node))

        if graph._outputs:
            return_instruction, output_prerequisites = self._schedule_return(graph._outputs)
            instructions.extend(output_prerequisites + [return_instruction])

        return tuple(instructions)

    def _schedule_node(self, node: Node) -> list[Instruction]:
        """Schedule instructions for a single node.

        Args:
            node: Node to schedule instructions for.

        Returns:
            List of instructions to execute this node.
        """
        instructions: list[Instruction] = []

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
            CallModule(module=node.module, args=args, kwargs=kwargs, target=node.id)
        )

        return instructions

    def _schedule_return(
        self, outputs: list[NodeRef | AttributeRef]
    ) -> tuple[ReturnOutputs, list[Instruction]]:
        """Schedule return instruction for graph outputs.

        Compiles all output references to register IDs or constant values.

        Args:
            outputs: List of output references from the graph.

        Returns:
            Tuple of (ReturnOutputs instruction, prerequisite instructions for output preparation).
            Prerequisite instructions handle attribute extraction for AttributeRef outputs.
        """
        output_ids = []
        prerequisites: list[Instruction] = []

        for output_ref in outputs:
            ref_id, new_instructions = self._schedule_ref_id(output_ref)
            prerequisites.extend(new_instructions)
            output_ids.append(ref_id)

        return ReturnOutputs(outputs=output_ids), prerequisites

    def _schedule_ref_id(self, ref: _BaseRef) -> tuple[uuid.UUID, list[Instruction]]:
        """Schedule instructions to resolve a reference to a register ID or constant value.

        For most references (NodeRef, InputREef, Const) we map directly to register IDs or
        values. AttributeRef requires runtime calculation so we first load the reference,
        extract the attribute and storing that in a register.

        Args:
            ref: Reference to schedule ID for.

        Returns:
            Tuple of (register_id_or_value, instructions_to_execute).
            Register ID (UUID) for data references, constant value for Const.
        """
        match ref:
            case NodeRef(id=node_id) | InputRef(id=node_id):
                return node_id, []
            case Const(value=v):
                # Store constant as ActivationDataset
                const_id = uuid.uuid4()
                store_instruction = StoreConstant(target=const_id, value=v)
                return const_id, [store_instruction]
            case AttributeRef(reference=base_ref, attribute=attr):
                # Ensure the base reference is computed and get its ID.
                base_id, base_instructions = self._schedule_ref_id(base_ref)

                # Allocate a new register slot for attribute extraction.
                target_id = uuid.uuid4()
                load_instruction = LoadAttribute(source=base_id, target=target_id, attribute=attr)
                return target_id, [*base_instructions, load_instruction]

            case _BaseRef():
                msg = f"Unsupported reference type: {type(ref).__name__}"
                raise TypeError(msg)
