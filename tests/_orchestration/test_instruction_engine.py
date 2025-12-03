# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import dataclasses
import uuid

from typing import Any, Collection

import pytest
import torch

from fastforward._orchestration.graph_module import (
    GraphModule,
)
from fastforward._orchestration.instruction_engine import (
    ActivationDataset,
    CallModule,
    DeleteRegisterEntries,
    InstructionEngine,
    InstructionScheduler,
    LoadAttribute,
    OptimizeModule,
    ReturnOutputs,
    lifetime_management_pass,
    optimization_only_pass,
)


def test_merge_zips_datasets_together() -> None:
    """Test that merge correctly zips multiple datasets element-wise."""
    # GIVEN three datasets with identical lengths
    ds1 = ActivationDataset([1, 2, 3])
    ds2 = ActivationDataset(["a", "b", "c"])
    ds3 = ActivationDataset([10, 20, 30])

    # WHEN we merge them
    merged = ActivationDataset.merge([ds1, ds2, ds3])

    # THEN batches are zipped into tuples element-wise
    assert len(merged) == 3
    assert merged.batches == [(1, "a", 10), (2, "b", 20), (3, "c", 30)]


def test_merge_with_mismatched_lengths_raises() -> None:
    """Test that merge raises ValueError when dataset lengths don't match."""
    # GIVEN datasets with different lengths
    ds1 = ActivationDataset([1, 2])
    ds2 = ActivationDataset(["a", "b", "c"])
    ds3 = ActivationDataset([10, 20])

    # WHEN we try to merge them
    # THEN it should raise ValueError
    with pytest.raises(ValueError):
        ActivationDataset.merge([ds1, ds2, ds3])


def test_prepare_input_register_validates_inputs() -> None:
    """Test that prepare_input_register validates args/kwargs and wraps in ActivationDatasets."""
    # GIVEN a graph with three inputs
    graph = GraphModule()
    x = graph.add_input("x")
    y = graph.add_input("y")
    z = graph.add_input("z")

    # WHEN we bind with mixed positional and keyword args
    register: dict[uuid.UUID, ActivationDataset] = InstructionEngine.prepare_input_register(
        graph._inputs, args=(10, 20), kwargs={"z": 30}
    )

    # THEN all inputs are wrapped as ActivationDatasets with correct IDs
    assert len(register) == 3
    assert register[x.id].batches == [10]
    assert register[y.id].batches == [20]
    assert register[z.id].batches == [30]

    # AND missing inputs raise TypeError
    with pytest.raises(TypeError, match="Missing required inputs"):
        InstructionEngine.prepare_input_register(graph._inputs, args=(10,), kwargs={})

    # AND duplicate bindings raise TypeError
    with pytest.raises(TypeError, match="Multiple values for argument"):
        InstructionEngine.prepare_input_register(graph._inputs, args=(10, 20), kwargs={"x": 99})

    # AND unknown kwargs raise TypeError
    with pytest.raises(TypeError, match="Unexpected keyword argument"):
        InstructionEngine.prepare_input_register(graph._inputs, args=(10,), kwargs={"unknown": 99})


def test_instruction_generator_linear_layers() -> None:
    """Test instruction generation for simple linear chain: input -> node1 -> node2."""
    # GIVEN an execution plan with two nodes in sequence
    graph = GraphModule()
    inputs = graph.add_input("inputs")
    node_1 = graph.add_node("node_1", torch.nn.Identity(), [inputs])
    node_2 = graph.add_node("node_1", torch.nn.Identity(), [node_1])
    graph.add_output(node_2)

    # WHEN we schedule execution
    scheduler = InstructionScheduler()
    engine = scheduler.schedule(graph)

    # THEN the engine contains: CallModule(node1), CallModule(node2), ReturnOutputs
    assert len(engine.instructions) == 3
    assert isinstance(engine.instructions[0], CallModule)
    assert engine.instructions[0].args == [inputs.id]
    assert engine.instructions[0].target == node_1.id

    assert isinstance(engine.instructions[1], CallModule)
    assert engine.instructions[1].args == [node_1.id]
    assert engine.instructions[1].target == node_2.id

    assert isinstance(engine.instructions[2], ReturnOutputs)
    assert engine.instructions[2].outputs == [node_2.id]


def test_instruction_generator_with_attribute_ref() -> None:
    """Test instruction generation when AttributeRef is used."""
    # GIVEN a graph where a node returns a tuple and we extract an element for next node
    graph = GraphModule()

    class ReturnTuple(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return x, x * 2

    inputs = graph.add_input("x")
    tuple_node = graph.add_node("tuple_node", ReturnTuple(), [inputs])
    identity_node = graph.add_node("identity", torch.nn.Identity(), [tuple_node[0]])
    graph.add_output(identity_node)

    # WHEN we schedule execution
    scheduler = InstructionScheduler()
    engine = scheduler.schedule(graph)

    # THEN the engine contains: CallModule(tuple_node), LoadAttribute(extract [0]), CallModule(identity), ReturnOutputs
    assert len(engine.instructions) == 4
    assert isinstance(engine.instructions[0], CallModule)
    assert engine.instructions[0].target == tuple_node.id

    assert isinstance(engine.instructions[1], LoadAttribute)
    assert engine.instructions[1].source == tuple_node.id
    assert engine.instructions[1].attribute == 0

    assert isinstance(engine.instructions[2], CallModule)
    assert engine.instructions[2].args == [engine.instructions[1].target]
    assert engine.instructions[2].target == identity_node.id

    assert isinstance(engine.instructions[3], ReturnOutputs)


def test_instruction_generator_with_optimization_spec() -> None:
    """Test that SubgraphSpec with optimization function injects OptimizeModule instruction."""
    # GIVEN a simple graph with two nodes
    graph = GraphModule()
    inputs = graph.add_input("x")
    node_1 = graph.add_node("node_1", torch.nn.Identity(), [inputs])
    node_2 = graph.add_node("node_2", torch.nn.Identity(), [node_1])
    graph.add_output(node_2)

    # GIVEN a SubgraphSpec that targets node_1 with an optimization function
    def dummy_optimize(module: torch.nn.Module, dataset: Collection[Any]) -> None:
        pass

    graph._nodes[node_1.id] = dataclasses.replace(graph._nodes[node_1.id], delegate=dummy_optimize)

    # WHEN we schedule execution with the spec
    scheduler = InstructionScheduler()
    engine = scheduler.schedule(graph)

    # THEN the engine contains: OptimizeModule(node_1), CallModule(node_1), CallModule(node_2), ReturnOutputs
    assert len(engine.instructions) == 4

    assert isinstance(engine.instructions[0], OptimizeModule)
    assert engine.instructions[0].module is graph._nodes[node_1.id].module
    assert engine.instructions[0].fn is dummy_optimize

    assert isinstance(engine.instructions[1], CallModule)
    assert engine.instructions[1].target == node_1.id

    assert isinstance(engine.instructions[2], CallModule)
    assert engine.instructions[2].target == node_2.id

    assert isinstance(engine.instructions[3], ReturnOutputs)


def test_optimization_only_pass() -> None:
    """Test that optimization_only_pass keeps only instructions needed for optimization."""
    # GIVEN a graph with optimization on node_1 and node_3, but not node_2
    graph = GraphModule()
    inputs = graph.add_input("x")
    node_1 = graph.add_node("node_1", torch.nn.Identity(), [inputs])
    node_2 = graph.add_node("node_2", torch.nn.Identity(), [node_1])
    node_3 = graph.add_node("node_3", torch.nn.Identity(), [node_2])
    graph.add_output(node_3)

    def dummy_optimize(module: torch.nn.Module, dataset: Collection[Any]) -> None:
        pass

    graph._nodes[node_1.id] = dataclasses.replace(graph._nodes[node_1.id], delegate=dummy_optimize)
    graph._nodes[node_3.id] = dataclasses.replace(graph._nodes[node_3.id], delegate=dummy_optimize)

    # WHEN instructions are scheduled with the optimization_only_pass
    scheduler = InstructionScheduler(passes=[optimization_only_pass])
    instructions = scheduler.schedule(graph).instructions

    # THEN the resulting instructions should be exactly 4,
    # OptimizeModule(node_1) -> CallModule(node_1) -> CallModule(node_2) -> OptimizeModule(node_3)
    # CallModule(node_3) is removed since nothing downstream depends on it, and because of that
    # ReturnOutputs is also removed.
    assert len(instructions) == 4
    assert isinstance(instructions[0], OptimizeModule)
    assert instructions[0].module is graph._nodes[node_1.id].module
    assert isinstance(instructions[1], CallModule)
    assert instructions[1].module is graph._nodes[node_1.id].module
    assert isinstance(instructions[2], CallModule)
    assert instructions[2].module is graph._nodes[node_2.id].module
    assert isinstance(instructions[3], OptimizeModule)
    assert instructions[3].module is graph._nodes[node_3.id].module


def test_lifetime_management_pass() -> None:
    """Test that lifetime_management_pass inserts ClearRegister after last use."""
    # GIVEN a sequential graph x -> node_1 -> node_2 -> out
    graph = GraphModule()
    inputs = graph.add_input("x")
    node_1 = graph.add_node("node_1", torch.nn.Identity(), [inputs])
    node_2 = graph.add_node("node_2", torch.nn.Identity(), [node_1])
    graph.add_output(node_2)

    # WHEN instructions are scheduled with the lifetime_management_pass
    scheduler = InstructionScheduler(passes=[lifetime_management_pass])
    instructions = scheduler.schedule(graph).instructions

    # THEN the resulting instructions should be exactly 5,
    # CallM(node_1) -> Del(inp) -> CallM(node_2) -> Del(node_1) -> Ret(node_2)
    assert len(instructions) == 5
    assert isinstance(instructions[0], CallModule)
    assert (
        isinstance(instructions[1], DeleteRegisterEntries)
        and instructions[1].targets[0] == inputs.id
    )
    assert isinstance(instructions[2], CallModule)
    assert (
        isinstance(instructions[3], DeleteRegisterEntries)
        and instructions[3].targets[0] == node_1.id
    )
    assert isinstance(instructions[4], ReturnOutputs)
