# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import dataclasses
import functools
import uuid

from contextlib import nullcontext
from typing import Any, Collection

import pytest
import torch

from fastforward._orchestration.graph_module import (
    DEFAULT_CONTEXT,
    Const,
    Delegate,
    GraphModule,
    NodeRef,
    SubgraphSpec,
    _BaseRef,
    local_optimize,
)
from fastforward._orchestration.instruction_engine import (
    ActivationBundle,
    ActivationDataset,
    ActivationRegister,
    CallModule,
    DeleteRegisterEntries,
    InstructionEngine,
    InstructionPasses,
    InstructionScheduler,
    LoadAttribute,
    MoveActivations,
    MoveModule,
    OffloadEverything,
    OptimizeModule,
    ReturnOutputs,
    StoreValue,
    _cancel_module_round_trips,
    _cancel_redundant_activation_moves,
    _weight_offloading_pass,
    lifetime_management_pass,
    optimization_only_pass,
)

from ._models import Add, AddConstant, Model, ReturnTuple
from .conftest import sgd_step


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


def test_broadcast_repeats_single_batch_to_match_n_batches() -> None:
    # GIVEN a 1-batch dataset and a 3-batch dataset
    single = ActivationDataset([torch.tensor(10.0)])
    multi = ActivationDataset([torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)])

    # WHEN broadcast aligns them
    result = ActivationDataset.broadcast([single, multi])

    # THEN the 1-batch dataset is repeated to length 3, the 3-batch is unchanged
    assert len(result[0]) == 3
    assert len(result[1]) == 3
    assert all(torch.equal(b, torch.tensor(10.0)) for b in result[0])


def test_broadcast_raises_on_incompatible_lengths() -> None:
    # GIVEN datasets with genuinely incompatible lengths (2 vs 3)
    ds_a = ActivationDataset([1, 2])
    ds_b = ActivationDataset([10, 20, 30])

    # WHEN / THEN broadcasting raises
    with pytest.raises(ValueError, match="Dataset length mismatch"):
        ActivationDataset.broadcast([ds_a, ds_b])


def test_broadcast_is_noop_when_all_lengths_match() -> None:
    # GIVEN datasets that already have the same length
    ds_a = ActivationDataset([1, 2, 3])
    ds_b = ActivationDataset([4, 5, 6])

    # WHEN broadcast is called
    result = ActivationDataset.broadcast([ds_a, ds_b])

    # THEN the original datasets are returned unchanged
    assert result[0] is ds_a
    assert result[1] is ds_b


def test_call_module_broadcasts_single_batch_to_match_n_batches() -> None:
    """Test that a 1-batch dataset is broadcast to N when paired with an N-batch dataset."""
    # GIVEN a graph with a module that takes two inputs
    graph = GraphModule()

    input1 = graph.add_input("x")
    input2 = graph.add_input("y")
    node = graph.add_node("add", Add(), [input1, input2])
    graph.add_output(node)

    # WHEN we pass a 3-batch dataset and a 1-batch dataset (batch-invariant value)
    dataset1 = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]
    dataset2 = [torch.tensor(10.0)]

    # THEN the 1-batch is broadcast to 3 and the graph runs successfully
    result = graph(dataset1, dataset2)

    # THEN there is only a single output (of length 3) in the dict
    _, (out1, out2, out3) = result.popitem()
    assert result == {}
    # THEN the outputs are exactly as expected
    assert out1.item() == 1.0 + 10.0
    assert out2.item() == 2.0 + 10.0
    assert out3.item() == 3.0 + 10.0


def test_call_module_with_no_inputs_calls_module_once() -> None:
    # GIVEN a CallModule with no positional and no keyword refs
    context = nullcontext()
    target = NodeRef(id=uuid.uuid4(), name="const_producer")

    sentinel = torch.tensor([42.0])

    def produce() -> torch.Tensor:
        return sentinel

    instr = CallModule(module=produce, args=[], kwargs={}, target=target, contexts=[context])
    register: ActivationRegister = {}

    # WHEN the instruction executes
    instr.execute(register)

    # THEN the module is called exactly once and its single output is stored
    stored = register[target][context]
    assert isinstance(stored, ActivationDataset)
    assert list(stored) == [sentinel]


def test_prepare_input_register_validates_inputs() -> None:
    """Test that prepare_input_register validates args/kwargs and wraps in ActivationDatasets."""
    # GIVEN a graph with three inputs
    graph = GraphModule()
    x = graph.add_input("x")
    y = graph.add_input("y")
    z = graph.add_input("z")

    # WHEN we bind with mixed positional and keyword args
    context = nullcontext()
    register: dict[_BaseRef, Any] = InstructionEngine.prepare_input_register(
        graph._inputs, args=(10, 20), kwargs={"z": 30}, contexts=[context]
    )

    # THEN all inputs are wrapped as ActivationDatasets with correct IDs
    assert len(register) == 3
    assert register[x][context].batches == [10]
    assert register[y][context].batches == [20]
    assert register[z][context].batches == [30]

    # AND missing inputs raise TypeError
    with pytest.raises(TypeError, match="Missing required inputs"):
        InstructionEngine.prepare_input_register(
            graph._inputs, args=(10,), kwargs={}, contexts=[context]
        )

    # AND duplicate bindings raise TypeError
    with pytest.raises(TypeError, match="Multiple values for argument"):
        InstructionEngine.prepare_input_register(
            graph._inputs, args=(10, 20), kwargs={"x": 99}, contexts=[context]
        )

    # AND unknown kwargs raise TypeError
    with pytest.raises(TypeError, match="Unexpected keyword argument"):
        InstructionEngine.prepare_input_register(
            graph._inputs, args=(10,), kwargs={"unknown": 99}, contexts=[context]
        )


def test_return_outputs_unpacking() -> None:
    """Test ReturnOutputs behavior: always returns dict[context, tuple[batches]]."""
    context = nullcontext()
    ref1 = NodeRef(id=uuid.uuid4(), name="n1")
    ref2 = NodeRef(id=uuid.uuid4(), name="n2")

    # Single output, single batch -> dict with tuple containing single batch
    register: ActivationRegister = {ref1: {context: ActivationDataset([1])}}
    result = ReturnOutputs([ref1]).execute(register)
    assert context in result
    assert result[context] == (1,)

    # Single output, multiple batches -> dict with tuple containing all batches
    register = {ref1: {context: ActivationDataset([1, 2])}}
    result = ReturnOutputs([ref1]).execute(register)
    assert context in result
    assert result[context] == (1, 2)

    # Multiple outputs, single batch -> dict with tuple containing single zipped batch
    register = {
        ref1: {context: ActivationDataset([1])},
        ref2: {context: ActivationDataset([2])},
    }
    result = ReturnOutputs([ref1, ref2]).execute(register)
    assert context in result
    assert result[context] == ((1, 2),)

    # Multiple outputs, multiple batches -> dict with tuple of zipped batches
    register = {
        ref1: {context: ActivationDataset([1, 2])},
        ref2: {context: ActivationDataset([3, 4])},
    }
    result = ReturnOutputs([ref1, ref2]).execute(register)
    assert context in result
    assert result[context] == ((1, 3), (2, 4))


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
    assert engine.instructions[0].args == [inputs]
    assert engine.instructions[0].target == node_1

    assert isinstance(engine.instructions[1], CallModule)
    assert engine.instructions[1].args == [node_1]
    assert engine.instructions[1].target == node_2

    assert isinstance(engine.instructions[2], ReturnOutputs)
    assert engine.instructions[2].outputs == [node_2]


def test_instruction_generator_with_attribute_ref() -> None:
    """Test instruction generation when AttributeRef is used."""
    # GIVEN a graph where a node returns a tuple and we extract an element for next node
    graph = GraphModule()

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
    assert engine.instructions[0].target == tuple_node

    assert isinstance(engine.instructions[1], LoadAttribute)
    assert engine.instructions[1].source == tuple_node
    assert engine.instructions[1].attribute == 0

    assert isinstance(engine.instructions[2], CallModule)
    assert engine.instructions[2].args == [engine.instructions[1].target]
    assert engine.instructions[2].target == identity_node

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

    delegate = Delegate(fn=dummy_optimize, contexts=[nullcontext()])
    graph._nodes[node_1.id] = dataclasses.replace(graph._nodes[node_1.id], delegate=delegate)

    # WHEN we schedule execution with the spec
    scheduler = InstructionScheduler()
    engine = scheduler.schedule(graph)

    # THEN the engine contains: OptimizeModule(node_1), CallModule(node_1), CallModule(node_2), ReturnOutputs
    assert len(engine.instructions) == 4

    assert isinstance(engine.instructions[0], OptimizeModule)
    assert engine.instructions[0].module is graph._nodes[node_1.id].target
    assert engine.instructions[0].delegate.fn is dummy_optimize

    assert isinstance(engine.instructions[1], CallModule)
    assert engine.instructions[1].target == node_1

    assert isinstance(engine.instructions[2], CallModule)
    assert engine.instructions[2].target == node_2

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

    delegate = Delegate(fn=dummy_optimize, contexts=[nullcontext()])
    graph._nodes[node_1.id] = dataclasses.replace(graph._nodes[node_1.id], delegate=delegate)
    graph._nodes[node_3.id] = dataclasses.replace(graph._nodes[node_3.id], delegate=delegate)

    # WHEN instructions are scheduled with the optimization_only_pass
    program = InstructionScheduler().schedule(graph)
    program = InstructionPasses.apply(program, [optimization_only_pass])
    instructions = program.instructions

    # THEN the resulting instructions should be exactly 4,
    # OptimizeModule(node_1) -> CallModule(node_1) -> CallModule(node_2) -> OptimizeModule(node_3)
    # CallModule(node_3) is removed since nothing downstream depends on it, and because of that
    # ReturnOutputs is also removed.
    assert len(instructions) == 4
    assert isinstance(instructions[0], OptimizeModule)
    assert instructions[0].module is graph._nodes[node_1.id].target
    assert isinstance(instructions[1], CallModule)
    assert instructions[1].module is graph._nodes[node_1.id].target
    assert isinstance(instructions[2], CallModule)
    assert instructions[2].module is graph._nodes[node_2.id].target
    assert isinstance(instructions[3], OptimizeModule)
    assert instructions[3].module is graph._nodes[node_3.id].target


def test_lifetime_management_pass() -> None:
    """Test that lifetime_management_pass inserts ClearRegister after last use."""
    # GIVEN a sequential graph x -> node_1 -> node_2 -> out
    graph = GraphModule()
    inputs = graph.add_input("x")
    node_1 = graph.add_node("node_1", torch.nn.Identity(), [inputs])
    node_2 = graph.add_node("node_2", torch.nn.Identity(), [node_1])
    graph.add_output(node_2)

    # WHEN instructions are scheduled with the lifetime_management_pass
    program = InstructionScheduler().schedule(graph)
    program = InstructionPasses.apply(program, [lifetime_management_pass])
    instructions = program.instructions

    # THEN the resulting instructions should be exactly 5,
    # CallM(node_1) -> Del(inp) -> CallM(node_2) -> Del(node_1) -> Ret(node_2)
    assert len(instructions) == 5
    assert isinstance(instructions[0], CallModule)
    assert (
        isinstance(instructions[1], DeleteRegisterEntries) and instructions[1].targets[0] == inputs
    )
    assert isinstance(instructions[2], CallModule)
    assert (
        isinstance(instructions[3], DeleteRegisterEntries) and instructions[3].targets[0] == node_1
    )
    assert isinstance(instructions[4], ReturnOutputs)


def test_instruction_passes_no_passes() -> None:
    """Test that applying no passes leaves the program's instructions unchanged."""
    # GIVEN a sequential graph x -> node_1 -> node_2 -> out
    graph = GraphModule()
    inputs = graph.add_input("x")
    node_1 = graph.add_node("node_1", torch.nn.Identity(), [inputs])
    node_2 = graph.add_node("node_2", torch.nn.Identity(), [node_1])
    graph.add_output(node_2)

    program = InstructionScheduler().schedule(graph)

    # WHEN no passes are applied (passes defaulting to None)
    result = InstructionPasses.apply(program)

    # THEN the instructions are unchanged
    assert result.instructions == program.instructions


def test_local_error_multiple_contexts() -> None:
    """Test that context requirements propagate backward only to dependencies, not all nodes."""
    # GIVEN a graph with two branches (x -> node_1 -> node_2 -> out, x -> node_3 -> out)
    graph = GraphModule()
    inputs = graph.add_input("x")
    node_1 = graph.add_node("node_1", torch.nn.Identity(), [inputs])
    node_2 = graph.add_node("node_2", torch.nn.Identity(), [node_1])
    node_3 = graph.add_node("node_3", torch.nn.Identity(), [inputs])
    graph.add_output(node_2, node_3)

    # GIVEN node_2 requires two contexts for execution
    def opt_node_2(
        module: torch.nn.Module,
        default_bundle: ActivationBundle,
        quantized_bundle: ActivationBundle,
    ) -> None:
        pass

    class DummyQuantizedContext:
        def __enter__(self) -> None:
            return

        def __exit__(self, *args: Any) -> None:
            return

    null_context = nullcontext()
    quant_context = DummyQuantizedContext()

    delegate_2 = Delegate(fn=opt_node_2, contexts=[null_context, quant_context])
    graph._nodes[node_2.id] = dataclasses.replace(graph._nodes[node_2.id], delegate=delegate_2)

    # WHEN we schedule the graph
    scheduler = InstructionScheduler()
    instructions = scheduler.schedule(graph).instructions

    # THEN CallModule(node_1) should run in both contexts as it is required for OptimizeModule(node_2)
    # but the order in which they are defined does not matter.
    call_node_1 = next(i for i in instructions if isinstance(i, CallModule) and i.target == node_1)
    for context in call_node_1.contexts:
        assert context in {quant_context, null_context, DEFAULT_CONTEXT}

    # THEN OptimizeModule(node_2) should have both contexts in its delegate
    # and order DOES matter.
    opt_node_2_instr = next(
        i
        for i in instructions
        if isinstance(i, OptimizeModule) and i.module is graph._nodes[node_2.id].target
    )
    assert opt_node_2_instr.delegate.contexts == [null_context, quant_context]

    # THEN CallModule(node_2) should run in default context (forward pass always produces output).
    call_node_2 = next(i for i in instructions if isinstance(i, CallModule) and i.target == node_2)
    assert call_node_2.contexts == [DEFAULT_CONTEXT]

    # THEN node_3 should run in default context since it was not part of any optimize dependency
    call_node_3 = next(i for i in instructions if isinstance(i, CallModule) and i.target == node_3)
    assert call_node_3.contexts == [DEFAULT_CONTEXT]


def test_graph_execution_with_const_argument() -> None:
    """Test end-to-end execution of a graph with a Const argument triggers context extraction."""
    from fastforward._orchestration.graph_module import Const

    # GIVEN a graph with a module that takes both an input and a constant
    graph = GraphModule()
    inputs = graph.add_input("x")

    const_value = Const(10)
    node = graph.add_node("add_const", AddConstant(), [inputs, const_value])
    graph.add_output(node)

    # WHEN we execute the graph (which internally calls program.contexts)
    # THEN it should not raise AttributeError and should execute correctly
    result = graph(torch.tensor([1.0, 2.0, 3.0]))

    # Verify the result is correct
    expected = torch.tensor([11.0, 12.0, 13.0])
    assert torch.allclose(result, expected)


def test_move_parameters_moves_module_and_has_no_register_refs() -> None:
    # GIVEN a linear module on CPU
    module = torch.nn.Linear(5, 3)
    instruction = MoveModule(device=torch.device("cpu"), module=module)

    # WHEN we execute and inspect uses/produces
    register: ActivationRegister = {}
    instruction.execute(register)

    # THEN the module stays on CPU and the instruction touches no register refs
    assert module.weight.device == torch.device("cpu")
    assert module.bias.device == torch.device("cpu")
    assert list(instruction.uses()) == []
    assert list(instruction.produces()) == []


def test_move_activations_moves_register_entry_and_reports_ref() -> None:
    # GIVEN a register entry with a tensor dataset
    ref = NodeRef(id=uuid.uuid4(), name="ref")
    context = nullcontext()
    ds = ActivationDataset([torch.randn(2, 3)])
    register: ActivationRegister = {ref: {context: ds}}
    instruction = MoveActivations(device=torch.device("cpu"), register_ref=ref)

    # WHEN we execute and inspect uses/produces
    instruction.execute(register)

    # THEN the dataset is on CPU, uses() yields the ref, produces() is empty
    assert register[ref][context].batches[0].device == torch.device("cpu")
    assert list(instruction.uses()) == [ref]
    assert list(instruction.produces()) == []


def _make_optimizer_specs(model: Model) -> list[SubgraphSpec]:
    return [
        SubgraphSpec(
            region=model.residual_1.linear,
            fn=functools.partial(sgd_step, lr=0.1),
        )
    ]


def test_local_optimizer_with_offload_everything_only_updates_targeted_layer(model: Model) -> None:
    # GIVEN a model, graph, calibration data, and a spec targeting residual_1's linear layer
    graph = model.to_graph_module()
    initial_w1 = model.residual_1.linear.weight.data.clone()
    initial_w2 = model.residual_2.linear.weight.data.clone()
    calibration_data = [torch.randn(1, 5) for _ in range(10)]

    # WHEN we optimize with OffloadEverything
    offloading = OffloadEverything(
        compute_device=torch.device("cpu"),
        storage_device=torch.device("cpu"),
    )
    with local_optimize(graph, _make_optimizer_specs(model), offloading_strategy=offloading):
        graph(calibration_data)

    # THEN only residual_1's weights changed
    assert not torch.allclose(initial_w1, model.residual_1.linear.weight.data)
    assert torch.allclose(initial_w2, model.residual_2.linear.weight.data)


def test_local_optimizer_without_offloading_only_updates_targeted_layer(model: Model) -> None:
    # GIVEN a model, graph, and calibration data
    graph = model.to_graph_module()
    initial_w1 = model.residual_1.linear.weight.data.clone()
    initial_w2 = model.residual_2.linear.weight.data.clone()
    calibration_data = [torch.randn(1, 5) for _ in range(10)]

    # WHEN we optimize without an offloading strategy
    with local_optimize(graph, _make_optimizer_specs(model)):
        graph(calibration_data)

    # THEN only residual_1's weights changed
    assert not torch.allclose(initial_w1, model.residual_1.linear.weight.data)
    assert torch.allclose(initial_w2, model.residual_2.linear.weight.data)


def test_move_parameters_with_dict_moves_each_parameter_to_its_device() -> None:
    # GIVEN a linear module whose weight and bias are both on CPU
    module = torch.nn.Linear(4, 2)
    assert module.weight.device == torch.device("cpu")
    assert module.bias.device == torch.device("cpu")

    # WHEN we execute MoveParameters with a per-parameter dict
    device_map = {"weight": torch.device("cpu"), "bias": torch.device("cpu")}
    instruction = MoveModule(device=device_map, module=module)
    register: ActivationRegister = {}
    instruction.execute(register)

    # THEN each parameter is on its mapped device and the register is untouched
    assert module.weight.device == torch.device("cpu")
    assert module.bias.device == torch.device("cpu")
    assert register == {}


def test_move_parameters_with_dict_ignores_unmapped_parameters() -> None:
    # GIVEN a linear module with weight and bias
    module = torch.nn.Linear(4, 2)

    # WHEN we execute MoveParameters with a dict that only maps 'weight'
    device_map = {"weight": torch.device("cpu")}
    instruction = MoveModule(device=device_map, module=module)
    register: ActivationRegister = {}
    instruction.execute(register)

    # THEN weight is moved but bias is untouched (still on CPU in this case)
    assert module.weight.device == torch.device("cpu")
    assert module.bias.device == torch.device("cpu")


def test_weight_offloading_pass_post_restore_uses_per_parameter_devices() -> None:
    # GIVEN a graph with a single linear module
    module = torch.nn.Linear(4, 2)
    graph = GraphModule()
    inp = graph.add_input("x")
    graph.add_node("linear", module, [inp])

    scheduler = InstructionScheduler()
    base_instructions = scheduler.schedule(graph).instructions

    cpu = torch.device("cpu")

    # WHEN we apply the weight offloading pass
    result = _weight_offloading_pass(base_instructions, cpu, cpu, graph)

    # THEN the last instruction(s) are MoveParameters with a dict device (post_restore)
    post_restore = [i for i in result if isinstance(i, MoveModule) and isinstance(i.device, dict)]
    assert len(post_restore) == 1
    assert post_restore[0].module is module
    # AND the dict contains entries for both weight and bias
    device_map = post_restore[0].device
    assert isinstance(device_map, dict)
    assert "weight" in device_map
    assert "bias" in device_map


def test_cancel_pass_eliminates_redundant_moves_after_nn_module() -> None:
    # GIVEN an instruction stream where an nn.Module produces output on compute_device,
    # followed by an activation round-trip and a MoveModule round-trip
    compute = torch.device("cuda:0")
    storage = torch.device("cpu")

    linear = torch.nn.Linear(4, 4)
    linear_out = NodeRef(uuid.uuid4(), "linear_out")
    instructions = (
        CallModule(
            module=linear,
            args=(),
            kwargs={},
            target=linear_out,
            contexts=(nullcontext(),),
        ),
        MoveActivations(device=storage, register_ref=linear_out),
        MoveActivations(device=compute, register_ref=linear_out),
        MoveModule(device=storage, module=linear),
        MoveModule(device=compute, module=linear),
    )

    # WHEN the cancellation passes run
    result = _cancel_module_round_trips(instructions, compute, storage)
    result = _cancel_redundant_activation_moves(result, compute)

    # THEN the activation round-trip is eliminated (output already on compute)
    activation_moves = [i for i in result if isinstance(i, MoveActivations)]
    assert activation_moves == []

    # AND the adjacent MoveModule round-trip is eliminated
    module_moves = [i for i in result if isinstance(i, MoveModule)]
    assert module_moves == []


def test_cancel_pass_preserves_necessary_moves_for_non_module_callables() -> None:
    # GIVEN a mixed instruction stream:
    #   (a) an aten op with no inputs — output device is unknown
    #   (b) an aten op chain where inputs are on compute — state propagates
    compute = torch.device("cuda:0")
    storage = torch.device("cpu")

    # (a) aten op with no inputs: state unknown → compute move must survive
    arange_out = NodeRef(uuid.uuid4(), "arange_out")

    # (b) aten chain: src is on compute → both aten outputs are on compute → round-trips cancel
    src = NodeRef(uuid.uuid4(), "linear_out")
    aten_a_out = NodeRef(uuid.uuid4(), "aten_a_out")
    aten_b_out = NodeRef(uuid.uuid4(), "aten_b_out")

    instructions = (
        # (a) unknown producer
        CallModule(
            module=lambda: torch.arange(8),
            args=(),
            kwargs={},
            target=arange_out,
            contexts=(nullcontext(),),
        ),
        MoveActivations(device=storage, register_ref=arange_out),
        MoveActivations(device=compute, register_ref=arange_out),
        # (b) aten chain: src moved to compute by _weight_offloading_pass
        MoveActivations(device=compute, register_ref=src),
        CallModule(
            module=torch.transpose,
            args=(src,),
            kwargs={},
            target=aten_a_out,
            contexts=(nullcontext(),),
        ),
        MoveActivations(device=storage, register_ref=aten_a_out),
        MoveActivations(device=compute, register_ref=aten_a_out),
        CallModule(
            module=torch.transpose,
            args=(aten_a_out,),
            kwargs={},
            target=aten_b_out,
            contexts=(nullcontext(),),
        ),
        MoveActivations(device=storage, register_ref=aten_b_out),
        MoveActivations(device=compute, register_ref=aten_b_out),
    )

    # WHEN the cancellation pass runs
    result = _cancel_redundant_activation_moves(instructions, compute)

    # THEN the compute move for the unknown producer survives (can't prove it's on compute)
    arange_moves = [
        i for i in result if isinstance(i, MoveActivations) and i.register_ref == arange_out
    ]
    assert len(arange_moves) == 1
    assert arange_moves[0].device == compute

    # AND the aten chain's round-trips are all eliminated (state is known to be compute)
    chain_moves = [
        i
        for i in result
        if isinstance(i, MoveActivations) and i.register_ref in (aten_a_out, aten_b_out)
    ]
    assert chain_moves == []


def test_move_to_device_preserves_dict_subclass_type() -> None:
    """_move_to_device on a dict subclass must return an instance of the same subclass.

    HuggingFace ModelOutput (e.g. CausalLMOutputWithPast) inherits from OrderedDict.
    When MoveActivations moves the graph output to storage_device, it calls
    _move_to_device on this dict-like object. The `case dict()` branch used to
    reconstruct via a plain `{k: v}` comprehension, losing the custom type. This
    broke `out.logits` attribute access on the model output.
    """
    from collections import OrderedDict

    from fastforward._orchestration.instruction_engine import _move_to_device

    # GIVEN a dict subclass mimicking HuggingFace's ModelOutput pattern
    class FakeModelOutput(OrderedDict[str, object]):
        @property
        def logits(self) -> object:
            return self["logits"]

    output = FakeModelOutput(logits=torch.randn(1, 4, 8), past_key_values=None)

    # WHEN we move it to a device
    moved = _move_to_device(output, torch.device("cpu"))

    # THEN the type is preserved (not collapsed to plain dict)
    assert type(moved) is FakeModelOutput, (
        f"Expected FakeModelOutput, got {type(moved).__name__}. "
        f"_move_to_device must preserve dict subclass types."
    )
    # AND attribute access works
    assert hasattr(moved, "logits")
    assert moved["logits"].shape == (1, 4, 8)  # type: ignore[attr-defined]


def test_move_to_device_plain_dict_stays_plain_dict() -> None:
    """_move_to_device on a plain dict returns a plain dict (no regression)."""
    from fastforward._orchestration.instruction_engine import _move_to_device

    value = {"a": torch.randn(2, 3), "b": torch.randn(4)}
    moved = _move_to_device(value, torch.device("cpu"))

    assert type(moved) is dict
    assert moved["a"].shape == (2, 3)
    assert moved["b"].shape == (4,)


##########
# ActivationBundle
#
# `ActivationBundle` is the per-context call-construction view handed to a delegate.
# It carries positional and keyword `ActivationDataset`s and yields `(args, kwargs)`
# per batch.
##########


def test_activation_bundle_args_only_yields_args_tuple_and_empty_kwargs() -> None:
    # GIVEN a register populated with two positional refs under one context
    context = nullcontext()
    ref_a = NodeRef(id=uuid.uuid4(), name="a")
    ref_b = NodeRef(id=uuid.uuid4(), name="b")
    register: ActivationRegister = {
        ref_a: {context: ActivationDataset([1, 2, 3])},
        ref_b: {context: ActivationDataset([10, 20, 30])},
    }

    # WHEN we gather a bundle with two positional refs and no kwargs
    bundle = ActivationBundle.gather(register, context, args=[ref_a, ref_b], kwargs={})

    # THEN iteration yields `((args...,), {})` per batch, aligned across refs
    assert list(bundle) == [((1, 10), {}), ((2, 20), {}), ((3, 30), {})]


def test_activation_bundle_kwargs_only_yields_empty_args_and_kwargs_dict() -> None:
    # GIVEN refs populated under one context, surfaced only as kwargs on the bundle
    context = nullcontext()
    ref_x = NodeRef(id=uuid.uuid4(), name="x")
    ref_y = NodeRef(id=uuid.uuid4(), name="y")
    register: ActivationRegister = {
        ref_x: {context: ActivationDataset(["a", "b"])},
        ref_y: {context: ActivationDataset([100, 200])},
    }

    # WHEN we gather a bundle with no positional and two named kwargs
    bundle = ActivationBundle.gather(
        register, context, args=[], kwargs={"first": ref_x, "second": ref_y}
    )

    # THEN iteration yields `((), {name: value, ...})` per batch
    assert list(bundle) == [
        ((), {"first": "a", "second": 100}),
        ((), {"first": "b", "second": 200}),
    ]


def test_activation_bundle_mixed_args_and_kwargs_align_per_batch() -> None:
    # GIVEN a register with one positional ref and one kwarg ref
    context = nullcontext()
    ref_x = NodeRef(id=uuid.uuid4(), name="x")
    ref_mask = NodeRef(id=uuid.uuid4(), name="mask")
    register: ActivationRegister = {
        ref_x: {context: ActivationDataset([1, 2])},
        ref_mask: {context: ActivationDataset([True, False])},
    }

    # WHEN we bundle with mixed positional and keyword
    bundle = ActivationBundle.gather(register, context, args=[ref_x], kwargs={"mask": ref_mask})

    # THEN each batch packages args and kwargs as a single call snapshot
    assert list(bundle) == [((1,), {"mask": True}), ((2,), {"mask": False})]


def test_activation_bundle_const_refs_are_wrapped_inline() -> None:
    # GIVEN one register-backed positional ref alongside a Const used as kwarg
    context = nullcontext()
    ref_x = NodeRef(id=uuid.uuid4(), name="x")
    register: ActivationRegister = {ref_x: {context: ActivationDataset([1, 2])}}

    # WHEN we gather with a Const ref (which is not a register entry)
    bundle = ActivationBundle.gather(
        register, context, args=[ref_x], kwargs={"k": Const("constant")}
    )

    # THEN the Const is broadcast to match the positional length
    assert list(bundle) == [((1,), {"k": "constant"}), ((2,), {"k": "constant"})]


def test_activation_bundle_broadcasts_singletons_against_n_length_streams() -> None:
    # GIVEN one stream of length 1 and one of length 3 in the register
    context = nullcontext()
    ref_single = NodeRef(id=uuid.uuid4(), name="single")
    ref_multi = NodeRef(id=uuid.uuid4(), name="multi")
    register: ActivationRegister = {
        ref_single: {context: ActivationDataset(["S"])},
        ref_multi: {context: ActivationDataset([1, 2, 3])},
    }

    # WHEN we bundle them together
    bundle = ActivationBundle.gather(register, context, args=[ref_single, ref_multi], kwargs={})

    # THEN the singleton repeats across batches
    assert list(bundle) == [(("S", 1), {}), (("S", 2), {}), (("S", 3), {})]


def test_activation_bundle_length_mismatch_raises() -> None:
    # GIVEN two refs with incompatible lengths (no singleton to broadcast)
    context = nullcontext()
    ref_a = NodeRef(id=uuid.uuid4(), name="a")
    ref_b = NodeRef(id=uuid.uuid4(), name="b")
    register: ActivationRegister = {
        ref_a: {context: ActivationDataset([1, 2])},
        ref_b: {context: ActivationDataset([10, 20, 30])},
    }

    # WHEN/THEN gathering raises because broadcast cannot reconcile 2 != 3
    with pytest.raises(ValueError):
        ActivationBundle.gather(register, context, args=[ref_a, ref_b], kwargs={})


def test_activation_bundle_empty_inputs_has_zero_length() -> None:
    # GIVEN an empty register and no refs
    bundle = ActivationBundle.gather({}, nullcontext(), args=[], kwargs={})

    # THEN the bundle is empty and iterates to nothing
    assert list(bundle) == []


def test_optimize_module_passes_kwargs_to_delegate() -> None:
    # GIVEN a register holding one positional and two keyword activations under one context
    context = nullcontext()
    ref_hidden = NodeRef(id=uuid.uuid4(), name="hidden")
    ref_mask = NodeRef(id=uuid.uuid4(), name="mask")
    ref_pos = NodeRef(id=uuid.uuid4(), name="pos")
    register: ActivationRegister = {
        ref_hidden: {context: ActivationDataset([torch.tensor([1.0]), torch.tensor([2.0])])},
        ref_mask: {context: ActivationDataset([torch.tensor([True]), torch.tensor([False])])},
        ref_pos: {context: ActivationDataset([torch.tensor([0.1]), torch.tensor([0.2])])},
    }

    seen: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def delegate(_module: torch.nn.Module, bundle: ActivationBundle) -> None:
        seen.extend(list(bundle))

    instr = OptimizeModule(
        module=torch.nn.Identity(),
        args=[ref_hidden],
        kwargs={"attention_mask": ref_mask, "position_embeddings": ref_pos},
        delegate=Delegate(fn=delegate, contexts=[context]),
    )

    # WHEN the instruction executes
    instr.execute(register)

    # THEN the delegate sees both args and kwargs aligned per batch
    assert len(seen) == 2
    args0, kwargs0 = seen[0]
    args1, kwargs1 = seen[1]
    assert torch.equal(args0[0], torch.tensor([1.0]))
    assert torch.equal(args1[0], torch.tensor([2.0]))
    assert set(kwargs0.keys()) == {"attention_mask", "position_embeddings"}
    assert torch.equal(kwargs0["attention_mask"], torch.tensor([True]))
    assert torch.equal(kwargs1["position_embeddings"], torch.tensor([0.2]))


def test_optimize_module_uses_yields_kwarg_refs() -> None:
    # GIVEN an OptimizeModule with both positional and keyword refs
    ref_a = NodeRef(id=uuid.uuid4(), name="a")
    ref_b = NodeRef(id=uuid.uuid4(), name="b")
    ref_c = NodeRef(id=uuid.uuid4(), name="c")
    instr = OptimizeModule(
        module=torch.nn.Identity(),
        args=[ref_a],
        kwargs={"b": ref_b, "c": ref_c},
        delegate=Delegate(fn=lambda *_a, **_k: None, contexts=[nullcontext()]),
    )

    # WHEN we enumerate refs `uses()` reports
    used = list(instr.uses())

    # THEN every positional and keyword ref appears (offloading depends on this)
    assert ref_a in used
    assert ref_b in used
    assert ref_c in used


def test_optimize_module_delegate_receives_one_bundle_per_context() -> None:
    # GIVEN two contexts and one ref produced under each
    ctx_a = nullcontext()

    class _Ctx:
        def __enter__(self) -> None:
            return

        def __exit__(self, *args: Any) -> None:
            return

    ctx_b = _Ctx()
    ref_x = NodeRef(id=uuid.uuid4(), name="x")
    register: ActivationRegister = {
        ref_x: {
            ctx_a: ActivationDataset([1, 2]),
            ctx_b: ActivationDataset([10, 20]),
        },
    }

    received: list[ActivationBundle] = []

    def delegate(
        _module: torch.nn.Module, bundle_a: ActivationBundle, bundle_b: ActivationBundle
    ) -> None:
        received.extend([bundle_a, bundle_b])

    instr = OptimizeModule(
        module=torch.nn.Identity(),
        args=[ref_x],
        kwargs={},
        delegate=Delegate(fn=delegate, contexts=[ctx_a, ctx_b]),
    )

    # WHEN the instruction executes
    instr.execute(register)

    # THEN the delegate receives one bundle per context, in declared order
    assert len(received) == 2
    assert list(received[0]) == [((1,), {}), ((2,), {})]
    assert list(received[1]) == [((10,), {}), ((20,), {})]


def test_merge_empty_sequence_raises() -> None:
    # GIVEN no datasets to merge
    # WHEN merge is called with an empty sequence
    # THEN it raises rather than returning an empty/degenerate dataset
    with pytest.raises(ValueError, match="at least one dataset"):
        ActivationDataset.merge([])


def test_prepare_input_register_too_many_positionals_raises() -> None:
    # GIVEN a graph with a single input
    graph = GraphModule()
    graph.add_input("x")

    # WHEN we bind more positional args than there are inputs
    # THEN it raises on the positional-count mismatch
    with pytest.raises(TypeError, match="Expected 1 positional"):
        InstructionEngine.prepare_input_register(
            graph._inputs, args=(10, 20), kwargs={}, contexts=[nullcontext()]
        )


def test_schedule_ref_id_unknown_reference_type_raises() -> None:
    # GIVEN a _BaseRef subclass that is none of NodeRef/InputRef/Const/AttributeRef
    class _UnknownRef(_BaseRef):
        pass

    scheduler = InstructionScheduler()

    # WHEN the scheduler tries to resolve it
    # THEN the exhaustiveness guard rejects the unsupported reference type
    with pytest.raises(TypeError, match="Unsupported reference type"):
        scheduler._schedule_ref_id(_UnknownRef())


def test_load_attribute_extracts_dict_key_item() -> None:
    # GIVEN a register whose dataset batches are dicts (item access via [key])
    context = nullcontext()
    source = NodeRef(id=uuid.uuid4(), name="source")
    target = NodeRef(id=uuid.uuid4(), name="target")
    register: ActivationRegister = {
        source: {context: ActivationDataset([{"logits": 1}, {"logits": 2}])}
    }

    # WHEN we LoadAttribute a string key
    LoadAttribute(source=source, target=target, attribute="logits").execute(register)

    # THEN each batch is reduced to that key's value via item access
    assert register[target][context].batches == [1, 2]


def test_load_attribute_falls_back_to_getattr_for_objects() -> None:
    # GIVEN batches that are objects without __getitem__ (so batch["x"] raises TypeError)
    import dataclasses

    @dataclasses.dataclass
    class Out:
        logits: int

    context = nullcontext()
    source = NodeRef(id=uuid.uuid4(), name="source")
    target = NodeRef(id=uuid.uuid4(), name="target")
    register: ActivationRegister = {source: {context: ActivationDataset([Out(1), Out(2)])}}

    # WHEN we LoadAttribute a string key (item access fails, getattr fallback succeeds)
    LoadAttribute(source=source, target=target, attribute="logits").execute(register)

    # THEN the attribute is read via getattr on each batch
    assert register[target][context].batches == [1, 2]


def test_move_to_device_recurses_into_nested_tuples_and_lists() -> None:
    # GIVEN a value mixing nested tuples and lists of tensors (plus a non-tensor leaf)
    from fastforward._orchestration.instruction_engine import _move_to_device

    value = ([torch.randn(2, 3), (torch.randn(4), "scalar")], torch.randn(1))

    # WHEN we move it to a device
    moved = _move_to_device(value, torch.device("cpu"))

    # THEN container structure is preserved recursively and tensors are moved
    assert isinstance(moved, tuple)
    inner_list, outer_tensor = moved
    assert isinstance(inner_list, list)
    assert inner_list[0].device == torch.device("cpu")
    inner_tuple = inner_list[1]
    assert isinstance(inner_tuple, tuple)
    assert inner_tuple[0].device == torch.device("cpu")
    # THEN non-tensor leaves pass through untouched
    assert inner_tuple[1] == "scalar"
    assert outer_tensor.device == torch.device("cpu")


def test_activation_dataset_contains_checks_membership() -> None:
    # GIVEN a dataset of batches
    dataset = ActivationDataset([1, 2, 3])

    # WHEN/THEN membership is delegated to the underlying batches
    assert 2 in dataset
    assert 99 not in dataset


def test_activation_dataset_from_value_passthrough_when_already_dataset() -> None:
    # GIVEN an existing ActivationDataset
    existing = ActivationDataset([1, 2])

    # WHEN from_value is given a value that is already a dataset
    result = ActivationDataset.from_value(existing)

    # THEN it is returned unchanged (no re-wrapping)
    assert result is existing


def test_store_value_uses_reports_its_target() -> None:
    # GIVEN a StoreValue writing under a target ref
    target = NodeRef(id=uuid.uuid4(), name="t")
    instr = StoreValue(target=target, value=123, contexts=[nullcontext()])

    # WHEN we enumerate the refs it uses
    # THEN the target is reported (offloading liveness depends on this)
    assert list(instr.uses()) == [target]


def test_load_attribute_uses_and_produces_report_source_and_target() -> None:
    # GIVEN a LoadAttribute extracting from a source into a target
    source = NodeRef(id=uuid.uuid4(), name="src")
    target = NodeRef(id=uuid.uuid4(), name="dst")
    instr = LoadAttribute(source=source, target=target, attribute=0)

    # WHEN we enumerate its register interactions
    # THEN uses() reports the source and produces() reports the target
    assert list(instr.uses()) == [source]
    assert list(instr.produces()) == [target]
