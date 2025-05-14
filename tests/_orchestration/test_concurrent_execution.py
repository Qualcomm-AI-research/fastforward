# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import itertools
import unittest.mock

from typing import Any, Callable, NoReturn

import pytest

from fastforward._orchestration.concurrent_execution import ConcurrentExecOrchestrator


def _create_orchestrator(
    num_partitions: int, execution_order: list[tuple[int, ...]], run_list: list[tuple[int, ...]]
) -> ConcurrentExecOrchestrator:
    def target(batch_input: int) -> None:
        for partition in range(num_partitions - 1):
            run_list.append((partition, batch_input, orchestrator.stage))
            callback(batch_input)
        run_list.append((num_partitions - 1, batch_input, orchestrator.stage))

    num_stages = sum(len(exec_block) for exec_block in execution_order)
    orchestrator = ConcurrentExecOrchestrator(target, num_stages, execution_order=execution_order)

    def callback(idx: int) -> None:
        orchestrator.synchronize(idx)

    return orchestrator


@pytest.mark.slow
def test_orchestrator_ordered_run() -> None:
    run_list: list[tuple[int, ...]] = []
    num_partitions = 6
    num_inputs = 30
    execution_order = [(1, 2), (0,), (5,), (4, 3)]
    orchestrator = _create_orchestrator(num_partitions, execution_order, run_list)

    for batch_idx in range(num_inputs):
        orchestrator.add_batch(batch_idx)

    orchestrator.start()

    expected_run_list: list[tuple[int, int, int]] = []
    for partition in range(num_partitions):
        for exec_block in execution_order:
            for batch_idx in range(num_inputs):
                for stage in exec_block:
                    expected_run_list.append((partition, batch_idx, stage))

    assert expected_run_list == run_list


def test_orchestrator_repeated_stage() -> None:
    run_list: list[tuple[int, ...]] = []
    expected_run_list: list[tuple[int, ...]] = []
    num_inputs = 3
    execution_order = [(1,), (2,), (0,)]
    partition_2_steps = 3

    def target(batch_idx: int) -> None:
        partition_1(batch_idx)
        partition_2(batch_idx)
        partition_3(batch_idx)

    orchestrator = ConcurrentExecOrchestrator(target, num_stages=3, execution_order=execution_order)

    def partition_1(batch_idx: int) -> None:
        run_list.append((batch_idx, 1, orchestrator.stage, 0))
        orchestrator.synchronize(batch_idx)

    def partition_2(batch_idx: int) -> None:
        num_steps = partition_2_steps if orchestrator.stage == 2 else 1
        for step in range(num_steps):
            repeat = step != num_steps - 1
            run_list.append((batch_idx, 2, orchestrator.stage, step))
            orchestrator.synchronize(batch_idx, repeat_stage=repeat)

    def partition_3(batch_idx: int) -> None:
        run_list.append((batch_idx, 3, orchestrator.stage, 0))
        orchestrator.synchronize(batch_idx)

    for batch_idx in range(num_inputs):
        orchestrator.add_batch(batch_idx)

    orchestrator.start()

    stage_steps = [1, 1, partition_2_steps]
    for partition in range(1, 4):
        for stage in [1, 2, 0]:
            for step in range(stage_steps[stage] if partition == 2 else 1):
                for batch_idx in range(num_inputs):
                    expected_run_list.append((batch_idx, partition, stage, step))

    assert run_list == expected_run_list


def _hook_order_logger(hook_stage: str, logs: list[str]) -> Callable[..., None]:
    def hook_side_effect(
        orchestrator: ConcurrentExecOrchestrator, *_args: Any, **_kwargs: Any
    ) -> None:
        logs.append(f"{hook_stage}_{orchestrator.stage}")

    return hook_side_effect


def test_orchestrator_hooks() -> None:
    run_list: list[tuple[int, ...]] = []
    num_partitions = 2
    execution_order: list[tuple[int, ...]] = [(0, 1)]
    num_inputs = 5
    num_stages = sum(len(exec_block) for exec_block in execution_order)
    orchestrator = _create_orchestrator(num_partitions, execution_order, run_list)

    pre_stage_hook = unittest.mock.MagicMock()
    post_stage_hook = unittest.mock.MagicMock()
    global_pre_stage_hook = unittest.mock.MagicMock()
    global_post_stage_hook = unittest.mock.MagicMock()

    for stage in range(num_stages):
        orchestrator.register_pre_stage_hook(stage, pre_stage_hook)
        orchestrator.register_post_stage_hook(stage, post_stage_hook)
        orchestrator.register_global_pre_stage_hook(stage, global_pre_stage_hook)
        orchestrator.register_global_post_stage_hook(stage, global_post_stage_hook)

    for idx in range(num_inputs):
        orchestrator.add_batch(idx)

    orchestrator.start()

    assert pre_stage_hook.call_count == num_partitions * num_inputs * num_stages
    assert post_stage_hook.call_count == num_partitions * num_inputs * num_stages
    assert global_pre_stage_hook.call_count == num_partitions * num_stages
    assert global_post_stage_hook.call_count == num_partitions * num_stages

    for idx_or_none in itertools.chain(range(num_inputs), [None]):
        pre_stage_hook.assert_any_call(orchestrator, idx_or_none)
        post_stage_hook.assert_any_call(orchestrator, idx_or_none)

    global_pre_stage_hook.assert_any_call(orchestrator)
    global_pre_stage_hook.assert_any_call(orchestrator)
    global_post_stage_hook.assert_any_call(orchestrator)
    global_post_stage_hook.assert_any_call(orchestrator)


def test_orchestrator_hook_order() -> None:
    run_list: list[tuple[int, ...]] = []
    num_partitions = 2
    execution_order: list[tuple[int, ...]] = [(0, 1)]
    num_inputs = 3
    num_stages = sum(len(exec_block) for exec_block in execution_order)
    orchestrator = _create_orchestrator(num_partitions, execution_order, run_list)

    hook_logs: list[str] = []
    pre_stage_hook = unittest.mock.MagicMock(side_effect=_hook_order_logger("pre_stage", hook_logs))
    post_stage_hook = unittest.mock.MagicMock(
        side_effect=_hook_order_logger("post_stage", hook_logs)
    )
    global_pre_stage_hook = unittest.mock.MagicMock(
        side_effect=_hook_order_logger("global_pre_stage", hook_logs)
    )
    global_post_stage_hook = unittest.mock.MagicMock(
        side_effect=_hook_order_logger("global_post_stage", hook_logs)
    )

    for stage in range(num_stages):
        orchestrator.register_pre_stage_hook(stage, pre_stage_hook)
        orchestrator.register_post_stage_hook(stage, post_stage_hook)
        orchestrator.register_global_pre_stage_hook(stage, global_pre_stage_hook)
        orchestrator.register_global_post_stage_hook(stage, global_post_stage_hook)

    for idx in range(num_inputs):
        orchestrator.add_batch(idx)

    orchestrator.start()

    assert pre_stage_hook.call_count == num_partitions * num_inputs * num_stages
    assert post_stage_hook.call_count == num_partitions * num_inputs * num_stages
    assert global_pre_stage_hook.call_count == num_partitions * num_stages
    assert global_post_stage_hook.call_count == num_partitions * num_stages

    for idx_or_none in itertools.chain(range(num_inputs), [None]):
        pre_stage_hook.assert_any_call(orchestrator, idx_or_none)
        post_stage_hook.assert_any_call(orchestrator, idx_or_none)

    global_pre_stage_hook.assert_any_call(orchestrator)
    global_pre_stage_hook.assert_any_call(orchestrator)
    global_post_stage_hook.assert_any_call(orchestrator)
    global_post_stage_hook.assert_any_call(orchestrator)

    # fmt: off
    expected_hook_logs = [
        "global_pre_stage_0", "pre_stage_0", "post_stage_0", "global_pre_stage_1", "pre_stage_1",
        "post_stage_1", "pre_stage_0", "post_stage_0", "pre_stage_1", "post_stage_1", "pre_stage_0",
        "post_stage_0", "global_post_stage_0", "pre_stage_1", "post_stage_1", "global_post_stage_1",
        "global_pre_stage_0", "pre_stage_0", "post_stage_0", "global_pre_stage_1", "pre_stage_1",
        "post_stage_1", "pre_stage_0", "post_stage_0", "pre_stage_1", "post_stage_1", "pre_stage_0",
        "post_stage_0", "global_post_stage_0", "pre_stage_1", "post_stage_1", "global_post_stage_1",
    ]
    # fmt: on

    assert hook_logs == expected_hook_logs


def test_orchestrator_thread_exception() -> None:
    run_list: list[tuple[int, ...]] = []
    num_partitions = 3
    execution_order: list[tuple[int, ...]] = [(0, 1), (2,)]
    num_inputs = 5
    orchestrator = _create_orchestrator(num_partitions, execution_order, run_list)

    def error_hook(*_args: Any, **_kwargs: Any) -> NoReturn:
        raise ValueError()

    orchestrator.register_global_post_stage_hook(1, error_hook)

    for idx in range(num_inputs):
        orchestrator.add_batch(idx)

    with pytest.raises(RuntimeError):
        orchestrator.start()
