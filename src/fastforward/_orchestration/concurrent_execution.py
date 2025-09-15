# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import functools
import itertools
import threading

from collections import OrderedDict
from typing import Any, Callable, NamedTuple, ParamSpec, Sequence, TypeAlias, TypeVar

from torch.utils.hooks import RemovableHandle

_T = TypeVar("_T")
_P = ParamSpec("_P")
_StageHook: TypeAlias = Callable[["ConcurrentExecOrchestrator", Any], None]
_GlobalStageHook: TypeAlias = Callable[["ConcurrentExecOrchestrator"], None]


def ensure_non_mainthread(func: Callable[_P, _T]) -> Callable[_P, _T]:
    """Check if `func` is called outside main thread.

    Raises an error if `func` is called on the main thread.

    Args:
        func: The function two wrap with a non-main thread check.

    Returns:
        wrapped `func`
    """

    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        if threading.current_thread() == threading.main_thread():
            module = func.__module__
            name = func.__qualname__
            if module is not None and module != "__builtin__":  # type: ignore[redundant-expr]
                name = module + "." + name
            msg = (
                f"{name} can only be called within an context. This may happen "
                "when a orchestration hook is not properly removed."
            )
            raise RuntimeError(msg)
        return func(*args, **kwargs)

    return wrapper


class _ThreadErrorInfo(NamedTuple):
    exc_value: BaseException
    batch: int
    stage: int


class _StopThreadExecution(Exception):
    pass


class ConcurrentExecOrchestrator:
    """Orchestrator to run target function repeatedly in a configurable order.

    Multiple execution for a single input can communicate between
    different executions. This can be useful to implement local empirical error
    minimizations between two different execution contexts.

    The orchestration occurs over two axes called batches and stages. Batches
    relate to different inputs whereas stages relate to multiple executions of
    the target function with the same input.

    Consider the following target function (here we assume a global orchestrator object)

        def target():
            print(f"part 1 batch: {orchestrator.batch}, stage: {orchestrator.stage}")
            orchestrator.synchronize()
            print(f"part 2 batch: {orchestrator.batch}, stage: {orchestrator.stage}")

    The order of execution for this target function depends on the number of
    stages and the execution order. In this example, we'll use 3 stages and 2
    inputs (batches) as indicated in the diagram below.

        stages:    0   1   2
                 +---+---+---+
        batch 0  | 0 | 1 | 2 |
                 +---+---+---+
        batch 1  | 3 | 4 | 5 |
                 +---+---+---+

    In total, the target function will execute 6 times in a concurrent fashion
    (not the be confused with parallel execution). Each time
    `ConcurrentExecOrchestrator.synchronize` is called, all executions will run
    up to that point and only continue after all executions have reached that
    point. The order of stages is determined by `execution_order` which is a
    sequence of int-tuples in which each stage must be mentioned exactly once.
    For example, if the execution order is [(0, 1), (2,)] the output is as
    follows:

        part1 batch: 0 stage: 0
        part1 batch: 0 stage: 1
        part1 batch: 1 stage: 0
        part1 batch: 1 stage: 1
        part1 batch: 0 stage: 2
        part1 batch: 1 stage: 2
        part2 batch: 0 stage: 0
        part2 batch: 0 stage: 1
        part2 batch: 1 stage: 0
        part2 batch: 1 stage: 1
        part2 batch: 0 stage: 2
        part2 batch: 1 stage: 2

    Notice how the first two stages are grouped by batch and the last stage is
    only executed once stage 0 and 1 are completed for all batches.

    This becomes mostly useful because the executions can communicate through
    the synchronize call and orchestrator hooks. In particular,
    `ConcurrentExecOrchestrator.synchronize` accepts an optional `data`
    argument. This data is stored in the `batch_data` attribute on the
    orchestrator and is accessible to all executions for a given batch. Note
    that this is thread local storage and can not be used from the main thread.
    The data stored in `batch_data` may be altered by any execution for the
    same batch. `synchronize` will return the updated value when the execution
    continues.

    Adding Batches
    ==============
    A batch is added using the `add_batch` function. The target function is
    invoked using the same arguments and keyword arguments as passed to
    `add_batch`. Note that for each call to `add_batch`, `num_stages` extra
    executions of the target function will get scheduled.

    Running the orchestrator
    ========================
    Once the input for each batch has been added through `add_batch`,
    `ConcurrentExecOrchestrator.start` is used to start the execution. This
    method will block the main thread and will only return after all executions
    have terminated.

    Error Handling
    ==============
    If and error occurs during one of the executions, all other executions are
    stopped. Once all executions have terminated a RuntimeError is raised on
    the main thread that references the caught exception in the execution.

    Hooks
    =====
    Hooks can be used to change the behaviour of `ConcurrentExecOrchestrator`.
    In particular there are pre and post 'stage' hooks and pre and post 'global
    stage' hooks. A stage hook is called either before (pre) or after (post) a
    the execution of a single stage. I.e., this hook is called
    `num_stages * #batches * #(calls to synchronize)` times. In contrast, the 'global stage'
    hooks the global_pre_stage hook is invoked once per stage between two
    synchronize calls. Consider the same target function as above and the following
    orchestrator setup.

        stages:    0   1
                 +---+---+   execution order = [(0, 1)]
        batch 0  | 0 | 1 |
                 +---+---+
        batch 1  | 3 | 4 |
                 +---+---+

    Hooks are invoked as follows (read as two columns):

            > global_pre_stage stage 0                > global_pre_stage stage 0
            > pre_stage stage 0                       > pre_stage stage 0
        part1 batch: 0 stage: 0                   part2 batch: 0 stage: 0
            > post_stage stage 0                      > post_stage stage 0
            > global_pre_stage stage 1                > global_pre_stage stage 1
            > pre_stage stage 1                       > pre_stage stage 1
        part1 batch: 0 stage: 1                   part2 batch: 0 stage: 1
            > post_stage stage 1                      > post_stage stage 1
            > pre_stage stage 0                       > pre_stage stage 0
        part1 batch: 1 stage: 0                   part2 batch: 1 stage: 0
            > post_stage stage 0                      > post_stage stage 0
            > global_post_stage stage 0               > global_post_stage stage 0
            > pre_stage stage 1                       > pre_stage stage 1
        part1 batch: 1 stage: 1                   part2 batch: 1 stage: 1
            > post_stage stage 1                      > post_stage stage 1
            > global_post_stage stage 1               > global_post_stage stage 1

    Args:
        target: Target function to run orchestrated
        num_stages: Number of stages
        execution_order: The execution order of
            stages. All stages 0-(num_stages-1) should occur exactly once.
    """

    def __init__(
        self,
        target: Callable[..., Any],
        num_stages: int,
        execution_order: Sequence[tuple[int, ...]],
    ) -> None:
        self.target = target
        self.num_batches = 0
        self.num_stages = num_stages
        self.execution_order = execution_order
        self._local = threading.local()
        self._thread_exc: _ThreadErrorInfo | None = None

        order_stages = sorted(itertools.chain(*execution_order))
        if list(range(num_stages)) != order_stages:
            msg = (
                f"Each stage 0-{num_stages - 1} must appear exactly once in execution_order. "
                f"Got {execution_order}."
            )
            raise ValueError(msg)

        self.pre_stage_hooks: list[OrderedDict[int, _StageHook]] = [
            OrderedDict() for _ in range(num_stages)
        ]
        self.post_stage_hooks: list[OrderedDict[int, _StageHook]] = [
            OrderedDict() for _ in range(num_stages)
        ]
        self.global_pre_stage_hooks: list[OrderedDict[int, _GlobalStageHook]] = [
            OrderedDict() for _ in range(num_stages)
        ]
        self.global_post_stage_hooks: list[OrderedDict[int, _GlobalStageHook]] = [
            OrderedDict() for _ in range(num_stages)
        ]

        self._threads: list[threading.Thread] = []
        self._events: list[threading.Event] = []
        self._batch_data: list[dict[int, Any]] = []

    def __getitem__(self, key: tuple[int, int]) -> threading.Event:
        return self._event_for(*key)

    def _event_for(self, batch: int, stage: int) -> threading.Event:
        idx = batch * self.num_stages + stage
        return self._events[idx]

    def _init_thread(self, batch: int, stage: int) -> None:
        self._local.batch = batch
        self._local.stage = stage
        self.synchronize(_init_sync=True)

    @property
    def _event(self) -> threading.Event:
        return self[self._local.batch, self.stage]

    @property
    @ensure_non_mainthread
    def stage(self) -> int:
        """Returns the stage index for the current execution.

        Raises a RuntimeError when called from the main thread.
        """
        return self._local.stage  # type: ignore[no-any-return]

    @property
    @ensure_non_mainthread
    def batch(self) -> int:
        """Returns the batch index for the current execution.

        Raises a RuntimeError when called from the main thread.
        """
        return self._local.batch  # type: ignore[no-any-return]

    @property
    @ensure_non_mainthread
    def batch_data(self) -> dict[int, Any]:
        """Returns batch_data for the current execution.

        batch_data is a dictionary that maps from stage index to recorded data.
        Note that there is no guarantee that recorded data exists for each
        stage. In that case, the value for a stage may be missing or None.

        Raises:
            `RuntimeError` when called from the main thread.
        """
        return self._batch_data[self._local.batch]  # type: ignore[no-any-return]

    def _next_event_index(self, repeat_event: bool = False) -> tuple[int, int]:
        batch = self._local.batch
        stage = self._local.stage

        # Some context: the orchestator uses an execution order, this is a list[tuple[int, ...]]
        # A single tuple in the execution order is referred to as execution block or exec_block.

        # Find the current execution block that the current stage is part of.
        exec_block_idx = 0
        for exec_block_idx in range(len(self.execution_order)):
            if stage in self.execution_order[exec_block_idx]:
                break
        else:
            msg = f"stage {stage} is not in execution_order {self.execution_order}"
            raise RuntimeError(msg)

        exec_block = self.execution_order[exec_block_idx]
        block_stage_idx = exec_block.index(stage)

        # If repeated event, we cycle through only batches and not stages. This is only allowed
        # if the current stage is part of an exec block with just that stage. In other cases
        # we end up with hard to reason about stage orders.
        if repeat_event:
            if len(exec_block) != 1:
                msg = (
                    "Repeated stages are only allowed for the single stage execution blocks. "
                    f"The current stage '{stage}' is stage {block_stage_idx + 1} of "
                    f"{len(exec_block)} in execution block '{exec_block}' of the provided "
                    f"execution order ({self.execution_order})."
                )
                raise RuntimeError(msg)

            # If current batch is the last batch, cycle back to first batch. Else choose next
            # batch
            if (batch + 1) < self.num_batches:
                return (batch + 1, stage)
            else:
                return (0, stage)

        # When not repeated, favor selecting the next stage in the execution block, if there is one
        if (block_stage_idx + 1) < len(exec_block):
            return (batch, exec_block[block_stage_idx + 1])

        # Otherwise, select the next batch, if there is one
        if (batch + 1) < self.num_batches:
            return (batch + 1, exec_block[0])

        # If we could not advance the current stage in the exec block or the current batch,
        # move to the next exec block (and cycle to the first if necessary).
        next_block_idx = (exec_block_idx + 1) % len(self.execution_order)
        return (0, self.execution_order[next_block_idx][0])

    def _next_event(self, repeat_event: bool = False) -> threading.Event:
        next_batch, next_stage = self._next_event_index(repeat_event)
        return self[next_batch, next_stage]

    def _pre_stage(self, stage: int, batch: int, batch_dict: dict[int, Any]) -> None:
        if batch == 0:
            for hook in reversed(self.global_pre_stage_hooks[stage].values()):
                hook(self)
        for stage_hook in reversed(self.pre_stage_hooks[stage].values()):
            stage_hook(self, batch_dict[stage])

    def _post_stage(self, stage: int, batch: int, batch_dict: dict[int, Any]) -> None:
        for stage_hook in self.post_stage_hooks[stage].values():
            stage_hook(self, batch_dict[stage])

        if batch == (self.num_batches - 1):
            for hook in self.global_post_stage_hooks[stage].values():
                hook(self)

    def _wait(self) -> None:
        self._event.wait()

        # Abandon all runner threads if an exception was caught on another
        # thread. _StopThreadExecution is caught in the thread main function
        # and further handling is performed there.
        if self._thread_exc is not None:
            raise _StopThreadExecution()

    def synchronize(
        self,
        data: Any | None = None,
        *,
        repeat_stage: bool = False,
        _wait: bool = True,
        _init_sync: bool = False,
    ) -> Any:
        """Synchronize execution runs.

        All execution runs will run up to the call site and wait for other
        executions to 'catch-up'.

        When an optional data argument is passed, it is stored in batch_data and made available
        to the other executions associated with the current batch. This value may be altered
        and is returned by this functions. This can be used to communicate between different
        executions for the same batch.

        Arguments:
            data: Data to be saved and shared with other executions
            repeat_stage: If `True`, repeaat the current stage instead of moving to the next.
            _wait: If `True`, wait to get control again before completing this
                function.
            _init_sync: If `True` this function is called as initial
                synchronize, this alters the behaviour.

        Returns:
            Any: The stored data, which may have been altered by other executions.
        """
        stage = self.stage
        batch = self.batch
        batch_dict = self.batch_data

        batch_dict[stage] = data

        if not _init_sync:
            self._post_stage(stage, batch, batch_dict)
            self._event.clear()
            self._next_event(repeat_event=repeat_stage).set()

        if _wait:
            self._wait()

        self._pre_stage(stage, batch, batch_dict)

        value = batch_dict[stage]
        del batch_dict[stage]
        return value

    def _mark_as_done(self, execute_hooks: bool = True) -> None:
        if execute_hooks:
            stage, batch = self.stage, self.batch
            batch_dict = self.batch_data
            batch_dict[stage] = None
            self._post_stage(stage, batch, batch_dict)
        self._event.clear()
        self._next_event().set()

    def register_post_stage_hook(self, stage: int, hook: _StageHook) -> RemovableHandle:
        """Register a post_stage hook. See class docstring for a more detailed discussion.

        Returns a RemovableHandle. The hook can be removed by invoking `remove` on this handle.

        Returns:
            RemovableHandle
        """
        handle = RemovableHandle(self.post_stage_hooks[stage])
        self.post_stage_hooks[stage][handle.id] = hook
        return handle

    def register_pre_stage_hook(self, stage: int, hook: _StageHook) -> RemovableHandle:
        """Register a pre_stage hook. See class docstring for a more detailed discussion.

        Returns a RemovableHandle. The hook can be removed by invoking `remove` on this handle.

        Returns:
            RemovableHandle
        """
        handle = RemovableHandle(self.pre_stage_hooks[stage])
        self.pre_stage_hooks[stage][handle.id] = hook
        return handle

    def register_global_post_stage_hook(
        self, stage: int, hook: _GlobalStageHook
    ) -> RemovableHandle:
        """Register a global_post_stage hook. See class docstring for a more detailed discussion.

        Returns a RemovableHandle. The hook can be removed by invoking `remove` on this handle.

        Returns:
            RemovableHandle
        """
        handle = RemovableHandle(self.global_post_stage_hooks[stage])
        self.global_post_stage_hooks[stage][handle.id] = hook
        return handle

    def register_global_pre_stage_hook(self, stage: int, hook: _GlobalStageHook) -> RemovableHandle:
        """Register a global_pre_stage hook. See class docstring for a more detailed discussion.

        Returns a RemovableHandle. The hook can be removed by invoking `remove` on this handle.

        Returns:
            RemovableHandle
        """
        handle = RemovableHandle(self.global_pre_stage_hooks[stage])
        self.global_pre_stage_hooks[stage][handle.id] = hook
        return handle

    def start(self) -> None:
        """Start orchestrated execution."""
        for thread in self._threads:
            thread.start()

        self._event_for(batch=0, stage=self.execution_order[0][0]).set()

        for thread in self._threads:
            thread.join()

        # 'reraise' caught exception on main thread
        if self._thread_exc is not None:
            batch, stage = self._thread_exc.batch, self._thread_exc.stage
            msg = (
                f"An uncaught error occurred for batch {batch} at stage {stage}. "
                "See the error report above for more information."
            )
            raise RuntimeError(msg) from self._thread_exc.exc_value

    def add_batch(self, *args: Any, **kwargs: Any) -> None:
        """Add input data to orchestration grid.

        See the class docstring for a more detailed discussion.

        All passed arguments and keyword arguments are passed to the target
        function on execution.
        """
        batch = self.num_batches
        for stage in range(self.num_stages):
            thread = threading.Thread(
                target=self._exec_main, args=(args, kwargs), kwargs={"batch": batch, "stage": stage}
            )
            self._threads.append(thread)
            self._events.append(threading.Event())

        self.num_batches += 1
        self._batch_data.append({})

    def _exec_main(
        self, args: tuple[Any, ...], kwargs: dict[str, Any], *, batch: int, stage: int
    ) -> None:
        try:
            self._init_thread(batch, stage)
            self.target(*args, **kwargs)
            self._mark_as_done()
        except _StopThreadExecution:
            self._mark_as_done(execute_hooks=False)
        except BaseException as e:
            self._thread_exc = _ThreadErrorInfo(e, self.batch, self.stage)
            self._mark_as_done(execute_hooks=False)
