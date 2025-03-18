# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import abc
import functools

from typing import Any, Callable, Generator, Protocol, TypeAlias

import torch

from torch.utils.hooks import RemovableHandle

from fastforward._orchestration.concurrent_execution import (
    ConcurrentExecOrchestrator as Orchestrator,
)
from fastforward._orchestration.concurrent_execution import (
    ensure_non_mainthread,
)

ExecGenFactory: TypeAlias = Callable[[], Generator[None, None, None]]


class _AbstractExecutionContext(abc.ABC):
    @abc.abstractmethod
    def enable(self) -> None: ...

    @abc.abstractmethod
    def disable(self) -> None: ...


class _GeneratorExecContext(_AbstractExecutionContext):
    def __init__(self, generator_factory: ExecGenFactory):
        self.factory = generator_factory
        self._active_generator: Generator[None, None, None] | None = None

    def enable(self) -> None:
        if self._active_generator is not None:
            raise RuntimeError("Execution context is already active and is not re-entrant")
        self._active_generator = self.factory()
        try:
            next(self._active_generator)
            return None
        except StopIteration:
            raise RuntimeError("Execution context generator did not yield")

    def disable(self) -> None:
        if self._active_generator is None:
            raise RuntimeError("Execution context is not active and cannot be exited")
        try:
            current_gen = self._active_generator
            self._active_generator = None
            next(current_gen)
        except StopIteration:
            return
        else:
            raise RuntimeError("Execution context generator did not stop")


def _register_context_hooks(
    exec_context: _AbstractExecutionContext, orchestrator: Orchestrator, stage: int
) -> list[RemovableHandle]:
    def _enable_context_hook(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        exec_context.enable()

    def _disable_context_hook(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        exec_context.disable()

    handles: list[RemovableHandle] = []
    handles.append(orchestrator.register_pre_stage_hook(stage, _enable_context_hook))
    handles.append(orchestrator.register_post_stage_hook(stage, _disable_context_hook))
    return handles


def execution_context(
    func: Callable[..., Generator[None, None, None]],
) -> Callable[[], _GeneratorExecContext]:
    """Decorator to create an ExecutorContext.

    Returns:
        _GeneratorExecContext that wraps func
    """

    @functools.wraps(func)
    def helper(self: Any) -> _GeneratorExecContext:
        return _GeneratorExecContext(func.__get__(self))

    return helper  # type: ignore[return-value]


class LocalErrorMethod(abc.ABC):
    """Abstract base class for local error methods.

    A local error method minimizes errors for sub networks. Specifically, it
    aims to minimize the error between a 'default' or reference forward pass
    and an alternative forward pass. An example of this a floating point
    forward pass and a quantized forward pass of a subset of the layers in a
    neural network.
    """

    @abc.abstractmethod
    def prepare(self, ctx: "RunnerContext") -> None:
        """Prepare local error method with the given context."""

    @abc.abstractmethod
    def cleanup(self) -> None:
        """Clean up resources used by the local error method."""

    def default_context(self) -> _AbstractExecutionContext | None:
        """Return the default execution context."""
        return None

    def alternative_context(self) -> _AbstractExecutionContext | None:
        """Return the alternative execution context."""
        return None

    @abc.abstractmethod
    def update(self, __default_value: torch.Tensor, __alternative_value: torch.Tensor, /) -> None:
        """Perform update step.

        Perform update step given the value obtain from the default context
        and the value obtained from the alternative context.
        """

    @abc.abstractmethod
    def propagate(
        self, __replay_value: torch.Tensor, /
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Propagate data from replay stage to default and alternative stage.

        Given the value obtained in the replay stage, generate new activation
        value for default and alternative stage. Return None to 'propagate' the
        actual values from the default and alternative stage.
        """

    @abc.abstractmethod
    def conclude_partition(self) -> None:
        """Conclude the evaluation and update of a partition of the network.

        This is called after all stages for a particular partition have been concluded.
        """


class RunnerContext(Protocol):
    """Protocol for RunnerContext.

    Defines methods that are exposed to implementors of the LocalErrorMethod
    interface through a context object.
    """

    def communicate(self, value: torch.Tensor) -> torch.Tensor:
        """Communicate values to other stages.

        Communicate a value obtained during the execution of a partition in a
        stage to the runner. This enables 'communication' between different
        execution contexts.

        Args:
            value: The tensor value to communicate to the runner and share with
                other contexts.
        """
        raise NotImplementedError


class Runner:
    """Runner class to manage execution stages and contexts."""

    DEFAULT_STAGE = 0
    ALTERNATIVE_STAGE = 1
    REPLAY_STAGE = 2

    _orchestrator: Orchestrator

    def __init__(self, target: Callable[..., Any], method: LocalErrorMethod):
        """Initialize the runner.

        Args:
            target: The target function to run.
            method: The method to manage errors.
        """
        self._method = method
        self._setup_orchestrator(target)

    def _setup_orchestrator(self, target: Callable[..., Any]) -> None:
        """Set up the orchestrator with the target function.

        Args:
            target: The target function to run.
        """
        exec_order: list[tuple[int, ...]] = [
            (self.DEFAULT_STAGE, self.ALTERNATIVE_STAGE),
            (self.REPLAY_STAGE,),
        ]
        self._orchestrator = Orchestrator(target=target, num_stages=3, execution_order=exec_order)

        self._orchestrator.register_global_post_stage_hook(
            self.ALTERNATIVE_STAGE, self._partition_complete
        )

    def _partition_complete(self, *args: Any, **kwargs: Any) -> None:
        """Notify underlying local error method of the partition complete event."""
        del args, kwargs
        self._method.conclude_partition()

    def record_input(self, *args: Any, **kwargs: Any) -> None:
        """Record input data for the orchestrator.

        Args:
            *args: Positional arguments for the input data.
            **kwargs: Keyword arguments for the input data.
        """
        self._orchestrator.add_batch(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        """Record input data by calling the runner."""
        self.record_input(*args, **kwargs)

    def start(self) -> None:
        """Start the runner and manage execution contexts."""
        # Register context hooks

        handles = []
        orchestrator = self._orchestrator
        if (default_ctx := self._method.default_context()) is not None:
            handles += _register_context_hooks(default_ctx, orchestrator, self.DEFAULT_STAGE)
            handles += _register_context_hooks(default_ctx, orchestrator, self.REPLAY_STAGE)

        if (alt_ctx := self._method.alternative_context()) is not None:
            handles += _register_context_hooks(alt_ctx, orchestrator, self.ALTERNATIVE_STAGE)

        self._method.prepare(self)
        try:
            self._orchestrator.start()
        finally:
            self._method.cleanup()
            for handle in handles:
                handle.remove()

    @ensure_non_mainthread
    def communicate(self, value: torch.Tensor) -> torch.Tensor:
        """Communicate a value within the runner's context.

        Args:
            value: The value to communicate.

        Returns:
            torch.Tensor: The communicated value.
        """
        if self._orchestrator.stage == self.ALTERNATIVE_STAGE:
            default_value = self._orchestrator.batch_data[self.DEFAULT_STAGE]
            self._method.update(default_value, value)
        elif self._orchestrator.stage == self.REPLAY_STAGE:
            default_value, alternative_value = self._method.propagate(value)
            batch_data = self._orchestrator.batch_data

            if default_value is not None:
                batch_data[self.DEFAULT_STAGE] = default_value
            if alternative_value is not None:
                batch_data[self.ALTERNATIVE_STAGE] = alternative_value

        return self._orchestrator.synchronize(value)  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        """Return a string representation of the runner."""
        return f"{type(self).__name__}(method={repr(self._method)})"
